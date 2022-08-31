import torch.nn as nn
from torch import Tensor
import torch
from tqdm import tqdm
from sinabs.from_torch import from_model
from eventcim import from_dynapcnn_model, Network
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork, DynapcnnNetwork
from aermanager.datasets import SpikeTrainDataset, FramesDataset
from eventcim.events import Event2d, Address2d
from typing import List, Tuple, Any, Union
import numpy as np
from typing import Union, Tuple
from eventcim.network import Network
from sinabs.layers import IAFSqueeze
from random import shuffle, seed
from torch.utils.tensorboard import SummaryWriter
import time
from multiprocessing import Pool

# ydRGBenv python env

# localtime = time.asctime(time.localtime(time.time()))
# tensorboard_dir = f"./tensorboardRC/{localtime}/"
# save_file_name = "simulator-proxy-WC"
# writer = SummaryWriter(f"{tensorboard_dir}{save_file_name}")

number_of_epochs = 50

train_data_dir = "/home/yannan/PycharmProjects/proxy-learning/Proxy-learning/nmnist/train_set_time50"
test_data_dir = "/home/yannan/PycharmProjects/proxy-learning/Proxy-learning/nmnist/test_set_time50"

train_xytp = SpikeTrainDataset(train_data_dir, target_transform=int)
train_frame = FramesDataset(train_data_dir, target_transform=int)


class ProxyAnnSnn(nn.Module):
    def __init__(self):
        super(ProxyAnnSnn, self).__init__()
        # define the ann_model
        self.ann = nn.Sequential(
            nn.Conv2d(
                2, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False
            ),  # 16, 18, 18
            nn.ReLU(),
            nn.Conv2d(
                16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),  # 8, 18,18
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 8, 17,17
            nn.Conv2d(
                16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),  # 8, 9, 9
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(4 * 4 * 8, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 10, bias=False),
            nn.ReLU(),
        )

        self.ann.load_state_dict(
            torch.load("/home/yannan/PycharmProjects/proxy-learning/Proxy-learning/nmnist/cnn_10epoch.pth"),
            strict=False)
        self.snn = self._initialize_simulation_network()
        self.device = torch.device("cpu")
        self.reshape_size = [
            0,
            0,
            (8, 16, 3, 3),
            (256, 8, 4, 4),
            (10, 256, 1, 1)
        ]
        self.ann_activation_record = []
        self.ann.to(self.device)
        for index, layer in enumerate(self.ann):
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(self.get_features(index))

    def _initialize_simulation_network(self) -> Network:
        # converting the ann to the equivalent snn and deploy the simulator configuration
        snn = from_model(
            nn.Sequential(
                *self.ann[:-1],
                IAFSqueeze(batch_size=1, spike_threshold=1, min_v_mem=-1),
            ),
            input_shape=(2, 34, 34),
        ).spiking_model
        quantized_snn = DynapcnnNetwork(snn, discretize=True, input_shape=(2, 34, 34))
        snn = from_dynapcnn_model(quantized_snn)
        return snn

    def _update_snn(self) -> None:

        quantized_weight = []
        quantized_threshold = []

        for ly_index, layer_weights in enumerate(list(self.ann.parameters())):
            layer_weights_clone = layer_weights.data.clone().detach().cpu()
            if ly_index in [2, 3]:
                layer_weights_clone /= 4
                layer_weights_clone = layer_weights_clone.data.reshape(self.reshape_size[ly_index])
            elif ly_index == 4:
                layer_weights_clone = layer_weights_clone.data.reshape(self.reshape_size[ly_index])
            scale, thresholds = self.obtain_scale_factor_of_a_layer(layer_weights_clone)
            quantized_weight.append(torch.round(layer_weights_clone.data * scale))
            quantized_threshold.append(tuple(map(lambda x: round(x * scale), (-1, 1))))

        self.snn.set_weight(quantized_weight)
        self.snn.set_threshold(quantized_threshold)

    @staticmethod
    def _reset_snn_state(spiking_net) -> None:
        for lyrs in spiking_net:
            if isinstance(lyrs, IAFSqueeze):
                lyrs.reset_states(randomize=True)

    def forward(self, frame: np.ndarray, xytp: np.ndarray) -> Tuple[Any, Tensor, Union[int, Any], Union[int, Any]]:

        # forward pass of ann
        # frame = torch.tensor(frame).sum(0).unsqueeze(0).unsqueeze(0)
        frame = torch.tensor(frame).unsqueeze(0)
        ann_output = self.ann(frame)
        # convert xytp to events
        events = self._xytp_to_events(xytp)

        # simulator forward pass
        self._update_snn()
        output_event, output_list_eachlayer = self.snn.forward(events)
        # output_event = snn.forward_each_layer(events)
        # output evnents --> Tensor
        snn_output = self._collect_output_events_from_simulator(
            output_event, number_of_class=10
        )

        ann_activation = sum(self.ann_activation_record)
        snn_activation = sum(output_list_eachlayer)

        self.ann_activation_record = []

        return ann_output, snn_output, ann_activation, snn_activation

    def xytp_simulator_forward(self, xytp: np.ndarray):
        events = self._xytp_to_events(xytp)
        output_event, output_list_eachlayer = self.snn.forward(events)
        return (output_event, output_list_eachlayer)

    def frame_torch_forward(self, frame: np.ndarray):
        frame = frame.to(self.device)
        ann_output = self.ann(frame)
        return ann_output

    def weight_initializer(self) -> None:
        for layer in self.ann:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def get_features(self, name):
        def hook(model, input, output):
            self.ann_activation_record.append(output.sum())

        return hook

    @staticmethod
    def _xytp_to_events(sample: np.ndarray) -> List[Event2d]:
        events = [Event2d(Address2d(int(ch), y, x), t) for (x, y, t, ch) in sample]
        return events

    @staticmethod
    def _collect_output_events_from_simulator(
            output_event_stream: List[Event2d], number_of_class: int
    ) -> Tensor:
        counts = torch.zeros(number_of_class)
        for item in output_event_stream:
            counts[int(str(item).split("\t")[0].split(":")[-1])] += 1
        return counts

    @staticmethod
    def _get_quantization_scale(input_tensor, bit_precision=8):
        """
        :param input_tensor: the input tensor that need to be quantized
        :param precision: quantization precision
        :return: The quantized tensor
        """

        min_val_disc = -(2 ** (bit_precision - 1))
        max_val_disc = 2 ** (bit_precision - 1) - 1

        # Range in which values lie
        min_val_obj = torch.min(input_tensor)
        max_val_obj = torch.max(input_tensor)

        # Determine if negative or positive values are to be considered for scaling
        # Take into account that range for diescrete negative values is slightly larger than for positive
        min_max_ratio_disc = abs(min_val_disc / max_val_disc)
        if abs(min_val_obj) <= abs(max_val_obj) * min_max_ratio_disc:
            scaling = abs(max_val_disc / max_val_obj)
        else:
            scaling = abs(min_val_disc / min_val_obj)

        return scaling

    def obtain_scale_factor_of_a_layer(
            self,
            weight,
            spike_thresh=1,
            low_thresh=-1,
            w_precision=8,
            state_precision=16,
            device=torch.device("cpu"),
    ):

        w_factor = self._get_quantization_scale(weight.data, bit_precision=w_precision)
        thresholds = torch.tensor((low_thresh, spike_thresh)).to(device)
        t_factor = self._get_quantization_scale(
            thresholds, bit_precision=state_precision
        )
        scale = min(w_factor, t_factor)

        return scale.item(), thresholds


def get_number_of_correct_samples(network_output: Tensor, label: int) -> int:
    _, predict = torch.max(network_output, 1)
    correct = (predict.cpu().detach() == label.cpu().detach()).sum().item()

    return correct


def weight_clipping(models: nn.Sequential, mins=-1, maxs=1):
    # clipping the weight to be symmetric
    for lyrs in models:
        if isinstance(lyrs, nn.Conv2d) or isinstance(lyrs, nn.Linear):
            lyrs.weight.data.clamp_(min=mins, max=maxs)


# model initialization and optimizer define
model = ProxyAnnSnn()
# optimizer = torch.optim.Adam(model.ann.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.ann.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()

# Generating the random reading sequence for frame and xytp datasets
assert len(train_xytp) == len(train_frame)
reading_sequence = list(range(len(train_xytp)))
seed(24)
shuffle(reading_sequence)

p = Pool(2)


def run(xytp_sample):
    # frame_sample, frame_target = train_frame[data_index]

    output_event, output_list = model.xytp_simulator_forward(xytp=xytp_sample)
    output_event = model._collect_output_events_from_simulator(output_event, number_of_class=10)
    return (output_event, output_list)

# l1 = []

# if __name__ == "__main__":
#
#     for i in reading_sequence[1:3]:
#         xytp_sample, xytp_target = train_xytp[i]
#         output = p.apply_async(run, args=(xytp_sample,))
#         l1.append(output)
#
#     p.close()
#     p.join()
#
#     for res in l1:
#         print(res.get())