import torch.nn as nn
from torch import Tensor
import torch
from tqdm import tqdm
from sinabs.from_torch import from_model
from eventcim import from_dynapcnn_model, Network
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork, DynapcnnNetwork
from aermanager.datasets import SpikeTrainDataset, FramesDataset
from eventcim.events import Event2d, Address2d
from typing import List, Tuple
import numpy as np
from typing import Union, Tuple
from eventcim.network import Network
from sinabs.layers import IAFSqueeze
from random import shuffle, seed
from torch.utils.tensorboard import SummaryWriter
import time

# ydRGBenv python env

localtime = time.asctime(time.localtime(time.time()))
tensorboard_dir = f"./tensorboardRC/{localtime}/"
save_file_name = "simulator-proxy-WC"
writer = SummaryWriter(f"{tensorboard_dir}{save_file_name}")

number_of_epochs = 50

train_data_dir = "../nmnist/train_set_time50/"
test_data_dir = "../nmnist/test_set_time50/"

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

        self.ann.load_state_dict(torch.load("../nmnist/cnn_10epoch.pth"), strict=False)
        self.snn, self.quantized_snn = self._initialize_simulation_network()

    def _initialize_simulation_network(self) -> Tuple[Network, DynapcnnNetwork]:
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
        return snn, quantized_snn

    def _update_snn(self) -> Network:

        lyr_counter = -1
        for lyrs in self.ann:
            if isinstance(lyrs, (nn.Conv2d, nn.Linear)):
                lyr_counter += 1
                if lyr_counter in [2, 3]:
                    # downsampling weight to adpat pooling layer
                    ann_weight = lyrs.weight.clone() / 4
                else:
                    ann_weight = lyrs.weight.clone()
                scale, thresh = self.obtain_scale_factor_of_a_layer(weight=ann_weight)
                if isinstance(lyrs, nn.Conv2d):
                    weight_data = torch.round(ann_weight * scale).int()
                else:
                    in_chan, in_h, in_w = self.quantized_snn.sequence[lyr_counter].input_shape
                    weight_data = ann_weight.detach().reshape((lyrs.out_features, in_chan, in_h, in_w))
                    weight_data = torch.round(weight_data * scale).int()

                self.quantized_snn.sequence[
                    lyr_counter
                ].conv_layer.weight.data = weight_data

                (
                    self.quantized_snn.sequence[lyr_counter].spk_layer.min_v_mem,
                    self.quantized_snn.sequence[lyr_counter].spk_layer.spike_threshold,
                ) = torch.round(torch.tensor(thresh) * scale).int()

        snn = from_dynapcnn_model(self.quantized_snn)
        return snn

    @staticmethod
    def _reset_snn_state(spiking_net) -> None:
        for lyrs in spiking_net:
            if isinstance(lyrs, IAFSqueeze):
                lyrs.reset_states(randomize=True)

    def forward(self, frame: np.ndarray, xytp: np.ndarray) -> Tuple[Tensor, Tensor]:

        # forward pass of ann
        # frame = torch.tensor(frame).sum(0).unsqueeze(0).unsqueeze(0)
        frame = torch.tensor(frame).unsqueeze(0)
        ann_output = self.ann(frame)
        # convert xytp to events
        events = self._xytp_to_events(xytp)

        # simulator forward pass

        snn = self._update_snn()
        output_event = snn.forward(events)
        # output_event = snn.forward_each_layer(events)
        # output evnents --> Tensor
        snn_output = self._collect_output_events_from_simulator(
            output_event, number_of_class=10
        )

        return ann_output, snn_output

    def weight_initializer(self) -> None:
        for layer in self.ann:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

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
    ) -> [float, Tuple[int, int]]:

        w_factor = self._get_quantization_scale(weight.data, bit_precision=w_precision)
        thresholds = torch.tensor((low_thresh, spike_thresh)).to(device)
        t_factor = self._get_quantization_scale(
            thresholds, bit_precision=state_precision
        )
        scale = min(w_factor, t_factor)

        return scale, thresholds


def get_number_of_correct_samples(network_output: Tensor, label: int) -> int:
    _, predict = torch.max(network_output, 1)
    correct = (predict == label).sum().item()

    return correct


def weight_clipping(models: nn.Sequential, mins=-1, maxs=1):
    # clipping the weight to be symmetric
    for lyrs in models:
        if isinstance(lyrs, nn.Conv2d) or isinstance(lyrs, nn.Linear):
            lyrs.weight.data.clamp_(min=mins, max=maxs)


# model initialization and optimizer define
model = ProxyAnnSnn()
# optimizer = torch.optim.Adam(model.ann.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.ann.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Generating the random reading sequence for frame and xytp datasets
assert len(train_xytp) == len(train_frame)
reading_sequence = list(range(len(train_xytp)))
seed(24)
shuffle(reading_sequence)

for epochs in range(number_of_epochs):

    # recording usage
    correct_ann = 0
    correct_simulator = 0
    number_of_samples = 0
    model.train()
    running_loss = 0

    pbr = tqdm(reading_sequence)

    for step, read_index in enumerate(pbr):
        frame_sample, frame_target = train_frame[read_index]
        xytp_sample, xytp_target = train_xytp[read_index]

        frame_target = torch.tensor(frame_target).view(-1)
        xytp_target = torch.tensor(xytp_target).view(-1)

        assert frame_target == xytp_target, "frame target and xytp target is not same!"

        out_ann, out_simulator = model.forward(frame=frame_sample, xytp=xytp_sample)

        writer.add_scalar("train/ann_output_sum", out_ann.sum(), global_step=step)
        writer.add_scalar("train/sim_output_sum", out_simulator.sum(), global_step=step)

        correct_ann += get_number_of_correct_samples(out_ann, frame_target)
        correct_simulator += get_number_of_correct_samples(
            out_simulator.view(1, -1), xytp_target
        )
        number_of_samples += 1

        out_ann.data.copy_(out_simulator)

        target_output = 50
        sim_out_count = out_simulator.sum()
        out_loss = (1 / sim_out_count ** 2) * np.sqrt(
            (sim_out_count - target_output) ** 2
        )
        loss = criterion(out_ann, frame_target)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()

        for layer_index, ls in enumerate(model.ann):
            if isinstance(ls, (nn.Conv2d, nn.Linear)):
                writer.add_scalar(
                    f"lyr{layer_index}/grad_max",
                    ls.weight.grad.data.max(),
                    global_step=step,
                )
                writer.add_scalar(
                    f"lyr{layer_index}/grad_min",
                    ls.weight.grad.data.min(),
                    global_step=step,
                )
                writer.add_scalar(
                    f"lyr{layer_index}/grad_average",
                    ls.weight.grad.data.mean(),
                    global_step=step,
                )
        optimizer.step()

        #
        weight_clipping(model.ann)

        writer.add_scalar(
            "train/acc_ann", round(correct_ann / number_of_samples, 4), global_step=step
        )
        writer.add_scalar(
            "train/acc_sim",
            round(correct_simulator / number_of_samples, 4),
            global_step=step,
        )

        writer.add_scalar("train/out_loss", out_loss, global_step=step)
        writer.add_scalar("train/input_count", len(xytp_sample), global_step=step)

        for layer_index, ls in enumerate(model.ann):
            if isinstance(ls, (nn.Conv2d, nn.Linear)):
                writer.add_scalar(
                    f"lyr{layer_index}/weight_max",
                    ls.weight.data.max(),
                    global_step=step,
                )
                writer.add_scalar(
                    f"lyr{layer_index}/weight_min",
                    ls.weight.data.min(),
                    global_step=step,
                )
                writer.add_scalar(
                    f"lyr{layer_index}/weight_average",
                    ls.weight.data.mean(),
                    global_step=step,
                )

        pbr.set_description(
            f"train_ann:{round(correct_ann / number_of_samples, 4)}, "
            f"train_sim{round(correct_simulator / number_of_samples, 4)}",
        )
