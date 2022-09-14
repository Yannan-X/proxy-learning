from re import I
import torch.nn as nn
from torch import Tensor
import torch
from tqdm import tqdm
from sinabs.from_torch import from_model
from eventcim import from_dynapcnn_model, Network
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork, DynapcnnNetwork
from aermanager.datasets import SpikeTrainDataset, FramesDataset
from eventcim.events import Event2d, Address2d
from typing import Callable, List, Tuple
import numpy as np
from typing import Union, Tuple
from eventcim.network import Network
from sinabs.layers import IAFSqueeze
from random import shuffle, seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# ydRGBenv python env
batch_size = 4
localtime = time.asctime(time.localtime(time.time()))
tensorboard_dir = f"./tensorboardRC/{localtime}/"
save_file_name = "simulator-proxy-WC"
writer = SummaryWriter(f"{tensorboard_dir}{save_file_name}")

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
        # load a pre-trained rate-based model
        self.ann.load_state_dict(torch.load("/home/yannan/PycharmProjects/proxy-learning/Proxy-learning/nmnist/cnn_10epoch.pth"), strict=False)
        self.snn = self._initialize_simulation_network()

        self.reshape_size = [
            0,
            0,
            (8, 16, 3, 3),
            (256, 8, 4, 4),
            (10, 256, 1, 1)
        ]
        
        self.ann_activation_record = []
        self.snn.set_batch_size(1)
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
        # generate a parameter discretized network
        quantized_snn = DynapcnnNetwork(snn, discretize=True, input_shape=(2, 34, 34))
        # generate simulator object
        snn = from_dynapcnn_model(quantized_snn)
        return snn

    def _update_snn(self) -> None:
        # This function does the simulator parameter update according to the ann. This should be done on the fly
        # when training with the proxy. The updated parameters are exactly same as the the one discretized using dynapcnnnetwork.
        # Todo: update the parameters upon any structure, now it is still architechture dependent
        quantized_weight = []
        quantized_threshold = []

        for ly_index, layer_weights in enumerate(list(self.ann.parameters())):
            layer_weights_clone = layer_weights.data.clone() 
            if ly_index in [2, 3]:
                layer_weights_clone /= 4 
                layer_weights_clone = layer_weights_clone.data.reshape(self.reshape_size[ly_index])
            elif ly_index == 4:
                layer_weights_clone = layer_weights_clone.data.reshape(self.reshape_size[ly_index])
            scale, thresholds = self.obtain_scale_factor_of_a_layer(layer_weights_clone)
            quantized_weight.append(torch.round(layer_weights_clone.data * scale))
            quantized_threshold.append(tuple(map(lambda x: round(x * scale), (-1, 1))))

        # python signature of set simulator weight
        self.snn.set_weight(quantized_weight)
        # python signature of set simulator threshold
        self.snn.set_threshold(quantized_threshold)

    @staticmethod
    def _reset_snn_state(spiking_net) -> None:
        for lyrs in spiking_net:
            if isinstance(lyrs, IAFSqueeze):
                lyrs.reset_states(randomize=True)

    def forward(self, frame: np.ndarray, xytp: np.ndarray) -> Tuple[Tensor, Tensor]:
        #tstart = time.time()
        # forward pass of ann
        # frame = torch.tensor(frame).sum(0).unsqueeze(0).unsqueeze(0)
        frame = torch.tensor(frame).unsqueeze(0)
        ann_output = self.ann(frame)
        # convert xytp to events
        events = self._xytp_to_events(xytp)

        # simulator forward pass
        self._update_snn()
        output_event, output_list_eachlayer = self.snn.forward([events])

        # output evnents --> Tensor
        snn_output = self._collect_output_events_from_simulator(
            output_event[0], number_of_class=10
        )

        ann_activation = sum(self.ann_activation_record)
        snn_activation = sum(output_list_eachlayer)

        self.ann_activation_record = []
        
        return ann_output, snn_output, ann_activation, snn_activation

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
        """_summary_

        Args:
            weight (_type_): weight tensor
            spike_thresh (int, optional): upper spiking threshold. Defaults to 1.
            low_thresh (int, optional): low spike boundary. Defaults to -1.
            w_precision (int, optional): target weight precision in bit. Defaults to 8.
            state_precision (int, optional): target state precision in bit. Defaults to 16.
            device (_type_, optional): the device of the tensor working on. Defaults to torch.device("cpu").

        Returns:
            scale: The obtained scale factor
            thresholds: tuple of (spike upper and spike_low threshold)
        """

        w_factor = self._get_quantization_scale(weight.data, bit_precision=w_precision)
        thresholds = torch.tensor((low_thresh, spike_thresh)).to(device)
        t_factor = self._get_quantization_scale(
            thresholds, bit_precision=state_precision
        )
        scale = min(w_factor, t_factor)

        return scale.item(), thresholds


def get_number_of_correct_samples(network_output: Tensor, label: int) -> int:
    # calculate correct prediction using output tensor and label tensor
    
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
optimizer = torch.optim.SGD(model.ann.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Generating the random reading sequence for frame and xytp datasets
assert len(train_xytp) == len(train_frame)
reading_sequence = list(range(len(train_xytp)))
seed(24)
shuffle(reading_sequence)


class Multi_process_batching:

    @staticmethod
    def generate_batch_based_reading_sequence(reading_sequences: List[int], batch: int) -> List[List[int]]:
        """
        reading_sequences: input list of reading indexes
        batch: batch size

        Return: lists of minibatch sample reading indexes
        """
        new_batch_sequence = []
        assert len(reading_sequences) // batch >= 1
        mini_batch_index = []
        for indexing in range(len(reading_sequences) // batch):
            if len(mini_batch_index) < batch:
                mini_batch_index.append(reading_sequences[indexing])
            else:
                new_batch_sequence.append(mini_batch_index)
                mini_batch_index = [reading_sequences[indexing]]
        return new_batch_sequence
    
    @staticmethod
    def get_batch_frame(dataset: Callable, index:list) -> Tensor:
        """get a batch frame with indexing, dataset should return a sample shape
        of (1, y, x)

        Args:
            dataset (Callable): frame dataset
            index (list): list of index, len(index)>1
        """
        
        frames = torch.empty(0)
        targets = []
        for i in index:
            frame, target = dataset[i]
            
            frames = torch.cat([frames, torch.tensor(frame).unsqueeze(0)], 0)
            targets.append(target)
        
        targets = torch.tensor(targets).view(-1)
        
        return frames, targets
    
    @staticmethod
    def get_batch_xytp(dataset:Callable, index:list):
        """
        get a batch of events and labels that compatiable to eventcimulator

        Args:
            dataset (Callable): callable xytp dataset
            index (list): the batch index, len(index)>1
        """
        xytps = []
        targets = []
        
        for i in index: 
            
            sample, target = dataset[i]
            xytps.append(_xytp_to_events(sample))
            targets.append(target)
        targets = torch.tensor(targets).view(-1)
        return xytps, targets
            
def _xytp_to_events(sample: np.ndarray) -> List[Event2d]:
    events = [Event2d(Address2d(int(ch), y, x), t) for (x, y, t, ch) in sample]
    return events
        
    
batching = Multi_process_batching()

multi_process_reading_index = batching.generate_batch_based_reading_sequence(reading_sequences=reading_sequence, batch=2)


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

        out_ann, out_simulator, ann_activation, snn_activation = model.forward(frame=frame_sample, xytp=xytp_sample)

        writer.add_scalar("train/ann_output_sum", out_ann.sum(), global_step=step)
        writer.add_scalar("train/sim_output_sum", out_simulator.sum(), global_step=step)
        writer.add_scalar("train/ann_activations_sum", ann_activation, global_step=step)
        writer.add_scalar("train/snn_activation_sum", snn_activation, global_step=step)
        
        correct_ann += get_number_of_correct_samples(out_ann, frame_target)
        correct_simulator += get_number_of_correct_samples(
            out_simulator.view(1, -1), xytp_target
        )
        number_of_samples += 1

        out_ann.data.copy_(out_simulator)
        ann_activation.copy_(torch.tensor(snn_activation))
        
        target_activation = 1e4
        proxy_activation_loss = 10 * (1 / ann_activation ** 2) * torch.sqrt((ann_activation - target_activation) ** 2)

        if target_activation > ann_activation:
            loss = criterion(out_ann, frame_target)
        else:
            loss = criterion(out_ann, frame_target) + proxy_activation_loss
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        
        writer.add_scalar("train/proxy_activation_loss", proxy_activation_loss, global_step=step)
        
        
        # for layer_index, ls in enumerate(model.ann):
        #     if isinstance(ls, (nn.Conv2d, nn.Linear)):
        #         writer.add_scalar(
        #             f"lyr{layer_index}/grad_max",
        #             ls.weight.grad.data.max(),
        #             global_step=step,
        #         )
        #         writer.add_scalar(
        #             f"lyr{layer_index}/grad_min",
        #             ls.weight.grad.data.min(),
        #             global_step=step,
        #         )
        #         writer.add_scalar(
        #             f"lyr{layer_index}/grad_average",
        #             ls.weight.grad.data.mean(),
        #             global_step=step,
        #         )
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

        writer.add_scalar("train/loss", loss.item(), global_step=step)
        writer.add_scalar("train/activation_loss", proxy_activation_loss, global_step=step)
        # writer.add_scalar("train/input_count", len(xytp_sample), global_step=step)

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
