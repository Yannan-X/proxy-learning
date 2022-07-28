import torch.nn as nn
from torch import Tensor
import torch
from tqdm import tqdm
from sinabs.from_torch import from_model
from eventcim import from_dynapcnn_model
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork, DynapcnnNetwork
from aermanager.datasets import SpikeTrainDataset, FramesDataset
from eventcim.events import Event2d, Address2d
from typing import List
import numpy as np
from typing import Union, Tuple
from eventcim.network import Network
from sinabs.layers import IAFSqueeze
from random import shuffle, seed

# ydRGBenv python env

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
            nn.LeakyReLU()
        )
        self.weight_initializer()
        self.ann.load_state_dict(torch.load("../nmnist/cnn_10epoch.pth"))
        self.ann[0].weight.data *= 10

    def _generate_simulation_network(self) -> Network:
        # converting the ann to the equivalent snn and deploy the simulator configuration
        snn = from_model(nn.Sequential(*self.ann[:-1], IAFSqueeze(batch_size=1, spike_threshold=1, min_v_mem=-1)), input_shape=(2, 34, 34)).spiking_model
        # snn = from_model(self.ann, input_shape=(2, 34, 34)).spiking_model
        self._reset_snn_state(snn)
        snn = DynapcnnNetwork(snn, discretize=True, input_shape=(2, 34, 34))
        snn = from_dynapcnn_model(snn)
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
        snn = self._generate_simulation_network()

        output_event = snn.forward(events)
        # output evnents --> Tensor
        snn_output = self._collect_output_events_from_simulator(output_event, number_of_class=10)

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
    def _collect_output_events_from_simulator(output_event_stream: List[Event2d], number_of_class: int) -> Tensor:
        counts = torch.zeros(number_of_class)
        for item in output_event_stream:
            counts[int(str(item).split("\t")[0].split(":")[-1])] += 1
        return counts


def get_number_of_correct_samples(network_output: Tensor, label: int) -> int:
    _, predict = torch.max(network_output, 1)
    correct = (predict == label).sum().item()

    return correct


# model initialization and optimizer define
model = ProxyAnnSnn()
optimizer = torch.optim.Adam(model.ann.parameters(), lr=1e-3)
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

    for read_index in pbr:
        frame_sample, frame_target = train_frame[read_index]
        xytp_sample, xytp_target = train_xytp[read_index]

        frame_target = torch.tensor(frame_target).view(-1)
        xytp_target = torch.tensor(xytp_target).view(-1)

        assert frame_target == xytp_target, "frame target and xytp target is not same!"

        out_ann, out_simulator = model.forward(frame=frame_sample, xytp=xytp_sample)
        print(f"{out_ann}, f{out_simulator}")
        correct_ann += get_number_of_correct_samples(out_ann, frame_target)
        correct_simulator += get_number_of_correct_samples(out_simulator.view(1, -1), xytp_target)
        number_of_samples += 1

        out_ann.data.copy_(out_simulator)
        loss = criterion(out_ann, frame_target)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbr.set_description(f"train_ann:{round(correct_ann / number_of_samples, 4)}, "
                            f"train_sim{round(correct_simulator / number_of_samples, 4)}", )
        print(f"sumout:{out_simulator.sum()}")
