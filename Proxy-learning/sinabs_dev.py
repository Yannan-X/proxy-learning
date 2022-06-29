import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from sinabs.layers import IAFSqueeze
from sinabs.activation import MembraneSubtract
import sys
import time
import numpy as np
from tqdm import tqdm
from transforms import *

batch_size = 500
learning_rate = 1e-4
T = 15
train_epoch = 100


dataset_dir = "./"
test_list_transforms = [
        transforms.ToTensor(),
    ]

train_list_transforms = [
    transforms.RandomCrop(26),
    transforms.Pad(1),
    transforms.ToTensor(),
]

train_transform = transforms.Compose(train_list_transforms)
test_transform = transforms.Compose(test_list_transforms)

train_data_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.FashionMNIST(
        root=dataset_dir,
        train=True,
        transform=train_transform,
        download=True),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True)

test_data_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.FashionMNIST(
        root=dataset_dir,
        train=False,
        transform=test_transform,
        download=True),
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128 * 4 * 5, bias=False),
            nn.ReLU(),
            nn.Linear(128 * 4 * 5, 128 * 3 * 3, bias=False),
            nn.ReLU(),
            nn.Linear(128 * 3 * 3, 128 * 2 * 1, bias=False),
            nn.ReLU (),
            nn.Linear(128 * 2 * 1, 10, bias=False),
        )


    def forward(self, x):
        return self.fc(self.conv(self.static_conv(x)))

img_current =ImageAndCurrent(time_steps=15)

class SNN(nn.Module):
    def __init__(self, v_threshold=2.0, v_reset=0.0):
        super().__init__()
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            IAFSqueeze(batch_size=batch_size, spike_threshold=v_threshold,reset_fn=MembraneSubtract()),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            IAFSqueeze(batch_size=batch_size, spike_threshold=v_threshold,reset_fn=MembraneSubtract()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            IAFSqueeze(batch_size=batch_size, spike_threshold=v_threshold,reset_fn=MembraneSubtract()),
            nn.MaxPool2d(2, 2),  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128 * 4 * 5, bias=False),
            IAFSqueeze(batch_size=batch_size, spike_threshold=v_threshold,reset_fn=MembraneSubtract()),
            nn.Linear(128 * 4 * 5, 128 * 3 * 3, bias=False),
            IAFSqueeze(batch_size=batch_size, spike_threshold=v_threshold,reset_fn=MembraneSubtract()),
            nn.Linear(128 * 3 * 3, 128 * 2 * 1, bias=False),
            IAFSqueeze(batch_size=batch_size, spike_threshold=v_threshold,reset_fn=MembraneSubtract()),
            nn.Linear(128 * 2 * 1, 10, bias=False),
            IAFSqueeze(batch_size=batch_size, spike_threshold=v_threshold,reset_fn=MembraneSubtract()),
        )

    def forward(self, data):
        # x = self.static_conv(x)
        # out_spikes_counter = self.fc(self.conv(x))
        # for t in range(1, self.T):
        #     if (t==0):
        #         out_spikes_counter = self.fc(self.conv(x))
        #     else:
        #         out_spikes_counter += self.fc(self.conv(x))
        _, data = img_current(data)
        (batch_size, t_len, channel, height, width) = data.shape
        samplein = data.reshape((batch_size * t_len, channel, height, width))

        out_spikes_counter = self.static_conv(samplein)
        out_spikes_counter = self.fc(self.conv(out_spikes_counter))
        out_spikes_counter = out_spikes_counter.reshape(batch_size, t_len, 10)
        out_spikes_counter = out_spikes_counter.sum(1)

        return out_spikes_counter

    def reset_states(self):

        for lyrs in self.conv:
            if isinstance(lyrs, IAFSqueeze):
                lyrs.reset_states(randomize=True)

        for lyrs in self.fc:
            if isinstance(lyrs, IAFSqueeze):
                lyrs.reset_states(randomize=True)

device = torch.device("cuda")
ann = ANN().to(device)
snn = SNN().to(device)

params_ann = ann.named_parameters()
params_snn = snn.named_parameters()
dict_params_snn = dict(params_snn)
for name, param in params_ann:
    if name in dict_params_snn:
        dict_params_snn[name].data = param.data

optimizer_ann = torch.optim.Adam(ann.parameters(), lr=learning_rate, betas=(0.8, 0.99), eps=1e-08,
                                     weight_decay=1e-06)



for epoch in range(train_epoch):
    ann.train()
    snn.train()
    if epoch >= 1:
        for m in ann.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in snn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    correct_ann = 0
    correct_snn = 0
    sample_num = 0
    for img, label in tqdm(train_data_loader, position=0):
        img = img.to(device)
        label = label.to(device)
        label_one_hot = F.one_hot(label, 10).float()
        optimizer_ann.zero_grad()
        outputs = ann(img)
        with torch.no_grad():
            out_spikes_counter = snn(img)

        predict_ann = outputs.argmax(1)
        correct_ann += (predict_ann == label).sum().item()
        sample_num += label.numel()
        correct_snn += (out_spikes_counter.argmax(1) == label).float().sum().item()
        outputs.data.copy_(out_spikes_counter)
        loss = F.mse_loss(outputs, label_one_hot)
        loss.backward()
        optimizer_ann.step()




        snn.reset_states()

    acc_ann = correct_ann / sample_num
    acc_snn = correct_snn / sample_num
    print(f'epoch={epoch}, train_ann={acc_ann}, train_snn={acc_snn}')
    torch.save(ann.state_dict(), "ann_weights_trained.pth")
    torch.save(snn.state_dict(), "snn_weights_trained.pth")
    torch.save(snn, "snn_model.pt")
    torch.save(ann, "ann_model.pt")

    ann.eval()
    snn.eval()
    with torch.no_grad():
        correct_snn = 0
        correct_ann = 0
        sample_num = 0
        for img, label in tqdm(test_data_loader, position=0):
            img = img.to(device)
            label = label.to(device)
            predict_ann = ann(img)
            correct_ann += (predict_ann.argmax(1) == label).sum().item()
            sample_num += label.numel()
            out_spikes_counter_frequency = snn(img)
            correct_snn += (out_spikes_counter_frequency.argmax(1) == label).sum()
            snn.reset_states()

        acc_ann = correct_ann / sample_num
        acc_snn = correct_snn / sample_num
        print(f'epoch={epoch}, test_ann={acc_ann}, test_snn={acc_snn}')

with torch.no_grad():
    correct_snn = 0
    correct_ann = 0
    sample_num = 0
    for img, label in tqdm(test_data_loader, position=0):
        img = img.to(device)
        label = label.to(device)
        predict_ann = ann(img)
        correct_ann += (predict_ann.argmax(1) == label).sum().item()
        sample_num += label.numel()
        out_spikes_counter_frequency = snn(img)
        correct_snn += (out_spikes_counter_frequency.argmax(1) == label).sum()
        snn.reset_states()
    acc_ann = correct_ann / sample_num
    acc_snn = correct_snn / sample_num
    print(f' Final Result: Acc_ANN={acc_ann}, Acc_SNN={acc_snn}')
