from aermanager.datasets import FramesDataset, SpikeTrainDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

class ANN(torch.nn.Module):

    def __init__(self):
        super(ANN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.fc1 = nn.Linear(4 * 4 * 8, 256, bias=False)
        self.fc2 = nn.Linear(256, 10, bias=False)

        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()

    def forward(self, sample):

        propgate = self.relu(self.conv1(sample))
        propgate = self.pool2(self.relu(self.conv2(propgate)))
        propgate = self.pool2(self.relu(self.conv3(propgate)))
        propgate = propgate.view(-1, 4 * 4 * 8)
        propgate = self.relu(self.fc1(propgate))
        propgate = F.leaky_relu(self.fc2(propgate))

        return propgate


epochs = 10
batchsize = 500


train_frame = FramesDataset(source_folder="../../datasets/train_set_time50", target_transform=int)
train_loader = DataLoader(train_frame, batch_size=batchsize, num_workers=24, shuffle=True)
device = torch.device("cuda")



criterion = nn.CrossEntropyLoss()
ann = ANN().to(device)
optimizer = torch.optim.Adam(ann.parameters(), lr=1e-3, amsgrad=True)

for epo in range(epochs):
    num_samples = 0
    correct_ann = 0
    pbr = tqdm(train_loader)
    for sample, target in pbr:
        # print(sample.shape)
        sample_ann = sample.sum(1).unsqueeze(1)
        sample_ann = sample_ann.to(device)
        target = target.to(device)
        out_ann = ann(sample_ann)
        _, ann_predict = torch.max(out_ann, 1)

        correct_ann += (ann_predict == target).sum().item()
        num_samples += batchsize
        loss = criterion(out_ann, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbr.set_description(f"accuracy: {round(correct_ann/num_samples, 4)}")

    torch.save(ann.state_dict(), "cnn_10epoch.pth")