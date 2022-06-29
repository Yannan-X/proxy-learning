from aermanager.datasets import FramesDataset,SpikeTrainDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from sinabs.from_torch import from_model
import copy

epochs = 50
batchsize = 500
save_file_name = "ann50"
tensorboard_dir = "./tensorboardRC/"
torch.manual_seed(24)


train_raster = SpikeTrainDataset(dt=1000, source_folder="./train_set_time50", target_transform=int,
                                 force_n_time_bins=50)
test_raster = SpikeTrainDataset(dt=1000, source_folder="./test_set_time50", target_transform=int, force_n_time_bins=50)

train_loader = DataLoader(train_raster, batch_size=batchsize, shuffle=True, drop_last=True,
                          num_workers=64)
test_loader = DataLoader(test_raster, batch_size=batchsize, shuffle=True, drop_last=True,
                         num_workers=64)

# for samples, target in testset:
#     print(samples.shape)
#     plt.imshow(samples.sum(0))
#     plt.show()
#     print(target)

ann = nn.Sequential(

    nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False),  # 16, 18, 18
    nn.ReLU(),

    nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 18,18
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),  # 8, 17,17

    nn.Dropout2d(0.5),
    nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 9, 9
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),

    nn.Flatten(),
    nn.Linear(4 * 4 * 8, 256, bias=False),
    nn.ReLU(),

    nn.Linear(256, 10, bias=False),
    nn.ReLU(),

)

def raster_forward(model, sample):
    (batch_size, t_len, channel, height, width) = sample.shape
    sample = sample.reshape((batch_size * t_len, channel, height, width))
    out = model(sample)
    out = out.reshape(batch_size, t_len, 10)
    out = out.sum(1)

    return out

device = torch.device("cuda:1")
ann.load_state_dict(torch.load("ann50_best.pth"))
ann = ann.to(device=device)

snn = from_model(ann, batch_size=batchsize).spiking_model
snn[0].weight.data *= 2

num_samples = 0
correct_samples = 0
ann.eval()
with torch.no_grad():
    pbb = tqdm(test_loader)
    for sample, target in pbb:
        sample = sample.clone().sum(2).unsqueeze(2)
        sample = sample.float()
        sample = sample.to(device)
        target = target.long().to(device)
        out = raster_forward(snn, sample)
        _, predict = torch.max(out, 1)
        correct_samples += (predict == target).sum().item()
        num_samples += batchsize
        pbb.set_description(f"test_acc: {round(100 * correct_samples / num_samples, 2)}")
