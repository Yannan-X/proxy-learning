from aermanager.datasets import FramesDataset, SpikeTrainDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sinabs.layers import IAFSqueeze
from sinabs.activation import PeriodicExponential, MembraneReset, MembraneSubtract
from sinabs.from_torch import from_model
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork
import torch.nn.functional as F

epochs = 100
batchsize = 500
save_file_name = "bptt50"
tensorboard_dir = "./tensorboardRC/"
localtime = time.asctime(time.localtime(time.time()))
writer = SummaryWriter(f"{tensorboard_dir}{save_file_name}-batch{batchsize}-epoch{epochs}-{localtime}")

torch.manual_seed(24)

# trainset = FramesDataset(source_folder= "./train_set_time50", target_transform=int)
# testset  = FramesDataset(source_folder="./test_set_time50", target_transform=int)

train_raster = SpikeTrainDataset(dt=1000, source_folder="./train_set_time50", target_transform=int,
                                 force_n_time_bins=50)
test_raster = SpikeTrainDataset(dt=1000, source_folder="./test_set_time50", target_transform=int, force_n_time_bins=50)

train_loader = DataLoader(train_raster, batch_size=batchsize, shuffle=True, drop_last=True,
                          num_workers=64)
test_loader = DataLoader(test_raster, batch_size=batchsize, shuffle=True, drop_last=True,
                         num_workers=64)

# for samples, target in train_raster:
#     print(samples.shape)
#     plt.imshow(samples.sum(0).sum(0))
#     plt.show()
#     print(target)

ann = nn.Sequential(

    nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False),  # 16, 18, 18
    nn.ReLU(),

    # nn.Dropout2d(0.5),
    nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 18,18
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),  # 8, 17,17

    # nn.Dropout2d(0.5),
    nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 9, 9
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),

    nn.Flatten(),
    # nn.Dropout2d(0.5),
    nn.Linear(4 * 4 * 8, 256, bias=False),
    nn.ReLU(),

    nn.Linear(256, 10, bias=False),
    # nn.ReLU(),

)


snn = from_model(ann, batch_size=batchsize, surrogate_grad_fn=PeriodicExponential(),
                 reset_fn=MembraneSubtract(), spike_threshold=1).spiking_model
print(snn)


def _reset_states(model):
    for lyrs in model:
        if isinstance(lyrs, IAFSqueeze):
            lyrs.reset_states(randomize=True)


def _detach(model):
    for lyrs in model:
        if isinstance(lyrs, IAFSqueeze):
            lyrs.v_mem.detach_()


def raster_forward(model, sample):
    (batch_size, t_len, channel, height, width) = sample.shape
    sample = sample.reshape((batch_size * t_len, channel, height, width))
    out = model(sample)
    out = out.reshape(batch_size, t_len, 10)
    out = out.sum(1)

    return out


def weight_initializer(model):
    for layer in model:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)

weight_initializer(snn)
snn[0].weight.data *=5
device = torch.device("cuda")
# optimizer = torch.optim.Adam(ann.parameters(), lr=1e-3, weight_decay=1e-6)

snn = snn.to(device)

C = nn.CrossEntropyLoss()
best_acc = 0





optimizer = torch.optim.Adam(snn.parameters(), lr=1e-3, weight_decay=1e-06)

for epoch in range(epochs):

    num_samples = 0
    correct_ann = 0
    correct_snn = 0

    if epoch == 2:
        #     ann = nn.Sequential(*ann, nn.ReLU())
        snn = nn.Sequential(*snn, IAFSqueeze(batch_size=batchsize, surrogate_grad_fn=PeriodicExponential(),
                 reset_fn=MembraneSubtract(), spike_threshold=1, min_v_mem=-1))
        snn = snn.to(device)
    #     ann = ann.to(device)

    snn.train()
    pbr = tqdm(train_loader)

    for sample, target in pbr:
        sample = sample.to(device)
        snn_sample = sample.clone().sum(2).unsqueeze(2)
        target = target.long().to(device)
        out_snn = raster_forward(snn, snn_sample)
        _, snn_predict = torch.max(out_snn, 1)
        correct_snn += (snn_predict == target).sum().item()

        num_samples += batchsize
        loss = C(out_snn, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _detach(snn)

        pbr.set_description(
            f"epoch:{epoch}, train_snnacc: {round(100 * correct_snn / num_samples, 2)}")

    writer.add_scalar("train_acc_snn", 100 * correct_snn / num_samples)
    torch.save(snn.state_dict(), f"{save_file_name}.pth")

    # testing
    num_samples = 0
    correct_ann = 0
    correct_snn = 0
    snn.eval()

    with torch.no_grad():
        pbb = tqdm(test_loader)
        for sample, target in pbb:
            sample = sample.to(device)
            snn_sample = sample.clone().sum(2).unsqueeze(2)
            target = target.long().to(device)
            out_snn = raster_forward(snn, snn_sample)

            _, snn_predict = torch.max(out_snn, 1)
            correct_snn += (snn_predict == target).sum().item()
            num_samples += batchsize
            pbb.set_description(
                f"epoch:{epoch}, snntest: {round(100 * correct_snn / num_samples, 2)}")
            _detach(snn)

        writer.add_scalar("test_acc_snn", 100 * correct_snn / num_samples)
        if 100 * correct_snn / num_samples > best_acc:
            best_acc = 100 * correct_snn / num_samples
            torch.save(ann.state_dict(), f"{save_file_name}_best.pth")
