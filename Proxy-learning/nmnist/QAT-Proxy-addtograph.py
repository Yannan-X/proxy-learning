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
from sinabs.layers import IAFSqueeze

epochs = 100
batchsize = 500
save_file_name = "proxy50"
tensorboard_dir = "./tensorboardRC/"
localtime = time.asctime(time.localtime(time.time()))
writer = SummaryWriter(f"{tensorboard_dir}{save_file_name}-batch{batchsize}-epoch{epochs}-{localtime}")

torch.manual_seed(24)


class StochasticRounding(torch.autograd.Function):
    """PyTorch-compatible function that applies stochastic rounding. The input x
    is quantized to ceil(x) with probability (x - floor(x)), and to floor(x)
    otherwise. The backward pass is provided as a surrogate gradient
    (equivalent to that of a linear function)."""

    @staticmethod
    def forward(ctx, inp):
        """"""
        int_val = inp.floor()
        frac = inp - int_val
        output = int_val + (torch.rand_like(inp) < frac).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        grad_input = grad_output.clone()
        return grad_input


def _quantization_scale(input_tensor, bit_precision=8):
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
        self.round = StochasticRounding

    def forward(self, sample):
        q1 = _quantization_scale(self.conv1.weight.data)
        propgate = self.relu(self.round.apply(self.conv1(sample) * q1) / q1)
        q2 = _quantization_scale(self.conv2.weight.data)
        propgate = self.pool2(self.relu(self.round.apply(self.conv2(propgate) * q2) / q2))
        q3 = _quantization_scale(self.conv3.weight.data)
        propgate = self.pool2(self.relu(self.round.apply(self.conv3(propgate) * q3) / q3))

        propgate = propgate.view(-1, 4 * 4 * 8)
        q4 = _quantization_scale(self.fc1.weight.data)
        propgate = self.relu(self.round.apply(self.fc1(propgate) * q4) / q4)
        q5 = _quantization_scale(self.fc2.weight.data)
        propgate = self.round.apply(self.fc2(propgate) * q5) / q5

        return propgate


class SNN(torch.nn.Module):

    def __init__(self, batchsize, device):
        super(SNN, self).__init__()
        self.device = device
        self.snn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False),  # 16, 18, 18
            IAFSqueeze(batch_size=batchsize, surrogate_grad_fn=PeriodicExponential(),
                       reset_fn=MembraneSubtract(), spike_threshold=1, min_v_mem=-1),

            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 18,18
            IAFSqueeze(batch_size=batchsize, surrogate_grad_fn=PeriodicExponential(),
                       reset_fn=MembraneSubtract(), spike_threshold=1, min_v_mem=-1),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 8, 17,17

            nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 9, 9
            IAFSqueeze(batch_size=batchsize, surrogate_grad_fn=PeriodicExponential(),
                       reset_fn=MembraneSubtract(), spike_threshold=1, min_v_mem=-1),
            nn.AvgPool2d(kernel_size=(2, 2)),

            nn.Flatten(),
            nn.Linear(4 * 4 * 8, 256, bias=False),
            IAFSqueeze(batch_size=batchsize, surrogate_grad_fn=PeriodicExponential(),
                       reset_fn=MembraneSubtract(), spike_threshold=1, min_v_mem=-1),

            nn.Linear(256, 10, bias=False),
        )

    def forward(self, sample):
        sample = sample.clone().sum(2).unsqueeze(2)
        sample = sample.to(self.device)
        (batch_size, t_len, channel, height, width) = sample.shape
        sample = sample.reshape((batch_size * t_len, channel, height, width))
        out = self.snn(sample)
        out = out.reshape(batch_size, t_len, 10)
        out = out.sum(1)
        return out

    def detach_state(self):
        for lyrs in self.snn:
            if isinstance(lyrs, IAFSqueeze):
                lyrs.v_mem.detach_()


ann_seq = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False),  # 16, 18, 18
    nn.ReLU(),

    nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 18,18
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),  # 8, 17,17

    nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 8, 9, 9
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2)),

    nn.Flatten(),
    nn.Linear(4 * 4 * 8, 256, bias=False),
    nn.ReLU(),

    nn.Linear(256, 10, bias=False),
)


def weight_transfer(module: nn.Module, seq: nn.Module):
    # seq[0].weight.data = torch.round(module.conv1.weight.data.clone() * _quantization_scale(module.conv1.weight.data))
    # seq[2].weight.data = torch.round(module.conv2.weight.data.clone() * _quantization_scale(module.conv2.weight.data))
    # seq[5].weight.data = torch.round(module.conv3.weight.data.clone() * _quantization_scale(module.conv3.weight.data))
    # seq[9].weight.data = torch.round(module.fc1.weight.data.clone() * _quantization_scale(module.fc1.weight.data))
    # seq[11].weight.data = torch.round(module.fc2.weight.data.clone() * _quantization_scale(module.fc2.weight.data))

    seq[0].weight.data = module.conv1.weight.data.clone()
    seq[2].weight.data = module.conv2.weight.data.clone()
    seq[5].weight.data = module.conv3.weight.data.clone()
    seq[9].weight.data = module.fc1.weight.data.clone()
    seq[11].weight.data = module.fc2.weight.data.clone()


def discrete_snn(model, w_precision=8, state_precision=16, low_thresh=-1, spike_thresh=1):
    """
    Does the identical quantization method with DYNAPCNN-COMPATIBLE network discrete method
    :param spike_thresh: spiking threshold
    :param low_thresh: low v_mem boundard of the layer
    :param model: sequential snn model, suppose to have IAFsqueeze layers followed by parameter layers
    :param w_precision: predefined by chip
    :param state_precision:  predefined by chip
    :return:
    """

    depth = len(model)

    for i, lyrs in enumerate(model):
        if i <= depth - 2:
            if isinstance(lyrs, nn.Conv2d) or isinstance(lyrs, nn.Linear):
                # print(f"current_layer numner: {i}")
                w_factor = _quantization_scale(lyrs.weight.data, bit_precision=w_precision)
                thresholds = torch.tensor((low_thresh, spike_thresh)).to(device)
                t_factor = _quantization_scale(thresholds, bit_precision=state_precision)
                scale = min(w_factor, t_factor)

                lyrs.weight.data = torch.round(lyrs.weight.data.clone() * scale).to(device)
                (model[i + 1].min_v_mem, model[i + 1].spike_threshold) = torch.round(thresholds.clone() * scale)
                model[i + 1].v_mem.data = torch.round(model[i + 1].v_mem.data.clone() * scale)

            else:
                if isinstance(lyrs, nn.Conv2d) or isinstance(lyrs, nn.Linear):
                    # print(f"current_layer numner: {i}")
                    w_factor = _quantization_scale(lyrs.weight.data, bit_precision=w_precision)
                    lyrs.weight.data = torch.round(lyrs.weight.data.clone() * w_factor).to(device)


train_raster = SpikeTrainDataset(dt=1000, source_folder="./train_set_time50", target_transform=int,
                                 force_n_time_bins=50)
test_raster = SpikeTrainDataset(dt=1000, source_folder="./test_set_time50", target_transform=int, force_n_time_bins=50)

train_loader = DataLoader(train_raster, batch_size=batchsize, shuffle=True, drop_last=True,
                          num_workers=24)
test_loader = DataLoader(test_raster, batch_size=batchsize, shuffle=True, drop_last=True,
                         num_workers=24)

# model = ANN_SNN_PROXY_MODULE(batch_size=batchsize, device=torch.device("cuda"))
device = torch.device("cuda")
ann = ANN()

ann_seq = ann_seq.to(device)
# for _ in range(1000):
#     out = model.ann_forward(torch.ones((500, 1, 1, 34, 34)))
#     print(out.sum().sum())


for m in ann.children():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)

snn = SNN(batchsize=batchsize, device=device)
snn_q = SNN(batchsize=batchsize, device=device)

ann.to(device)
snn.to(device)
snn_q.to(device)

weight_transfer(ann, snn.snn)
weight_transfer(ann, snn_q.snn)
discrete_snn(snn_q.snn)

optimizer = torch.optim.Adam(ann.parameters(), lr=1e-3, weight_decay=1e-8)

criterion = nn.CrossEntropyLoss()

for _ in range(100):
    ann.train()
    snn.train()

    num_samples = 0
    correct_ann = 0
    correct_snn = 0
    pbr = tqdm(train_loader)

    for sample, target in pbr:
        # prepare samples

        sample = sample.to(device)
        sample_ann = sample.clone().sum(1).sum(1).unsqueeze(1)
        sample_snn = sample.clone().sum(2).unsqueeze(2)
        target = target.to(device)

        out_ann = ann(sample_ann)
        with torch.no_grad():
            out_snn = snn(sample_snn)

        _, ann_predict = torch.max(out_ann, 1)
        _, snn_predict = torch.max(out_snn, 1)

        correct_ann += (ann_predict == target).sum().item()
        correct_snn += (snn_predict == target).sum().item()
        num_samples += batchsize

        out_ann.data.copy_(out_snn)

        loss = criterion(out_ann, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        weight_transfer(ann, snn.snn)

        pbr.set_description(
            f"train_ann:{round(correct_ann / num_samples, 4)}, train_snn:{round(correct_snn / num_samples, 4)}")


    ann.eval()
    snn.eval()
    snn_q.eval()

    weight_transfer(ann, snn_q.snn)
    discrete_snn(snn_q.snn)

    num_samples = 0
    correct_ann = 0
    correct_snn = 0
    correct_snn_q = 0
    pbr = tqdm(test_loader)

    with torch.no_grad():
        for sample, target in pbr:
            sample = sample.to(device)
            sample_snn = sample.clone().sum(2).unsqueeze(2)
            sample_ann = sample.clone().sum(1).sum(1).unsqueeze(1)
            target = target.to(device)

            out_ann = ann(sample_ann)
            out_snn = snn(sample_snn)
            out_snnq = snn_q(sample_snn)

            _, ann_predict = torch.max(out_ann, 1)
            _, snn_predict = torch.max(out_snn, 1)
            _, snnq_predict = torch.max(out_snnq, 1)

            correct_ann += (ann_predict == target).sum().item()
            correct_snn += (snn_predict == target).sum().item()
            correct_snn_q += (snnq_predict == target).sum().item()

            num_samples += batchsize

            pbr.set_description(
                f"T_ann:{round(correct_ann / num_samples, 4)}||| T_snn:{round(correct_snn / num_samples, 4)}, T_q:{round(correct_snn_q / num_samples, 4)}")
