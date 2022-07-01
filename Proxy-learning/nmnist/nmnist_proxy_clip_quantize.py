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
save_file_name = "proxy50"
tensorboard_dir = "./tensorboardRC/"
localtime = time.asctime(time.localtime(time.time()))
writer = SummaryWriter(f"{tensorboard_dir}{save_file_name}-batch{batchsize}-epoch{epochs}-{localtime}")

torch.manual_seed(24)


def _determine_discretization_scale(obj: torch.Tensor, bit_precision: int) -> float:
    """Determine a scale for discretization

    Determine how much the values of a torch tensor can be scaled in order to fit
    the given precision

    Parameters
    ----------
        obj: torch.Tensor
            Tensor that is to be scaled
        bit_precision: int
            The precision in bits

    Returns
    -------
        float
            The scaling factor
    """

    # Discrete range
    min_val_disc = -(2 ** (bit_precision - 1))
    max_val_disc = 2 ** (bit_precision - 1) - 1

    # Range in which values lie
    min_val_obj = torch.min(obj)
    max_val_obj = torch.max(obj)

    # Determine if negative or positive values are to be considered for scaling
    # Take into account that range for diescrete negative values is slightly larger than for positive
    min_max_ratio_disc = abs(min_val_disc / max_val_disc)
    if abs(min_val_obj) <= abs(max_val_obj) * min_max_ratio_disc:
        scaling = abs(max_val_disc / max_val_obj)
    else:
        scaling = abs(min_val_disc / min_val_obj)

    return scaling.to(device)

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

device = torch.device("cuda")

def _quantization(model: nn.Sequential, w_precision=8, state_precision=16):
    """
    Does the identical quantization method with DYNAPCNN-COMPATIBLE network discrete method
    :param model: sequential snn model, suppose to have IAFsqueeze layers followed by parameter layers
    :param w_precision: predefined by chip
    :param state_precision:  predefined by chip
    :return:
    """

    depth = len(model)

    for i, lyrs in enumerate(model):
        if i <= depth-2:
            if isinstance(lyrs, nn.Conv2d) or isinstance(lyrs, nn.Linear):
                # print(f"current_layer numner: {i}")
                w_factor = _determine_discretization_scale(lyrs.weight.data, bit_precision=w_precision)
                thresholds = torch.tensor((model[i + 1].min_v_mem, model[i + 1].spike_threshold)).to(device)
                t_factor = _determine_discretization_scale(thresholds, bit_precision=state_precision)
                scale = min(w_factor, t_factor)

                lyrs.weight.data = torch.round(lyrs.weight.data.clone() * scale).to(device)
                (model[i + 1].min_v_mem, model[i + 1].spike_threshold) = torch.round(thresholds.clone() * scale)
                model[i+1].v_mem.data = torch.round(model[i+1].v_mem.data.clone() *scale)
            else:
                if isinstance(lyrs, nn.Conv2d) or isinstance(lyrs, nn.Linear):
                    # print(f"current_layer numner: {i}")
                    w_factor = _determine_discretization_scale(lyrs.weight.data, bit_precision=w_precision)
                    lyrs.weight.data = torch.round(lyrs.weight.data.clone() * w_factor).to(device)




snn = from_model(ann, batch_size=batchsize, surrogate_grad_fn=PeriodicExponential(),
                 reset_fn=MembraneSubtract(), spike_threshold=1).spiking_model


def _detach(model):
    for lyrs in model:
        if isinstance(lyrs, IAFSqueeze):
            lyrs.v_mem.detach_()

# quantize = DynapcnnCompatibleNetwork(snn, discretize=True, input_shape=(1,34,34))
# quant_self = _quantization(snn)
# print(snn)

def _reset_states(model):
    for lyrs in model:
        if isinstance(lyrs, IAFSqueeze):
            lyrs.reset_states(randomize=True)


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








device = torch.device("cuda:0")
# optimizer = torch.optim.Adam(ann.parameters(), lr=1e-3, weight_decay=1e-6)

weight_initializer(ann)
weight_initializer(snn)

ann[0].weight.data *=5
snn[0].weight.data *=5

C = nn.CrossEntropyLoss()
best_acc = 0


# params_ann = ann.named_parameters()
# params_snn = snn.named_parameters()
# dict_params_snn = dict(params_snn)
# for name, param in params_ann:
#     if name in dict_params_snn:
#         dict_params_snn[name].data = param.data

def _weight_sharing_(source:nn.Sequential, target):
    """
    source and target should have identical structure and only have conv and linear layers
    :param source: source model that parameter copy from
    :param target: target model that parameter
    :return:
    """
    for lyrs in source.state_dict():
        if not "v_mem" in lyrs:
            layer_number = int(lyrs.split(".")[0])
            target[layer_number].weight.data.copy_(source[layer_number].weight.data)
        # target[layer_number].weight.data=source[layer_number].weight.data
    for i, lyrs in enumerate(source):
        if isinstance(lyrs, nn.Dropout2d):
            target[i] = copy.deepcopy(source[i])




# ann[0].weight.data *= 5

#
# ann = nn.Sequential(*ann, nn.LeakyReLU())
# snn = nn.Sequential(*snn, IAFSqueeze(batch_size=batchsize, surrogate_grad_fn=PeriodicExponential(),
#                  reset_fn=MembraneSubtract(), spike_threshold=1, min_v_mem=-1))


ann = ann.to(device)
snn = snn.to(device)
print(ann)
_reset_states(snn)
_quantization(snn)
# _weight_sharing_(snn, ann)

# def forward_hook(module, input, output):
#     print(module, output.shape, output.sum().sum().sum().sum())
# for lyrss in snn:
#     if isinstance(lyrss, IAFSqueeze):
#         lyrss.register_forward_hook(forward_hook)

# def backward_hook(module, grad_input, grad_output):
#     print(module)
#     print(grad_input)
#     print(grad_output)
# ann[0].register_backward_hook(backward_hook)


# optimizer = torch.optim.Adam(ann.parameters(), lr=1)
optimizer = torch.optim.Adam(ann.parameters(), lr=1e-4)
for epoch in range(epochs):

    num_samples = 0
    correct_ann = 0
    correct_snn = 0


    # if epoch > 5:
    #     ann = nn.Sequential(*ann, nn.ReLU())
    #     snn = nn.Sequential(*snn, IAFSqueeze(batch_size=batchsize))
    #     snn = snn.to(device)
    #     ann = ann.to(device)

    ann.train()
    snn.train()
    pbr = tqdm(train_loader)

    for sample, target in pbr:
        sample = sample.to(device)
        snn_sample = sample.clone().sum(2).unsqueeze(2)
        ann_sample = sample.clone().sum(1).sum(1).unsqueeze(1)
        # sample = sample.float()

        # label_one_hot = F.one_hot(target, 10).float().to(device)
        target = target.long().to(device)

        out_ann = ann(ann_sample)
        with torch.no_grad():
            out_snn = raster_forward(snn, snn_sample)

        # print("ann  ",out_ann[0])
        # print("snn  ",out_snn[0])

        #
        # print(out_ann[0])
        # print(out_snn[0])
        _, ann_predict = torch.max(out_ann, 1)
        _, snn_predict = torch.max(out_snn, 1)

        correct_ann += (ann_predict == target).sum().item()
        correct_snn += (snn_predict == target).sum().item()
        num_samples += batchsize

        # if epoch > 5:
        out_ann.data.copy_(out_snn)

        loss = C(out_ann,target)
        # loss = 0.5*(C(out_ann, target) + C(out_snn, target))
        # loss = F.mse_loss(out_ann, label_one_hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _weight_sharing_(ann, snn)
        _quantization(snn)
        # _weight_sharing_(snn, ann)

        # print(ann[-2].weight.data[0][0])
        # print(snn[-2].weight.data[0][0])
        _detach(snn)
        pbr.set_description(f"epoch:{epoch},loss:{loss.detach().cpu().numpy()} train_annacc: {round(100 * correct_ann / num_samples, 2)}, train_snnacc: {round(100 * correct_snn / num_samples, 2)}")


    writer.add_scalar("train/acc_ann", 100 * correct_ann / num_samples)
    writer.add_scalar("train/acc_snn", 100 * correct_snn / num_samples)

    torch.save(ann.state_dict(), f"{save_file_name}.pth")

    # testing
    # num_samples = 0
    # correct_ann = 0
    # correct_snn = 0
    # ann.eval()
    # snn.eval()
    #
    # with torch.no_grad():
    #     pbb = tqdm(test_loader)
    #     for sample, target in pbb:
    #
    #         sample = sample.to(device)
    #         snn_sample = sample.clone().sum(2).unsqueeze(2)
    #         ann_sample = sample.clone().sum(1).sum(1).unsqueeze(1)
    #         target = target.long().to(device)
    #
    #         out_ann = ann(ann_sample)
    #         with torch.no_grad():
    #             out_snn = raster_forward(snn, snn_sample)
    #
    #         _, ann_predict = torch.max(out_ann, 1)
    #         _, snn_predict = torch.max(out_snn, 1)
    #
    #         correct_ann += (ann_predict == target).sum().item()
    #         correct_snn += (snn_predict == target).sum().item()
    #         num_samples += batchsize
    #         pbb.set_description(
    #             f"epoch:{epoch}, anntest: {round(100 * correct_ann / num_samples, 2)}, snntest: {round(100 * correct_snn / num_samples, 2)}")
    #
    #         _detach(snn)
    #     writer.add_scalar("test_acc_ann", 100 * correct_ann / num_samples)
    #     writer.add_scalar("test_acc_snn", 100 * correct_snn / num_samples)
    #     if 100 * correct_snn / num_samples > best_acc:
    #         best_acc = 100 * correct_snn / num_samples
    #         torch.save(ann.state_dict(), f"{save_file_name}_best.pth")
