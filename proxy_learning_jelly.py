import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.clock_driven import neuron, functional, encoding, surrogate, layer
import sys
import time
import numpy as np
from tqdm import tqdm


_seed_ =  2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)



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

class SNN(nn.Module):
    def __init__(self, T, v_threshold=2.0, v_reset=0.0):
        super().__init__()
        self.T = T
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128 * 4 * 5, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(128 * 4 * 5, 128 * 3 * 3, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(128 * 3 * 3, 128 * 2 * 1, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(128 * 2 * 1, 10, bias=False),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        x = self.static_conv(x)
        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            if (t==0):
                out_spikes_counter = self.fc(self.conv(x))
            else:
                out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter


def main():
    # Parameters Setting
    device = "cuda:0"
    dataset_dir = "./"
    batch_size = 100
    learning_rate = 1e-4
    T = 15
    train_epoch = 100
    log_dir = "./"
    model_dir = "path to ANN and SNN saved models on your local machine or on your Google Drive"

    # Data transormations
    test_list_transforms = [
        transforms.ToTensor(),
    ]

    train_list_transforms = [
        transforms.RandomCrop(26),
        transforms.Pad(1),
        transforms.ToTensor(),
    ]

    # Data loaders
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

    # Building or loading models (ANN and SNN)
    print('Load pretrained model? (y/n) ')
    pretrained = input()
    if pretrained == 'y':
        print('Loading... ')
        ann = torch.load(model_dir + '/ANN_Params.pt', map_location=device)
        snn = torch.load(model_dir + '/SNN_Params.pt', map_location=device)
        print('Pretrained model loaded!')
        print('Evaluation on test data:')
        train_epoch = 0

    else:
        ann = ANN().to(device)
        snn = SNN(T=T).to(device)
        print('Model initialized with random weights!')

    # Weight Sharing: set ptr of snn's param to point ann's param
    params_ann = ann.named_parameters()
    params_snn = snn.named_parameters()
    dict_params_snn = dict(params_snn)
    for name, param in params_ann:
        if name in dict_params_snn:
            dict_params_snn[name].data = param.data

    # Optimizer Settings
    optimizer_ann = torch.optim.Adam(ann.parameters(), lr=learning_rate, betas=(0.8, 0.99), eps=1e-08,
                                     weight_decay=1e-06)
    # criterion = nn.CrossEntropyLoss()

    # Learning
    print('Learning started...')
    iterations = 0
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
        t_start = time.perf_counter()
        for img, label in tqdm(train_data_loader, position=0):
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()
            optimizer_ann.zero_grad()
            outputs = ann(img)  # ANN output
            acc_ann = (outputs.argmax(1) == label).float().mean().item()
            with torch.no_grad():
                out_spikes_counter = snn(img)  # SNN spike counts
            print(outputs[0], out_spikes_counter[0])
            # Computing trainig accuracies on each batch
            predict_ann = outputs.argmax(1)
            correct_ann += (predict_ann == label).sum().item()
            sample_num += label.numel()
            correct_snn += (out_spikes_counter.argmax(1) == label).float().sum().item()

            # out_spikes_counter_frequency =  out_spikes_counter
            outputs.data.copy_(out_spikes_counter)  # Replacing SNN output in ANN output layer
            loss = F.mse_loss(outputs, label_one_hot)  # Comuting the loss in ANN (by the SNN output)
            loss.backward()  # computing the gradients in ANN
            optimizer_ann.step()
            # updating the shared weights
            functional.reset_net(snn)  # reseting the snn for next inputs

            iterations += 1

        acc_ann = correct_ann / sample_num
        acc_snn = correct_snn / sample_num
        print(f'epoch={epoch}, train_ann={acc_ann}, train_snn={acc_snn}')  # , t_train={t_train}, t_test={t_test}')
        t_train = time.perf_counter() - t_start

        # Evaluation on test samples
        ann.eval()
        snn.eval()
        t_start = time.perf_counter()
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

                functional.reset_net(snn)

            acc_ann = correct_ann / sample_num
            acc_snn = correct_snn / sample_num
            t_test = time.perf_counter() - t_start

            print(f'epoch={epoch}, acc_ann={acc_ann}, acc_snn={acc_snn}')  # , t_train={t_train}, t_test={t_test}')


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

                # correct_snn += (out_spikes_counter_frequency.argmax(1) == label).sum()
                lab = F.one_hot(label, 10).float()
                lab = (lab.cpu()).numpy()
                lab = lab.astype(bool)
                out_spikes_counter_frequency = ((out_spikes_counter_frequency.cpu()).detach()).numpy()
                lab2 = out_spikes_counter_frequency[lab]
                correct_snn += (out_spikes_counter_frequency.max(1) == lab2).sum()

                functional.reset_net(snn)

            acc_ann = correct_ann / sample_num
            acc_snn = correct_snn / sample_num
            print(f' Final Result: Acc_ANN={acc_ann}, Acc_SNN={acc_snn}')  # , t_train={t_train}, t_test={t_test}')


if __name__ == '__main__':
    main()