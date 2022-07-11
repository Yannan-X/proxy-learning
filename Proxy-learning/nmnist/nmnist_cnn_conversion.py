from aermanager.datasets import FramesDataset,SpikeTrainDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import copy

epochs = 50
batchsize = 500
save_file_name = "ann50"
tensorboard_dir = "./tensorboardRC/"
localtime = time.asctime(time.localtime(time.time()))
writer = SummaryWriter(f"{tensorboard_dir}-setting{save_file_name}-batch{batchsize}-epoch{epochs}-{localtime}")

torch.manual_seed(24)


trainset = FramesDataset(source_folder= "./train_set_time50", target_transform=int)
testset  = FramesDataset(source_folder="./test_set_time50", target_transform=int)
train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=12)
test_loader  = DataLoader(testset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=12)

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
    # nn.ReLU(),

)
device = torch.device("cuda:1")
optimizer = torch.optim.Adam(ann.parameters(), lr = 1e-3, weight_decay=1e-6)
C = nn.CrossEntropyLoss()
ann = ann.to(device=device)
best_acc=0


for epoch in range(epochs):
    num_samples = 0
    correct_samples = 0
    if epoch >= 5:
        ann = nn.Sequential(*ann, nn.ReLU())
        ann.to(device)
    ann.train()
    pbr = tqdm(train_loader)
    for sample, target in pbr:
        sample = sample.sum(1).unsqueeze(1)
        sample = sample.float()

        target = target.long().to(device)
        sample = sample.to(device)
        out = ann(sample)
        _, predict = torch.max(out, 1)
        loss = C(out, target)
        correct_samples += (predict == target).sum().item()
        num_samples += batchsize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbr.set_description(f"epoch:{epoch}, loss: {round(loss.item(), 3)}, train_acc: {round(100 * correct_samples / num_samples, 2)}")
    torch.save(ann.state_dict(), f"{save_file_name}.pth")
    writer.add_scalar("train_acc_ann_only", 100 * correct_samples / num_samples)

    # testing
    num_samples = 0
    correct_samples = 0
    ann.eval()
    with torch.no_grad():
        pbb = tqdm(test_loader)
        for sample, target in pbb:
            sample = sample.sum(1).unsqueeze(1)
            sample = sample.float()
            sample = sample.to(device)
            target = target.long().to(device)
            out = ann.forward(sample)
            _, predict = torch.max(out, 1)
            correct_samples += (predict == target).sum().item()
            num_samples += batchsize
            pbb.set_description(f"epoch:{epoch}, test_acc: {round(100 * correct_samples / num_samples, 2)}")
        writer.add_scalar("test_acc_ann_only", 100 * correct_samples / num_samples)
        if 100 * correct_samples / num_samples > best_acc:
            best_acc = 100 * correct_samples / num_samples
            torch.save(ann.state_dict(), f"{save_file_name}_best.pth")





