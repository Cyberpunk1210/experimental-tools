import os

import torch
import torch.distributed as dist
import torch.utils
from torch.utils.data import DataLoader, TensorDataset, distributed
from torch.multiprocessing import Process
import torchvision
from torchvision import transforms

# single machine single device
def train():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST("./", train=True, transform=trans, target_transform=None, download=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set, batch_size=256)
    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    net = net.cuda()
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(10):
        for i, data in enumerate(data_loader_train):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print("loss: {}".format(loss.item()))
    torch.save(net, "smsd_net.path")


# single machine multi device
def train(rank):
    dist.init_process_group("gloo", rank=rank, world_size=3)
    torch.cuda.set_device(rank)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (1.0,))])
    data_set = torchvision.datasets.MNIST("./", train=True, transform=trans, target_transform=None, download=True)
    train_sampler = distributed.DistributedSampler(data_set)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set, batch_size=256,)

    net = torchvision.models.resnet101(num_classes=10)
    net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    net = net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(10):
        for batch, data in enumerate(data_loader_train):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            if batch % 10 == 0:
                print("loss: {}".format(loss.item()))
    if rank == 0:
        torch.save(net, "smmd_net.pth")


# multi machine single device


if __name__ == "__main__":
    size = 3
    processes = []
    for rank in range(size):
        p = Process(target=train, args=(rank, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    '''
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    '''
