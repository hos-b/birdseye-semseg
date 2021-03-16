import os
import torch
import datetime
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from data.config import SemanticCloudConfig, TrainingConfig

class ToyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ToyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def say_hi(self, x):
        print (f'hi {x}')


def train(gpu, args):
    rank = gpu


    torch.manual_seed(0)
    model = ToyModel()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./datasets',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))



if __name__ == "__main__":
    model = ToyModel().cuda()
    replicas = nn.parallel.replicate(model, [0, 1])
    import pdb; pdb.set_trace()

    world_size = 2
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    train_cfg = TrainingConfig('config/training.yml')
    # create the processes on the current node with
    # given args & a per-node increasing id [0, nprocs]
    mp.spawn(train, nprocs=2, args=(geom_cfg, train_cfg) )