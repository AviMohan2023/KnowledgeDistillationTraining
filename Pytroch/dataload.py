
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm
from network_MNIST import *

def dataload(key, bs):
    '''data agumentaiton'''

    if key == 'HIGH10':
        traindir ='../data/train/'
        testdir = '../data/test/'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ]))
    if key == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='../data', train=True, download=True,
                                       transform=transform)
        test_dataset = datasets.MNIST(root='../data', train=False,
                                      transform=transform)

    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset,batch_size=bs, shuffle=True)

    return trainloader, testloader

if __name__ == '__main__':
    trainloader, testloader = dataload('HIGH10',bs=16)
    print('Train batch:', len(trainloader) ,',Test batch:', len(testloader))