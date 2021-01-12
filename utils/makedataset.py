import os
import sys
from omegaconf import DictConfig, ListConfig
import logging

import hydra
from hydra import utils
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

# make_dataset
dataset = MNIST("/Workdir/data", download=False, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])
trainloader = DataLoader(train, batch_size=128, shuffle=True)
testloader = DataLoader(val, batch_size=128, shuffle=False)