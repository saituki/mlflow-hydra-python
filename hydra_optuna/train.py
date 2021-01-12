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

from experiment_recorder import *
from model import SAMPLE_DNN


   
@hydra.main(config_path='config.yaml')
def main(cfg):

    # make_dataset
    dataset = MNIST("/Workdir/data", download=False, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])
    trainloader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True)
    testloader = DataLoader(val, batch_size=cfg.test.batch_size, shuffle=False)

    # model,criterion,optim
    model = SAMPLE_DNN(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr,
                          momentum=cfg.optimizer.momentum)

    # tracking
    recorder = ExperimentRecorder('test3',run_name=f'optimizer.lr={cfg.optimizer.lr},model.node={cfg.model.node1}')
    recorder.log_all_params(cfg) 

    # learn
    for epoch in range(cfg.train.epoch):
        running_loss = 0.0
        for i, (x, y) in enumerate(trainloader):
            steps = epoch * len(trainloader) + i
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            mlflow.log_metric("loss", loss.item(), step=steps)

        correct = 0
        total = 0
        with torch.no_grad():
            for (x, y) in testloader:
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = float(correct / total)
        mlflow.log_metric("acc", accuracy, step=epoch)
    
    # end
    recorder.end_run()
    
    # 5. remove outputs
    os.chdir("/Workdir")
    try:
        shutil.rmtree("multiruns")
    except:
        pass
    
    try:
        shutil.rmtree("outputs")
    except:
        pass

    return accuracy

if __name__ == '__main__':
    main()