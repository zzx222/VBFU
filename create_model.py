#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class LogisticRegression(nn.Module):
    def __init__(self, layer_1, layer_2):
        super(LogisticRegression, self).__init__()
        self.classifier = nn.Linear(layer_1, layer_2)
        self.layer_1 = layer_1

    def forward(self, x):
        x = self.classifier(x.view(-1, self.layer_1))
        return x

class LogisticRegression_UL(nn.Module):
    def __init__(self, layer_1, layer_2):
        super(LogisticRegression_UL, self).__init__()
        self.classifier = nn.Linear(layer_1, layer_2)
        self.classifier_ul = nn.Linear(layer_1, layer_2)
        self.layer_1 = layer_1

    def forward(self, x, unlearn=False):
        x = x.view(-1, self.layer_1)
        if unlearn:
            x = self.classifier_ul(x)
        else:
            x = self.classifier(x)
        return x


class CNN_CIFAR(nn.Module):

    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.classifier = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.reshape(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))
        x = self.classifier(x)
        return x



class CNN_CIFAR_UL(nn.Module):

    def __init__(self):
        super(CNN_CIFAR_UL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.classifier = nn.Linear(64, 10)
        self.classifier_ul = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.)

    def forward(self, x, unlearn=True):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.reshape(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        if unlearn:
            x = self.classifier_ul(x)
        else:
            x = self.classifier(x)
        return x


class CNN_CIFAR100(nn.Module):

    def __init__(self):
        super(CNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.classifier = nn.Linear(256, 100)

        self.dropout = nn.Dropout(p=0.)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.reshape(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))
        x = self.classifier(x)
        return x


class CNN_CIFAR100_UL(nn.Module):

    def __init__(self):
        super(CNN_CIFAR100_UL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.classifier = nn.Linear(256, 100)
        self.classifier_ul = nn.Linear(256, 100)

        self.dropout = nn.Dropout(p=0.)

    def forward(self, x, unlearn=True):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.reshape(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        if unlearn:
            x = self.classifier_ul(x)
        else:
            x = self.classifier(x)
        return x



FLATTEN_SIZE = 64*7*7

class CNN_FashionMNIST(nn.Module):
    def __init__(self):
        super(CNN_FashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(FLATTEN_SIZE, 256)
        self.classifier = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, FLATTEN_SIZE)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.classifier(x)
        pred = F.log_softmax(x, dim=1)
        return pred

class CNN_FashionMNIST_UL(nn.Module):
    def __init__(self):

        super(CNN_FashionMNIST_UL, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(FLATTEN_SIZE, 256)
        self.classifier = nn.Linear(256, 10)
        self.classifier_ul = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, unlearn=True):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, FLATTEN_SIZE)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        if unlearn:
            x = self.classifier_ul(x)
        else:
            x = self.classifier(x)

        pred = F.log_softmax(x, dim=1)
        return pred


def load_model(dataset_name: str, model_type: str = "default", seed: int = 42):

    torch.manual_seed(seed)

    dataset = dataset_name.split("_")[0]

    if dataset in ["MNIST", "MNIST-shard"]:
        model = LogisticRegression(784, 10)
        loss_f = torch.nn.CrossEntropyLoss()
        model_ul = LogisticRegression_UL(784, 10)

    elif dataset == "FashionMNIST":
        model = CNN_FashionMNIST()
        loss_f = torch.nn.CrossEntropyLoss()
        model_ul = CNN_FashionMNIST_UL()
    elif dataset == "CIFAR10":
        if model_type == "default":
            assert False, "only the CNN is programmed with CIFAR10"
        elif model_type == "CNN":
            model = CNN_CIFAR()
            model_ul = CNN_CIFAR_UL()
        loss_f = torch.nn.CrossEntropyLoss()

    elif dataset == "CIFAR100":
        if model_type == "default":
            assert False, "only the CNN is programmed with CIFAR100"
        elif model_type == "CNN":
            model = CNN_CIFAR100()
            model_ul = CNN_CIFAR100_UL()
            loss_f = torch.nn.CrossEntropyLoss()


    print(model)

    return model, loss_f, model_ul
