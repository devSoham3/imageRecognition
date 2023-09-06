import math
import time
import sys
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f} %".format(epoch + 1, result['val_loss'], (result['val_acc'] * 100)))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


class CIFAR10Model(ImageClassificationBase):
    def __init__(self, use_batch_norm):
        super().__init__()
        # part 1
        self.use_batch_norm = use_batch_norm
        print("Use batch normalization: {}".format(self.use_batch_norm))
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(9)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm2d(9)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # part 2
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn3 = nn.BatchNorm2d(18)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn4 = nn.BatchNorm2d(18)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # part 2
        self.conv5 = nn.Conv2d(in_channels=18, out_channels=36, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn5 = nn.BatchNorm2d(36)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=3, padding=1)
        if self.use_batch_norm:
            self.bn6 = nn.BatchNorm2d(36)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # part 3
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(576, 100)
        if self.use_batch_norm:
            self.bn7 = nn.BatchNorm1d(100)
        self.relu7 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, out):
        out = self.conv1(out)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        if self.use_batch_norm:
            out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        if self.use_batch_norm:
            out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool2(out)
        out = self.conv5(out)
        if self.use_batch_norm:
            out = self.bn5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        if self.use_batch_norm:
            out = self.bn6(out)
        out = self.relu6(out)
        out = self.pool3(out)
        out = self.flatten(out)
        out = self.fc1(out)
        if self.use_batch_norm:
            out = self.bn7(out)
        out = self.relu7(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CIFAR10_CNN_pa4', description='Use arguments: <batch_size> <use_batch_norm> <epochs_per_run> <learning_rate_start> <no_of_runs>')
    parser.add_argument('batch_size', type=int, help="(int) batch size to be used")
    parser.add_argument('use_batch_norm', type=int, help="(int -> 0 or 1) whether to use batch normalization or not")
    parser.add_argument('epochs_per_run', type=int, help="(int) number of epochs for each fitting run")
    parser.add_argument('learning_rate_start', type=float, help="(float) initial learning rate")
    parser.add_argument('no_of_runs', type=int, help="(int) number of fitting runs")
    args = parser.parse_args()

    use_batch_norm = (args.use_batch_norm == 1)

    # defining the training and testing datasets
    dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())
    dataset_size = len(dataset)
    test_dataset_size = len(test_dataset)

    # used to check classes
    classes = dataset.classes  # unused
    num_classes = len(dataset.classes)  # unused

    torch.manual_seed(75)  # to reproduce results

    # training-validation split
    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print("Training-validation split complete")

    batch_size = args.batch_size  # batch size declaration
    print("Batch size set to {}".format(batch_size))

    # data loaders for training, validation and testing data
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size * 2, num_workers=4, pin_memory=True)

    # device selection - use cuda (GPU) if possible
    device = get_default_device()
    print("Using device: {}".format(device))

    # wrapping data loaders for devices
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    # input-output size specification
    input_size = 3 * 32 * 32  # unused
    output_size = 10  # unused

    # moving model to device
    model = to_device(CIFAR10Model(use_batch_norm=use_batch_norm), device)
    parameter_count = count_parameters(model)
    print("Count of parameters = {}".format(parameter_count))

    # initial metrics
    history = [evaluate(model, val_loader)]
    print("Initial metrics before training: loss = {:.4f}, accuracy = {:.4f} %".format(history[0]['val_loss'], (history[0]['val_acc'] * 100)))

    # run parameters
    e = args.epochs_per_run
    lr = args.learning_rate_start
    runs = args.no_of_runs

    # timer
    start = time.time()
    print("Timer started")

    # fitting the model over multiple runs
    for run in range(runs):
        print("Starting run {} of {}".format(run + 1, runs))
        print("Fitting with parameters: epochs[{}], learning_rate[{}]".format(e, lr))
        history += fit(e, lr, model, train_loader, val_loader)
        lr *= 0.5
        interval = time.time() - start
        print("timer: {} s".format(math.ceil(interval)))
        print("-" * 100)

    # plot loss and accuracy progression
    plot_losses(history)
    plot_accuracies(history)
    print("-" * 100)

    # training results
    final_train_result = evaluate(model, val_loader)
    val_acc = final_train_result['val_acc']
    val_loss = final_train_result['val_loss']
    print("TRAINING RESULTS: loss = {:.4f}, accuracy = {:.4f} %".format(final_train_result['val_loss'], (final_train_result['val_acc'] * 100)))
    print("-" * 100)

    # testing results
    final_test_result = evaluate(model, test_loader)
    test_acc = final_train_result['val_acc']
    test_loss = final_train_result['val_loss']
    print("TESTING RESULTS: loss = {:.4f}, accuracy = {:.4f} %".format(final_test_result['val_loss'], (final_test_result['val_acc'] * 100)))
    print("-" * 100)

    # arch = '3 layers (256,128,10)'
    # lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    # epochs = [10, 10, 10, 10]
    # test_acc = final_result['val_acc']
    # test_loss = final_result['val_loss']
    # final_result
    # torch.save(model.state_dict(), 'cifar10-feedforward.pth')
