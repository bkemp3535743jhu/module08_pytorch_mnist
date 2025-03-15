import argparse
from utilities import ConvNet, track_accuracy, training, plot

# TODO: Try to move the code that requires these to utilities.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def main():
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = myargs()
    # Hyperparameters
    num_epochs = args.num_epochs  # Number of times the entire dataset is passed through the model
    batch_size = args.batch_size  # Number of samples per batch to be passed through the model
    learning_rate = args.learning_rate  # Step size for parameter updates

    # MNIST dataset
    # Download & load the training dataset, applying a transformation to convert images to PyTorch tensors
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

    # Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor())

    # Data loader
    # DataLoader provides an iterable over the dataset with support for batching, shuffling, & parallel data loading
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    # DataLoader for the test dataset, used for evaluating the model
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    # Create an instance of the model & move it to the configured device (GPU/CPU)
    model = ConvNet().to(device)

    # Print the model for observation
    print(model)

    # Loss & optimizer
    # CrossEntropyLoss combines nn.LogSoftmax & nn.NLLLoss in one single class
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    model, loss_list, acc_train, acc_test = training(model, num_epochs,
                                                    train_loader, criterion, optimizer, device, test_loader)
    # Make some plots!
    plot(loss_list, acc_train, acc_test)


def myargs():
    parser = argparse.ArgumentParser(description="Train a classification model")
    # Add the arguments
    parser.add_argument("-e", "--num_epochs", type=int, help="Number of Epochs for training model", default = 2)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size used for training model", default = 1000)
    parser.add_argument("-l", "--learning_rate", type=int, help="Learning rate used for training model", default = 0.001)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()