
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.layer1 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(32),  # Batch normalization for faster convergence
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.layer2 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(64),  # Batch normalization
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Fully connected layer: input size 7*7*64, output size 1000
        self.fc1 = nn.Linear(7*7*64, 1000)
        # Fully connected layer: input size 1000, output size 10 (number of classes)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Forward pass through the first convolutional layer
        out = self.layer1(x)
        # Forward pass through the second convolutional layer
        out = self.layer2(out)
        # Flatten the output for the fully connected layer
        out = out.reshape(out.size(0), -1) # also look at https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
        # Forward pass through the first fully connected layer
        out = self.fc1(out)
        # Forward pass through the second fully connected layer
        out = self.fc2(out)
        return out

def track_accuracy(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)  # Move images to the configured device
            labels = labels.to(device)  # Move labels to the configured device
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Increment total by the number of labels
            correct += (predicted == labels).sum().item()  # Increment correct by the number of correct predictions
        accuracy = 100 * correct / total  # Calculate accuracy
    model.train()  # Set the model back to training mode
    return accuracy

def training(model, num_epochs, loader, criterion, optimizer, device, test_loader):
    # Train the model
    total_step = len(loader)  # Total number of batches
    loss_list = []  # List to store loss values
    acc_train = []  # List to store training accuracy values
    acc_test = []  # List to store testing accuracy values

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        # Loop over each batch in the training DataLoader

        pbar = tqdm(loader,
                    desc = f"Training: {epoch + 1:03d}/{num_epochs}",
                    ncols = 125,
                    leave = True)


        # Creating running loss & accuracy empty lists
        running_loss = []
        running_accuracy = []

        for i, (images, labels) in enumerate(pbar, 1):
            # Important note: In PyTorch, the images in a batch is typically represented as (batch_size, channels, height, width)
            # For example, (100, 1, 28, 28) for the MNIST data
            images = images.to(device)  # Move images to the configured device
            labels = labels.to(device)  # Move labels to the configured device

            # Forward pass: compute predicted y by passing x to the model
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())

            # Backward pass & optimize
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Add the runnign loss
            pbar.set_postfix(loss = f"{sum(running_loss)/i : 10.6f}")

        # Get the average of all losses in the running_loss as the loss of the current epoch
        loss_list.append(sum(running_loss)/i)

        # Track accuracy on the train set
        acc_train.append(track_accuracy(model, loader, device))

        # Track accuracy on the test set
        acc_test.append(track_accuracy(model, test_loader, device))

    return model, loss_list, acc_train, acc_test

def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    # Plot the loss & accuracy curves
    plt.figure(figsize=(10, 4))

    # Plot the training loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.savefig(save_path)

if __name__ == "__main__":
    pass








