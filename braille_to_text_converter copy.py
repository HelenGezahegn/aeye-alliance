import time
import platform
import io
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import csv
import os
import pandas as pd
from urllib import request
from scipy import misc
from io import BytesIO
import urllib
import numpy as np
import tensorflow as tf
import urllib.request
from torchvision import transforms
import string
import cv2
from urllib import request
from scipy import misc
from io import BytesIO
import urllib
import urllib.request
from torchvision import transforms




# Upload and read the csv file from the github repo
df = pd.read_csv("https://raw.githubusercontent.com/HelenG123/aeye-alliance/master/Data/data_day3.csv")
df_test = pd.read_csv("https://raw.githubusercontent.com/HelenG123/ai-alliance/master/brailleFinalv2.csv")




# using 0-25 to represent a-z instead
target = {}
alphabet = list(string.ascii_lowercase)
number = 0 
for letter in alphabet: 
  target[letter] = number
  number += 1




# Iterate over the CSV files to add the targets
data=[]

for i, row in df.iterrows():
  picture = []
  url = row['Labeled Data']
  label = row['Label']
  curr_target = target[label[10]]
  x = urllib.request.urlopen(url)
  resp = x.read()
  image = np.array(bytearray(resp), dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  # resize image to 28x28x3
  image = cv2.resize(image, (28, 28))
  image = image.astype(np.float32)/255.0
  image = torch.from_numpy(image)
  picture.append(image)
  curr_target=torch.LongTensor([curr_target])
  picture.append(curr_target)
  data.append(picture)

data_test=[]

for i, row in df_test.iterrows():
  picture = []
  url = row['Labeled Data']
  label = row['External ID']
  curr_target = target[label[0]]
  x = urllib.request.urlopen(url)
  resp = x.read()
  image = np.array(bytearray(resp), dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  # resize image to 28x28x3
  image = cv2.resize(image, (28, 28))
  image = image.astype(np.float32)/255.0
  image = torch.from_numpy(image)
  picture.append(image)
  curr_target=torch.LongTensor([curr_target])
  picture.append(curr_target)
  data_test.append(picture)



# Load the dataset
batch_size = 10
batch_size_test = 5

train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size_test, shuffle=False)

# import the nn.Module class
import torch.nn as nn


# defines the convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            # 3x28x28
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # 16x28x28
            nn.MaxPool2d(kernel_size=2)
            # 16x14x14
        )
        # 16x14x14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # 32x14x14
            nn.MaxPool2d(kernel_size=2)
            # 32x7x7
        )
        # linearly
        self.block3 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 500),
            nn.Linear(500, 300),
            nn.Linear(300, 100),
            nn.Linear(100, 26)
        )

        # 1x26

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        # flatten the dataset=
        out = out.view(-1, 32 * 7 * 7)
        out = self.block3(out)

        return out


# convolutional neural network model
model = CNN()

# Set the learning rate, criterion, & optimizer
learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

t0 = time.time()

# variable to store the total loss
total_loss = []

# for loop that iterates over all the epochs
num_epochs = 10
for epoch in range(num_epochs):

    # variables to store/keep track of the loss and number of iterations
    train_loss = 0
    num_iter = 0

    # train the model
    model.train()

    # Iterate over data.
    for i, (images, labels) in enumerate(train_loader):

        # need to permute so that the images are of size 3x28x28
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Zero the gradient buffer
        # resets the gradient after each epoch so that the gradients don't add up
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # calculate the loss
        loss = criterion(outputs, labels.view(-1))
        print('loss:', loss)
        total_loss.append(loss)
        # Backward
        loss.backward()

        # Optimize
        # loops through all parameters and updates weights by using the gradients
        optimizer.step()
        # update the training loss and number of iterations
        train_loss += loss.data[0]
        num_iter += 1

    print('Epoch: {}, Loss: {:.4f}'.format(
        epoch + 1, train_loss / num_iter))

    # evaluate the model
    model.eval()

    correct = 0
    total = 0

    # Iterate over data.
    for images, labels in test_loader:

        # need to permute so that the images are of size 3x28x28
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels.view(-1))
        _, predicted = torch.max(outputs.data, 1)

        # Statistics
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy on the test set: {}%'.format(100 * correct / total))

tf = time.time()
print()
print("time: {} s".format(tf - t0))

