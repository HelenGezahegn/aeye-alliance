
# coding: utf-8

# In[3]:


# to measure run-time
import time

# to shuffle data
import random

# for plotting
import matplotlib.pyplot as plt

# pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim

# for csv dataset
import torchvision
import csv
import os
import pandas as pd 
from urllib import request
import requests

# import statements for iterating over csv file
import cv2
import numpy as np
import urllib.request

# to get the alphabet
import string


# In[4]:


# Upload and read the csv files from the github repo
# change once get more data
df = pd.read_csv("https://raw.githubusercontent.com/HelenG123/aeye-alliance/master/Labelled%20Data/symbols_letters.csv")


# In[5]:


# generate the targets 
# the targets are one hot encoding vectors

alphabet = list(string.ascii_lowercase)
target = {}

# Initalize a target dict that has letters as its keys and empty one-hot encoding vectors of size 37 as its values
for letter in alphabet: 
    target[letter] = [0] * 37

# Do the one-hot encoding for each letter now 
curr_pos = 0 
for curr_letter in target.keys():
    target[curr_letter][curr_pos] = 1
    curr_pos += 1  

# extra symbols 
symbols = [' ', '#', '.', ',', ':', '\'', '-', ';', '?', '!', 'C'] # C stands for CAPS

# create vectors
for curr_symbol in symbols:
    target[curr_symbol] = [0] * 37

# create one-hot encoding vectors
for curr_symbol in symbols:
    target[curr_symbol][curr_pos] = 1
    curr_pos += 1

print(target)


# In[7]:


t0 = time.time()

# collect all data from the csv file
data=[]

# iterate over csv file
for i, row in df.iterrows():
    # store the image and label
    picture = []
    url = row['Labeled Data']
    label = row['Label']
    curr_target = target[label[11]]
    x = urllib.request.urlopen(url)
    resp = x.read()
    image = np.array(bytearray(resp), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # resize image to 28x28x3
    image = cv2.resize(image, (28, 28))
    # normalize to 0-1
    image = image.astype(np.float32)/255.0
    image = torch.from_numpy(image)
    picture.append(image)
    # convert the target to a long tensor
    curr_target=torch.LongTensor([curr_target])
    picture.append(curr_target)
    # append the current image & target
    data.append(picture)
    
tf = time.time()
print("time: {} s" .format(tf-t0))


# In[8]:


# create a dictionary of all the characters 
# array of all the characters
characters = alphabet + symbols

index2char = {}
number = 0
for char in characters: 
    index2char[number] = char
    number += 1

print(index2char)


# In[9]:


# find the number of each character in a dataset
def num_chars(dataset, index2char):
    chars = {}
    for _, label in dataset:
        char = index2char[int(torch.argmax(label))]
        # update
        if char in chars:
            chars[char] += 1
        # initialize
        else:
            chars[char] = 1
    return chars


# In[10]:


print(num_chars(data, index2char))


# In[23]:


# Create dataloader objects

# shuffle all the data
random.shuffle(data)

# batch sizes for train, test, and validation
batch_size_train = 10
batch_size_test = 3
batch_size_validation = 3

# 3130
# splitting data to get training, test, and validation sets
# change once get more data
# 2504 for train
train_dataset = data[:2504]
# test has 313
test_dataset = data[2504:2817]
# validation has 313
validation_dataset = data[2817:]

# create the dataloader objects
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size_validation, shuffle=True)

print(len(train_loader))
print(len(test_loader))
print(len(validation_loader))


# In[24]:


print(len(num_chars(test_dataset, index2char)))
print(num_chars(test_dataset, index2char))


# In[25]:


# to check if a dataset is missing a char
test_chars = num_chars(test_dataset, index2char)

num = 0
for char in characters:
    if char in test_chars:
        num += 1
    else:
        break
print(num) 


# In[33]:


# defines the convolutional neural network

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            #3x28x28
            nn.Conv2d(in_channels=3, 
                      out_channels=16, 
                      kernel_size=5,
                      stride=1, 
                      padding=2),
            # batch normalization
            # nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True), 
            #16x28x28
            nn.MaxPool2d(kernel_size=2),
            #16x14x14
            nn.LeakyReLU()
        )
        #16x14x14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=5,
                      stride=1, 
                      padding=2),
            # batch normalization
            # nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True), 
            #32x14x14
            nn.MaxPool2d(kernel_size=2),
            #32x7x7
            nn.LeakyReLU()
        ) 
        # linearly 
        self.block3 = nn.Sequential(
            nn.Linear(32*7*7, 100),
            # batch normalization
            # nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 37)
        )
        #1x37
    
    def forward(self, x): 
        out = self.block1(x)
        out = self.block2(out)
        # flatten the dataset
        out = out.view(-1, 32*7*7)
        out = self.block3(out)
        
        return out

# convolutional neural network model
model = CNN()

# print summary of the neural network model to check if everything is fine. 
print(model)
print("# parameter: ", sum([param.nelement() for param in model.parameters()]))


# In[34]:


# setting the learning rate
learning_rate = 2e-4

# Using a variable to store the cross entropy method
criterion = nn.CrossEntropyLoss()

# Using a variable to store the optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# In[35]:


# function to get which characters were missclassified
def get_chars(indices, incorrect_dict, index2char):
    for i in indices:
        char = index2char[i]
        # update
        if char in incorrect_dict:
            incorrect_dict[char] += 1
    return incorrect_dict


# In[36]:


t0 = time.time()

# list of all train_losses 
train_losses = []

# list of all validation losses 
validation_losses = []

# for loop that iterates over all the epochs
num_epochs = 20
for epoch in range(num_epochs):
    
    # variables to store/keep track of the loss and number of iterations
    train_loss = 0
    num_iter_train = 0

    # train the model
    model.train()
    
    # Iterate over train_loader
    for i, (images, labels) in enumerate(train_loader):  
        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Zero the gradient buffer
        # resets the gradient after each epoch so that the gradients don't add up
        optimizer.zero_grad()  

        # Forward, get output
        outputs = model(images)
        
        # convert the labels from one hot encoding vectors into integer values 
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1)

        # calculate training loss
        loss = criterion(outputs, y_true)
        
        # Backward (computes all the gradients)
        loss.backward()
        
        # Optimize
        # loops through all parameters and updates weights by using the gradients 
        # takes steps backwards to optimize (to reach the minimum weight)
        optimizer.step()
        
        # update the validation and number of iterations
        train_loss += loss.data[0]
        num_iter_train += 1

    print('Epoch: {}'.format(epoch+1))
    print('Training Loss: {:.4f}'.format(train_loss/num_iter_train))
    # append training loss over all the epochs
    train_losses.append(train_loss/num_iter_train)

    # evaluate the model
    model.eval()
    
    # variables to store/keep track of the loss and number of iterations
    validation_loss = 0
    num_iter_validation = 0
    
    # Iterate over validation_loader
    for i, (images, labels) in enumerate(validation_loader):  
        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Forward, get output
        outputs = model(images)

        # convert the labels from one hot encoding vectors to integer values
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1)
        
        # calculate the validation loss
        loss = criterion(outputs, y_true)

        # update the training loss and number of iterations
        validation_loss += loss.data[0]
        num_iter_validation += 1

    print('Validation Loss: {:.4f}'.format(validation_loss/num_iter_validation))
    # append all validation_losses over all the epochs
    validation_losses.append(validation_loss/num_iter_validation)

    num_iter_test = 0
    correct = 0
    incorrect_dict = dict([(c, 0) for c in characters]) # initializing an empty dictionary of all the characters
    
    # Iterate over test_loader
    for images, labels in test_loader:  

        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(0, 3, 1, 2)

        # Forward
        outputs = model(images)

        # convert the labels from one hot encoding vectors into integer values 
        labels = labels.view(-1, 37)
        y_true = torch.argmax(labels, 1)

        # find the index of the prediction
        y_pred = torch.argmax(outputs, 1).type('torch.FloatTensor')
        
        # convert to FloatTensor
        y_true = y_true.type('torch.FloatTensor')

        # find the mean difference of the comparisons
        correct += torch.sum(torch.eq(y_true, y_pred).type('torch.FloatTensor'))

        # find missclassified characters
        # convert to numpy arrays
        np_y_true = np.array(y_true)
        np_y_pred = np.array(y_pred)        
        # which labels were given for the misclassified characters
        incorrect_pred = np_y_pred[np_y_pred != np_y_true]
        # what the correct labels should be 
        incorrect_true = np_y_true[np_y_pred != np_y_true]
        
        # update incorrect_dict
        if len(incorrect_true) > 0:
            incorrect_dict = get_chars(incorrect_true, incorrect_dict, index2char)
        
    print('Accuracy on the test set: {:.4f}%'.format(correct/len(test_dataset) * 100))
    print('Number of missclassified characters:', incorrect_dict)
    print()

# calculate time it took to train the model
tf = time.time()
print()
print("time: {} s" .format(tf-t0))


# In[37]:


# learning curve function
def plot_learning_curve(train_losses, validation_losses):
    # plot the training and validation losses
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    plt.plot(train_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.legend(loc=1)


# In[38]:


# plot the learning curve
plt.title("Learning Curve (Loss vs Number of Epochs)")
plot_learning_curve(train_losses, validation_losses)

