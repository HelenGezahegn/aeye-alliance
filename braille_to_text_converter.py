# import statements
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
import requests
from scipy import misc
from io import BytesIO
import urllib
# might need to comment out for now 
# import cv2
import numpy as np
import tensorflow as tf
import urllib.request
from torchvision import transforms

# create training and test datasets
# there are 2 csv files
# one is the training set
# the other is the test set
# Upload and read the csv files from the github repo
df_train = pd.read_csv("https://raw.githubusercontent.com/HelenG123/ai-alliance/master/brailleFinalv2.csv")
df_test = pd.read_csv("")

# iterate over the csv files
data_train=[]
data_test=[]

for i, row in df_train.iterrows():
  picture = []
  url = row['Labeled Data']
  label = row['External ID']
  curr_target = target[label[0]]

  x = urllib.request.urlopen(url)
  resp = x.read()
  image = np.array(bytearray(resp), dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  # resize image
  # becomes 28 x 28 x 3
  image = cv2.resize(image, (28, 28))
#   image = image.astype(np.float32)/255.0
#   image = image.flatten().astype(np.float32)/255.0
  image = torch.from_numpy(image)
  picture.append(image)
  curr_target=torch.Tensor(curr_target)
  picture.append(curr_target)
  data_train.append(picture)

print(image.shape) # these are the dimensions of our image
print(data[0][0])

for i, row in df_test.iterrows():
  picture = []
  url = row['Labeled Data']
  label = row['External ID']
  curr_target = target[label[0]]

  x = urllib.request.urlopen(url)
  resp = x.read()
  image = np.array(bytearray(resp), dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  # resize image
  # becomes 28 x 28 x 3
  image = cv2.resize(image, (28, 28))
#   image = image.astype(np.float32)/255.0
#   image = image.flatten().astype(np.float32)/255.0
  image = torch.from_numpy(image)
  picture.append(image)
  curr_target=torch.Tensor(curr_target)
  picture.append(curr_target)
  data_test.append(picture)

print(image.shape) # these are the dimensions of our image
print(data[0][0])

# create dataloader objects
batch_size = 5

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Generate the targets
import string
alphabet = list(string.ascii_lowercase)

target = {}

# Initalize a target dict that has the letters as its keys and as its value
# an empty one-hot encoding of size 26
for letter in alphabet: 
  target[letter] = [0] * 26

# Do the one-hot encoding for each letter now 
curr_pos = 0 
for curr_letter in target.keys():
  target[curr_letter][curr_pos] = 1
  curr_pos += 1  

# visualize the image
# Display 'y' in Brailles
import matplotlib.pyplot as plt
import numpy as np
dd = data[24][0].numpy()
print('Braille Target: Y/y')
plt.imshow(dd)
plt.show()

# define the model

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
                      padding=2)
            #16x28x28
            nn.MaxPool2d(kernel_size=2),
            #16x14x14
        )
        #16x14x14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=5, 
                      stride=1, 
                      padding=2)
            #32x14x14
            nn.MaxPool2d(kernel_size=2)
            #32x7x7
        ) 
        # linearly 
        self.block3 = nn.Sequential(
            nn.Linear(32*7*7, 500),
            nn.Linear(500, 300),
            nn.Linear(300, 100),
            nn.Linear(100, 26)
        )
        
        #1x26
    
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

# set the learning rate, criterion (cross entropy loss), & optimizer
#setting the learning rate
learning_rate = 1e-3

# Using a variable to store the cross entropy method
criterion = nn.CrossEntropyLoss()

# Using a variable to store the optimizer 
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


# train and evaluate the data
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
    for i, (images, labels) in enumerate(training_set[i]):  
      
       
        print(images.shape)
        
        # need to permute so that the images are of size 3x28x28 
        # essential to be able to feed images into the model
        images = images.permute(2,0,1)
#         images.unsqueeze_(0)

        # Zero the gradient buffer
        # resets the gradient after each epoch so that the gradients don't add up
        optimizer.zero_grad()  
        
        # Forward
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, labels)
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
          epoch+1, train_loss/num_iter))
    
    # evaluate the model
    model.eval()

    correct = 0
    total = 0

    # Iterate over data.
    for images, labels in test_set:  
       
       # Forward
       outputs = model(images)
       loss = criterion(outputs, labels)  
       _, predicted = torch.max(outputs.data, 1)
    
       # Statistics
       total += labels.size(0)
       correct += (predicted == labels).sum()
       
    print('Accuracy on the test set: {}%'.format(100 * correct / total))
tf = time.time()
print()
print("time: {} s" .format(tf-t0))
