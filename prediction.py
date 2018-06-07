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
from PIL import Image


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


model = CNN()
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load("model.pth"))


# image = Image.open('./Dataset_3/j.jpg')
image = Image.open('z.jpg')
image = np.array(image)
image = cv2.resize(image, (28, 28))
image = image.astype(np.float32) / 255.0
image = torch.from_numpy(image[None, :, :, :])
image = image.permute(0, 3, 1, 2)
predicted_tensor = model(image)
_, predicted_letter = torch.max(predicted_tensor, 1)

print(predicted_letter)
print(chr(97+predicted_letter))
