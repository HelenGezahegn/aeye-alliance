import pandas as pd
import numpy as np

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


from sklearn.preprocessing import MultiLabelBinarizer


IMG_PATH = './Data/train-jpg/'
IMG_EXT = '.jpg'
TRAIN_DATA = './Data/train.csv'


class BrailleDataset(Dataset):

    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
            "Some images referenced in the CSV file were not found"

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['letter']

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)

transformations = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])

dset_train = BrailleDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)

# train_loader = DataLoader(dset_train,
#                           batch_size=10,
#                           shuffle=True,
#                           num_workers=1
#                          )
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(2304, 256)
#         self.fc2 = nn.Linear(256, 17)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(x.size(0), -1) # Flatten layer
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.sigmoid(x)
#
# model = Net()
#
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#
# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.binary_cross_entropy(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.data[0]))
#
#
# for epoch in range(1, 2):
#     train(epoch)