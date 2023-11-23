import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

class TriangleNet(nn.Module):
    def __init__(self):
        super(TriangleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Adjust the following line for 640x480 images
        self.fc1 = nn.Linear(32 * 120 * 160, 128)  # Adjusted flatten layer
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 120 * 160)  # Adjusted flatten layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
