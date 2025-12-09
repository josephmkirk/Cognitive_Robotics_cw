import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

class Animal10Net(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(Animal10Net, self).__init__()

        # Output layer sizes
        out_sizes = [16,32,64,96,128,256]

        self.conv1 = nn.Conv2d(3, out_sizes[0], kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_sizes[0])
        
        self.conv2 = nn.Conv2d(out_sizes[0], out_sizes[1], kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_sizes[1])

        self.conv3 = nn.Conv2d(out_sizes[1], out_sizes[2], kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(out_sizes[2])

        self.conv4 = nn.Conv2d(out_sizes[2], out_sizes[3], kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(out_sizes[3])

        self.conv5 = nn.Conv2d(out_sizes[3], out_sizes[4], kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(out_sizes[4])

        self.conv6 = nn.Conv2d(out_sizes[4], out_sizes[5], kernel_size=3, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(out_sizes[5])

        self.pool = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(out_sizes[-1], 256)  # flatten after conv+pool
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = F.relu(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
