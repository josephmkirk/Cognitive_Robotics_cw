import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

class Net(nn.Module):
    def __init__(self, num_classes=75, dropout=0.1):
        super(Net, self).__init__()

        # Output layer sizes
        out_sizes = [16,32,64,96,128]

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

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout)

        FINAL_DIM = 256 // (2 ** len(out_sizes)) # 224 / 8 = 28
        FC_INPUT_FEATURES = out_sizes[-1] * FINAL_DIM * FINAL_DIM # 128 * 28 * 28 = 100352

        self.fc1 = nn.Linear(FC_INPUT_FEATURES, 96)  # flatten after conv+pool
        self.fc2 = nn.Linear(96, 82)
        self.fc3 = nn.Linear(82, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        # x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        # x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        # x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        # x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
