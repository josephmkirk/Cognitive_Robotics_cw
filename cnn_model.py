import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions

class ButterflyNet(nn.Module):
    def __init__(self, input_size, num_classes=75, dropout=0.1):
        super(ButterflyNet, self).__init__()

        # Output layer sizes
        out_sizes = [16,32,64,96,128,164]

        self.conv1 = nn.Conv2d(input_size[0], out_sizes[0], kernel_size=3, padding=1)
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

        self.dropout = nn.Dropout(dropout)

        FINAL_DIM_1 = input_size[1] // (2 ** len(out_sizes)) # 224 / 8 = 28
        FINAL_DIM_2 = input_size[2] // (2 ** len(out_sizes)) # 224 / 8 = 28
        FC_INPUT_FEATURES = out_sizes[-1] * FINAL_DIM_1 * FINAL_DIM_2 # 128 * 28 * 28 = 100352

        self.fc1 = nn.Linear(FC_INPUT_FEATURES, 128)  # flatten after conv+pool
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 82)
        self.fc4 = nn.Linear(82, num_classes)

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

        x = self.conv6(x)
        # x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class BrainTumorNet(nn.Module):
    def __init__(self, input_size, num_classes=2, dropout=0.1):
        super(BrainTumorNet, self).__init__()

        # Output layer sizes
        out_sizes = [16,32,64,96,128]

        self.conv1 = nn.Conv2d(input_size[0], out_sizes[0], kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_sizes[0])
        
        self.conv2 = nn.Conv2d(out_sizes[0], out_sizes[1], kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_sizes[1])

        self.conv3 = nn.Conv2d(out_sizes[1], out_sizes[2], kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(out_sizes[2])

        self.conv4 = nn.Conv2d(out_sizes[2], out_sizes[3], kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(out_sizes[3])

        self.conv5 = nn.Conv2d(out_sizes[3], out_sizes[4], kernel_size=3, padding=1)
        # self.batchnorm4 = nn.BatchNorm2d(out_sizes[3])

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout)

        FINAL_DIM_1 = input_size[1] // (2 ** len(out_sizes))
        FINAL_DIM_2 = input_size[2] // (2 ** len(out_sizes))
        FC_INPUT_FEATURES = out_sizes[-1] * FINAL_DIM_1 * FINAL_DIM_2

        self.fc1 = nn.Linear(FC_INPUT_FEATURES, 64)  # flatten after conv+pool
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, num_classes)

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
        # x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class Animal10Net(nn.Module):
    def __init__(self, input_size, num_classes=10, dropout=0.1):
        super(Animal10Net, self).__init__()

        # Output layer sizes
        out_sizes = [16,32,64,96]

        self.conv1 = nn.Conv2d(input_size[0], out_sizes[0], kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_sizes[0])
        
        self.conv2 = nn.Conv2d(out_sizes[0], out_sizes[1], kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_sizes[1])

        self.conv3 = nn.Conv2d(out_sizes[1], out_sizes[2], kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(out_sizes[2])

        self.conv4 = nn.Conv2d(out_sizes[2], out_sizes[3], kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(out_sizes[3])


        self.pool = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # FINAL_DIM_1 = input_size[1] // (2 ** len(out_sizes))
        # FINAL_DIM_2 = input_size[2] // (2 ** len(out_sizes))
        # FC_INPUT_FEATURES = out_sizes[-1] * FINAL_DIM_1 * FINAL_DIM_2

        FC_INPUT_FEATURES = out_sizes[-1]

        self.fc1 = nn.Linear(FC_INPUT_FEATURES, 48)  # flatten after conv+pool
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

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

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
