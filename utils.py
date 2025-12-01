import kagglehub
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.preprocessing import LabelEncoder

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
        
class ButterflyDataset():
    def __init__(self, transform=None):
        # Set kaggle API token as environment variable
        os.environ['KAGGLE_API_TOKEN'] = 'KGAT_3bd7023f3e62101d2b008a2f3b4168de'

        # Download latest version, returns dataset location
        self.dataset_path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")
        
        # Read csv for labels
        df = pd.read_csv(f"{self.dataset_path}/Training_set.csv")
        df['filename'] = df['filename'].apply(
            lambda img_filename: os.path.join(f"{self.dataset_path}/train", img_filename)
        )

        self.filepaths = np.array(df["filename"])
        self.labels = np.array(df["label"])

        self.transform = transform

        self.encoder=LabelEncoder()
        self.labels = self.encoder.fit_transform(np.array(self.labels))


    def __len__(self):
        # Returns the total number of samples (images) in the dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        # This function defines what happens when you index the dataset (e.g., dataset[5])
        img_path = self.filepaths[idx]
        label_id = self.labels[idx]

        # Read Image (PIL is standard for PyTorch transforms)
        image = Image.open(img_path).convert('RGB') # CNNs use 3-channel RGB images
        
        # Apply Transformations
        if self.transform:
            image = self.transform(image)
        
        # Return the image tensor and its encoded label
        return image, torch.tensor(label_id, dtype=torch.long)
    