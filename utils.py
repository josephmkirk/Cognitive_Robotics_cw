import kagglehub
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

class Animal10Dataset():
    def __init__(self):
        # Kagglehub API token
        os.environ['KAGGLE_API_TOKEN'] = 'KGAT_3bd7023f3e62101d2b008a2f3b4168de'

        # Download latest version, returns dataset location
        self.dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")

        self.name = "Animal10"
        self.num_classes = 10

        # Folder names are in italian
        translate = {
            "cane": "dog",
            "cavallo": "horse",
            "elefante": "elephant",
            "farfalla": "butterfly",
            "gallina": "chicken",
            "gatto": "cat",
            "mucca": "cow",
            "pecora": "sheep",
            "scoiattolo": "squirrel",
            "ragno": "spider"
        }

        all_filepaths = []
        all_labels = []

        # Iterate through the known list of animal folders
        for animal_label in translate.keys():
            
            # Construct the full path to the current animal folder
            folder_path = os.path.join(f"{self.dataset_path}/raw-img", animal_label)

            # Check if the folder actually exists before trying to read it
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder not found: {folder_path}. Skipping.")
                continue
                
            # Iterate through files directly in the animal folder
            for filename in os.listdir(folder_path):
                    
                    # Construct the full filepath
                    full_path = os.path.join(folder_path, filename)
                    
                    # Store the data
                    all_filepaths.append(full_path)
                    all_labels.append(translate[animal_label]) # The folder name is the label

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ])

        self.filepaths = np.array(all_filepaths)
        self.labels = np.array(all_labels)
 
        self.input_size = (3,256,256)

        # Define encoder for categorisation
        self.encoder=LabelEncoder()
        self.labels = self.encoder.fit_transform(np.array(self.labels))


    def __len__(self):
        # Returns the total number of samples (images) in the dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Makes the object indexable
        img_path = self.filepaths[idx]
        # Read Image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms        
        if self.transform:
            image = self.transform(image)

        label_id = self.labels[idx]

        # Return the tensor and label
        return image, torch.tensor(label_id, dtype=torch.long)
    

class CaltechDataset():
    def __init__(self):
        # Kagglehub API token
        os.environ['KAGGLE_API_TOKEN'] = 'KGAT_3bd7023f3e62101d2b008a2f3b4168de'

        # Download latest version, returns dataset location
        self.dataset_path = f"{kagglehub.dataset_download('imbikramsaha/caltech-101')}/caltech-101"

        self.name = "Caltech101"
        self.num_classes = 99

        all_filepaths = []
        all_labels = []
            
        # Removed 3 vague / strange categories
        categories = os.listdir(self.dataset_path)
        items_to_remove = ["BACKGROUND_Google", "Faces", "Faces_easy"]

        # Create a new list containing elements from my_list ONLY IF they are NOT in items_to_remove
        categories = [item for item in categories if item not in items_to_remove]

        for category in categories:
            # Construct the full path to the current animal folder
            folder_path = os.path.join(f"{self.dataset_path}", category)

            # Check if the folder actually exists before trying to read it
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder not found: {folder_path}. Skipping.")
                continue
                
            # Iterate through files directly in the animal folder
            for filename in os.listdir(folder_path):
                    
                    # Construct the full filepath
                    full_path = os.path.join(folder_path, filename)
                    
                    # Store the data
                    all_filepaths.append(full_path)
                    all_labels.append(category) # The folder name is the label
        
        # Did not apply random transforms as some images already had them applied
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.filepaths = np.array(all_filepaths)
        self.labels = np.array(all_labels)
 
        self.input_size = (3,256,256)

        self.encoder=LabelEncoder()
        self.labels = self.encoder.fit_transform(np.array(self.labels))


    def __len__(self):
        # Returns the total number of samples (images) in the dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Makes the object indexable
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
                    
        if self.transform:
            image = self.transform(image)

        label_id = self.labels[idx]

        # Return the tensor and label
        return image, torch.tensor(label_id, dtype=torch.long)
    