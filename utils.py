import kagglehub
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

        
class ButterflyDataset():
    def __init__(self):
        # Set kaggle API token as environment variable
        os.environ['KAGGLE_API_TOKEN'] = 'KGAT_3bd7023f3e62101d2b008a2f3b4168de'

        # Download latest version, returns dataset location
        self.dataset_path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")
        
        self.name = "Butterfly"


        # Read csv for labels
        df = pd.read_csv(f"{self.dataset_path}/Training_set.csv")
        df['filename'] = df['filename'].apply(
            lambda img_filename: os.path.join(f"{self.dataset_path}/train", img_filename)
        )

        self.filepaths = np.array(df["filename"])
        self.labels = np.array(df["label"])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ])

        self.input_size = (3,224,224)

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

class BrainTumorDataset():
    def __init__(self):
        # Kagglehub API token
        os.environ['KAGGLE_API_TOKEN'] = 'KGAT_3bd7023f3e62101d2b008a2f3b4168de'

        # Download latest version, returns dataset location
        self.dataset_path = kagglehub.dataset_download("preetviradiya/brian-tumor-dataset")

        self.name = "BrainTumor"

        df = pd.read_csv(f"{self.dataset_path}/metadata_rgb_only.csv")
        df['filename'] = df.apply(self.create_filepath, axis=1)

        self.filepaths = np.array(df["filename"])
        self.labels = np.array(df["class"])

        self.transform = transforms.Compose(
            [
                transforms.Resize((256,256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ]
        )

        self.input_size = (3,256,256)

        self.encoder=LabelEncoder()
        self.labels = self.encoder.fit_transform(np.array(self.labels))

    def create_filepath(self, row):
        """Function to calculate the filepath based on the row's class."""
        if row['class'] == 'tumor':
            folder = 'Brain Tumor'
        elif row['class'] == 'normal':
            folder = 'Healthy'

        return os.path.join(
            f"{self.dataset_path}/Brain Tumor Data Set/Brain Tumor Data Set/{folder}",
            row['image']
        )

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
    

class Animal10Dataset():
    def __init__(self, caching=False):
        # Kagglehub API token
        os.environ['KAGGLE_API_TOKEN'] = 'KGAT_3bd7023f3e62101d2b008a2f3b4168de'

        # Download latest version, returns dataset location
        self.dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")

        self.name = "Animal10"
        self.caching = caching

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

        full_df = pd.DataFrame()
        full_df["filepath"] = all_filepaths
        full_df["label"] = all_labels

        # Define the fraction of data you want to KEEP (75% or 0.75)
        FRACTION_TO_KEEP = 0.75

        # Use a fixed random state for reproducibility
        RANDOM_STATE = 42

        # Group the DataFrame by 'label' and sample 75% from each group
        df_reduced = full_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=FRACTION_TO_KEEP, random_state=RANDOM_STATE)
        )

        print(f"Original dataset size: {len(full_df)}")
        print(f"Reduced dataset size (approx {int(FRACTION_TO_KEEP*100)}%): {len(df_reduced)}")

        if caching:
            cache_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(), # Converts image to a tensor (H, W, C) -> (C, H, W) and scales to [0, 1]
            ])

            self.cached_images = []

            print("Starting in-memory image caching...")
            counter = 0
            for img_path in df_reduced["filepath"]: # Iterate through the collected paths
                try:
                    # Load, convert to RGB, and apply basic transforms for caching
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = cache_transform(image)
                    
                    # Store the tensor in the list
                    self.cached_images.append(image_tensor)
                    
                except Exception as e:
                    print(f"Skipping corrupted file: {img_path}. Error: {e}")
                    # You must ensure the filepaths and labels lists are synchronized if you skip a file!

                if counter % 1000 == 0:
                    print(f"[{counter}/{len(df_reduced['filepath'])}] in cache")
                counter += 1            

            print(f"Caching complete. {len(self.cached_images)} tensors loaded into RAM.")
            
            
            # Store augmentations separately
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30)
            ]) # Keep the augmentation pipeline if passed
        else:

            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor()
            ]) # Keep the augmentation pipeline if passed

        self.filepaths = np.array(df_reduced["filepath"])
        self.labels = np.array(df_reduced["label"])
 
        self.input_size = (3,256,256)

        self.encoder=LabelEncoder()
        self.labels = self.encoder.fit_transform(np.array(self.labels))


    def __len__(self):
        # Returns the total number of samples (images) in the dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        if self.caching:
            # Retrieve the pre-loaded tensor and label directly from memory
            image = self.cached_images[idx]
        else:
            img_path = self.filepaths[idx]
            # Read Image (PIL is standard for PyTorch transforms)
            image = Image.open(img_path).convert('RGB') # CNNs use 3-channel RGB images
        
            
        if self.transform:
            image = self.transform(image)

        label_id = self.labels[idx]

        # Return the tensor and label
        return image, torch.tensor(label_id, dtype=torch.long)
    

class CaltechDataset():
    def __init__(self, caching=False):
        # Kagglehub API token
        os.environ['KAGGLE_API_TOKEN'] = 'KGAT_3bd7023f3e62101d2b008a2f3b4168de'

        # Download latest version, returns dataset location
        self.dataset_path = kagglehub.dataset_download("imbikramsaha/caltech-101")

        self.name = "Animal10"
        self.caching = caching

        all_filepaths = []
        all_labels = []
            
        categories = [
            "anchor",
            "dollar_bill",
            "metronome",
            "helicopter",
            "Motorbikes",
            "airplanes",
            "binocular",
            "chandelier",
            "electric_guitar",
            "grand_piano",
            "soccer_ball",
            "gramophone",
            "laptop",
            "watch"
        ]

        print(f"Num categories: {len(categories)}")

        for category in categories:
            # Construct the full path to the current animal folder
            folder_path = os.path.join(f"{self.dataset_path}/caltech-101", category)

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

        full_df = pd.DataFrame()
        full_df["filepath"] = all_filepaths
        full_df["label"] = all_labels

        # Define the fraction of data you want to KEEP (75% or 0.75)
        FRACTION_TO_KEEP = 1

        # Use a fixed random state for reproducibility
        RANDOM_STATE = 42

        # Group the DataFrame by 'label' and sample 75% from each group
        df_reduced = full_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=FRACTION_TO_KEEP, random_state=RANDOM_STATE)
        )

        print(f"Original dataset size: {len(full_df)}")
        print(f"Reduced dataset size (approx {int(FRACTION_TO_KEEP*100)}%): {len(df_reduced)}")

        if caching:
            cache_transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(), # Converts image to a tensor (H, W, C) -> (C, H, W) and scales to [0, 1]
            ])

            self.cached_images = []

            print("Starting in-memory image caching...")
            counter = 0
            for img_path in df_reduced["filepath"]: # Iterate through the collected paths
                try:
                    # Load, convert to RGB, and apply basic transforms for caching
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = cache_transform(image)
                    
                    # Store the tensor in the list
                    self.cached_images.append(image_tensor)
                    
                except Exception as e:
                    print(f"Skipping corrupted file: {img_path}. Error: {e}")
                    # You must ensure the filepaths and labels lists are synchronized if you skip a file!

                if counter % 1000 == 0:
                    print(f"[{counter}/{len(df_reduced['filepath'])}] in cache")
                counter += 1            

            print(f"Caching complete. {len(self.cached_images)} tensors loaded into RAM.")
            
            
            # Store augmentations separately
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ]) # Keep the augmentation pipeline if passed
        else:

            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor()
            ]) # Keep the augmentation pipeline if passed

        self.filepaths = np.array(df_reduced["filepath"])
        self.labels = np.array(df_reduced["label"])
 
        self.input_size = (3,256,256)

        self.encoder=LabelEncoder()
        self.labels = self.encoder.fit_transform(np.array(self.labels))


    def __len__(self):
        # Returns the total number of samples (images) in the dataset
        return len(self.filepaths)

    def __getitem__(self, idx):
        if self.caching:
            # Retrieve the pre-loaded tensor and label directly from memory
            image = self.cached_images[idx]
        else:
            img_path = self.filepaths[idx]
            # Read Image (PIL is standard for PyTorch transforms)
            image = Image.open(img_path).convert('RGB') # CNNs use 3-channel RGB images
        
            
        if self.transform:
            image = self.transform(image)

        label_id = self.labels[idx]

        # Return the tensor and label
        return image, torch.tensor(label_id, dtype=torch.long)
    