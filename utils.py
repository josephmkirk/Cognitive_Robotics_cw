import kagglehub
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def download_dataset():
    # Set kaggle API token as environment variable
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_3bd7023f3e62101d2b008a2f3b4168de'

    # Download latest version, returns dataset location
    return kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")


def prepare_data(dataset_path, include_val=False):
    # Can only use training data as that's only data labelled
    df = pd.read_csv(f"{dataset_path}/Training_set.csv")
    df.head()

    X = df["filename"]
    y = df["label"]

    # Convert string labels into numerical integers (0, 1, 2, ... 74)
    # Maybe 1 hot encoding for CNNs
    le = LabelEncoder()
    y = le.fit_transform(np.array(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y # Use stratify if you want balanced classes across splits
    )

    if include_val:
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, 
            y_test, 
            test_size=0.5, 
            random_state=42, 
            stratify=y_test # Stratify on the remaining set's labels
        )

    print(f"--- Dataset shapes ---")
    print(f"Train set (80%): {X_train.shape}")

    if include_val:
        print(f"Validation set (10%): {X_val.shape}")
    print(f"Test set (10%): {X_test.shape}")

    if include_val:
        data= {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "encoder": le
        }
    else:
        data= {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "encoder": le
        }
        
    return data
         
        