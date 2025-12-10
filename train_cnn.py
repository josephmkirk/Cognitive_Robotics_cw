import os
import numpy as np
import torch
import pandas as pd

import random, math

from torchsummary import summary
from torch.utils.data import Subset
from torchvision import transforms

from sklearn.model_selection import KFold, ParameterSampler, train_test_split
from scipy.stats import uniform, randint, loguniform

from cnn_model import Net
from utils import CaltechDataset, Animal10Dataset

from pathlib import Path

def train_model(model,
                train_loader,
                val_loader,
                n_epochs,
                learning_rate=1e-3,
                weight_decay=1e-6,
                save_metrics=False,
                output_file="tmp"
                ):
    
    # Set device
    USE_GPU = True
    # This ensures PyTorch does not see any CUDA devices, forcing CPU usage.
    # Only needed for when GPU is so old that CUDA is outdated
    if not USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Initialize model
    model.to(device)

    # Initialize loss function
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    # Initialize Optimiser
    # Define which layers to apply weight decay to
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if "batchnorm" in name.lower() or "bn" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.Adam([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=learning_rate)

    # Measurement Records
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []

    # Training loop
    for epoch in range(n_epochs):
        print(f"Starting Epoch {epoch+1}...")

        train_loss, model = train(model, train_loader, optimizer, device, criterion)
        # Evaluate on validation set
        val_loss, accuracy = evaluate(model, val_loader, criterion, device)

        # Print output
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(accuracy)

    if save_metrics:
        # Save performance metrics
        df = pd.DataFrame()

        df["Train Loss"] = train_loss_history
        df["Val Loss"] = val_loss_history
        df["Accuracy"] = val_accuracy_history

        # Convert the string path to a Path object
        p = Path("TestMetrics")
        
        # Use the mkdir() method
        # parents=True creates parent directories if they don't exist.
        # exist_ok=True prevents the error if the directory already exists.
        p.mkdir(parents=True, exist_ok=True)
        print(f"Directory 'TestMetrics' ensured to exist.")

        df = df.to_csv(f"TestMetrics/{output_file}.csv")

    return accuracy


def train(model, data_loader, optimizer, device, criterion):
    # Set model to train
    model.train()
    epoch_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_loss = epoch_loss / len(data_loader)
    return train_loss, model


def evaluate(model, data_loader, criterion, device):
    model.to(device)
    model.eval()

    targets = []
    predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            targets.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            total_loss += loss.item()

    accuracy = (np.array(predictions) == np.array(targets)).mean()
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy


def sample_hyperparameters(num_trials):
    """Generates a list of hyperparameter configurations using random sampling."""    

    param_distributions = {
        "epochs": randint(10, 51),
        "learning_rate": loguniform(1e-5, 1e-1),
        "weight_decay": loguniform(1e-6, 1e-3),
        "dropout": uniform(0.0, 0.3),
        "batch_size": [32, 64, 128, 256]
    }
    
    # ParameterSampler generates 'n_iter' unique parameter settings
    sampler = ParameterSampler(
        param_distributions=param_distributions,
        n_iter=num_trials,
        random_state=42 # Set a seed for reproducible sampling
    )

    # Convert the generator output to a list of dictionaries
    random_configs = list(sampler)
    return random_configs


def run_hyperparameter_search(model, dataset, hyperparameters):
    # Define train/val/test split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    accuracies = []

    for i, params in enumerate(hyperparameters):
        print(f"Parameters: {params}")
        # Define dataloaders
        train_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=params["batch_size"],
                                                    shuffle=True,
                                                    num_workers=4,
                                                    pin_memory=True
                                                    )
        val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=params["batch_size"],
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True
                                                )
        
        model.update_dropout(params["dropout"])

        accuracy = train_model(model,
                            train_loader,
                            val_loader,
                            n_epochs=params["epochs"],
                            learning_rate=params["learning_rate"],
                            weight_decay=params["weight_decay"]
                            )
        print(f"Test [{i}/{len(hyperparameters)}], Accuracy: {accuracy}")
        accuracies.append(accuracy)

    df = pd.DataFrame(hyperparameters)
    df["Accuracy"] = accuracies

    df.to_csv(f"{dataset.name}_results.csv")


if __name__ == "__main__":
    print("Running Hyperparameter Samples For Animal-10 Dataset")
    run_hyperparameter_search(Net(num_classes=10), Animal10Dataset(), sample_hyperparameters(2))

    print("Running Hyperparameter Samples For Caltech-101 Dataset")
    run_hyperparameter_search(Net(num_classes=99), CaltechDataset(), sample_hyperparameters(2))