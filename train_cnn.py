import os
import numpy as np
import torch
import pandas as pd

from torchsummary import summary
from torch.utils.data import Subset
from torchvision import transforms

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from cnn_model import ButterflyNet, BrainTumorNet, Animal10Net
from utils import ButterflyDataset, BrainTumorDataset, Animal10Dataset

from pathlib import Path

def train_model(model,
                train_loader,
                val_loader,
                n_epochs,
                learning_rate=1e-3,
                weight_decay=1e-6,
                dropout=0.1,
                output_file="tmp",
                input_size = (1,1,1)
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
    summary(model, input_size)

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

        train_loss, model = train(model, train_loader, optimizer, device, criterion)
        # Evaluate on validation set
        val_loss, accuracy = evaluate(model, val_loader, criterion, device)

        # Print output
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(accuracy)

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


def main(task):
    if task == "Butterfly":
        dataset = ButterflyDataset()
        model = ButterflyNet(dataset.input_size)

    elif task == "BrainTumor":
        dataset = BrainTumorDataset()
        model = BrainTumorNet(dataset.input_size)
    elif task == "Animals":
        dataset = Animal10Dataset()
        model = Animal10Net(dataset.input_size)


    # Define train/val/test split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 16
    
    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=os.cpu_count(),
                                                pin_memory=True
                                                )
    val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=os.cpu_count(),
                                            pin_memory=True
                                            )

    train_model(model,
                train_loader,
                val_loader,
                n_epochs=10,
                input_size=dataset.input_size,
                output_file=task
                )
    

if __name__ == "__main__":
    # main("Butterfly")
    # main("BrainTumor")
    main("Animals")