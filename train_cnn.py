import os
import numpy as np
import torch, torchvision
import pandas as pd

from torchsummary import summary
from torch.utils.data import Subset

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from cnn_model import Net
from utils import ButterflyDataset


def k_fold_training(dataset, 
                    n_epochs, 
                    batch_size=4,
                    weight_decay=0.1,
                    dropout=0.1,
                    learning_rate=5e-4,
                    USE_GPU=False,
                    metric_output="tmp",
                    n_splits=5
                    ):

    # random_state ensures reproducibility of the splits.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Starting {n_splits}-fold cross-validation...")
    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        # Subset is a PyTorch utility to create a dataset using a list of indices
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        # dataloaders
        train_loader = torch.utils.data.DataLoader(train_subset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                )
        val_loader = torch.utils.data.DataLoader(val_subset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                )
        model = Net()

        train_model(model,
                    train_loader,
                    val_loader,
                    n_epochs
                    )


def train_model(model,
                train_loader,
                val_loader,
                n_epochs,
                learning_rate=1e-3,
                weight_decay=1e-6,
                dropout=0.1,
                output_file="tmp",
                fold=1
                ):
    
    USE_GPU = False
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize model, loss function, and optimizer
    model = Net(dropout=dropout)
    model.to(device)

    summary(model, (3, 224, 224))

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

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

    df = pd.DataFrame()

    df["Train Loss"] = train_loss_history
    df["Val Loss"] = val_loss_history
    df["Accuracy"] = val_accuracy_history

    df = df.to_csv(f"Test Metrics/{output_file}.csv")

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


def get_transforms(mode='train'):
    if mode == 'train':
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    elif mode == 'eval': # no stochastic transforms, or use p=0
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # convert images to tensors
        ])
        for tf in tfs.transforms:
            if hasattr(tf, 'train'):
                tf.eval()  # set to eval mode if applicable # type: ignore
    else:
        raise ValueError(f"Unknown mode {mode} for transforms, must be 'train' or 'eval'.")
    return tfs

def main():
    model = Net()

    transforms = get_transforms()

    # Define dataset
    dataset = ButterflyDataset(transform=transforms)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Access the first sample in the training subset
    # train_data[0] calls the __getitem__ method of the underlying Dataset
    first_image_tensor, first_label = train_data[0] 

    # Check the size (shape) of the image tensor
    # The output will typically be (Channels, Height, Width) e.g., (3, 224, 224)
    tensor_shape = first_image_tensor.shape

    print(f"Shape of a single image tensor (C, H, W): {tensor_shape}") 
    
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                )
    val_loader = torch.utils.data.DataLoader(val_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            )

    train_model(model,
                train_loader,
                val_loader,
                n_epochs=10
                )
    

if __name__ == "__main__":
    main()