import os

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold

from cnn_model import Animal10Net
from utils import Animal10Dataset, CaltechDataset
from train_cnn import train_model

import numpy as np


def objective(trial):
    # ---- Hyperparameter search space ----
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 10, 50)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")
    criterion = nn.CrossEntropyLoss()
    
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_sub, val_sub = torch.utils.data.random_split(train_data, [train_size, val_size])

        
    # 2. Create DataLoaders for the current fold
    train_loader = DataLoader(train_sub,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True
                                )
    # Note: Validation batch size is typically kept constant for testing
    val_loader = DataLoader(val_sub,
                            batch_size=256,
                            shuffle=False,
                            num_workers=os.cpu_count(),
                            pin_memory=True
                            )

    # 3. Model, loss, optimizer (MUST be re-initialized for each fold)
    if MODEL_NAME == "Animals":
        model = Animal10Net(num_classes=10, dropout=dropout).to(device)
    elif MODEL_NAME == "Caltech":
        model = Animal10Net(num_classes=99, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Training + Validation Loop for the current fold ----
    for epoch in range(epochs):
        
        print(f"Epoch [{epoch+1}/{epochs}]")

        # ----- Training Step (Identical to your original code) -----
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # ----- Validation Step (Identical to your original code) -----
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)

            # Calculate accuracy for the current epoch and fold
            accuracy = correct / total
            
            # Report the intermediate result for pruning (using the current fold's accuracy)
            # Optuna's pruning logic will treat this as a single sequential training run, 
            # which is an acceptable approximation for CV
            trial.report(accuracy, epoch) 

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        print(f"Epoch [{epoch+1}/{epochs}], accuracy: {accuracy}")

    return accuracy

def save_study_plots(study):
    outdir=f"optuna_plots/{MODEL_NAME}"

    import os
    os.makedirs(outdir, exist_ok=True)

    plots = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "param_importances": optuna.visualization.plot_param_importances,
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
        "slice": optuna.visualization.plot_slice,
        "edf": optuna.visualization.plot_edf,
    }

    for name, func in plots.items():
        fig = func(study)
        fig.write_html(f"{outdir}/{name}.html")
        # fig.write_image(f"{outdir}/{name}.png")

    # Save best hyperparameters
    trial = study.best_trial
    import json
    with open(f"{outdir}/best_hparams.json", "w") as f:
        json.dump(trial.params, f, indent=4)

def start_opt():

    # Create and run Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=50)

    # Save plots
    save_study_plots(study)    

    # Print best trial
    print("Best Trial:")
    trial = study.best_trial

    print(f"  Accuracy: {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Return best parameters for testing
    return trial

def setup_tuning(dataset, name, model):
    # Global variable to track which model is being tuned
    global MODEL_NAME, train_data
    MODEL_NAME = name

    print(f"Training on {name} Dataset")

    # Create train-val split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size]) 

    best_trial = start_opt()

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=best_trial.params["batch_size"],
                                                shuffle=True,
                                                num_workers=os.cpu_count(),
                                                pin_memory=True
                                                )
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=256,
                                            shuffle=False,
                                            num_workers=os.cpu_count(),
                                            pin_memory=True
                                            )

    train_model(
        model(input_size=dataset.input_size, dropout=best_trial.params["dropout"]),
        train_loader=train_loader,
        val_loader=test_loader,
        n_epochs=best_trial.params["epochs"],
        learning_rate=best_trial.params["lr"],
        weight_decay=best_trial.params["weight_decay"],
        output_file=f"BEST_{name}"
    )

# Running the optimisation
if __name__ == "__main__":
    setup_tuning(Animal10Dataset(caching=False), "Animals", Animal10Net)
    setup_tuning(CaltechDataset(cachiing=False), "Caltech", Animal10Net)
    

