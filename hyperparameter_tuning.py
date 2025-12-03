import os

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold

from cnn_model import ButterflyNet, BrainTumorNet, Animal10Net
from utils import ButterflyDataset, BrainTumorDataset, Animal10Dataset
from train_cnn import train_model

import numpy as np


def objective(trial):
    # ---- Hyperparameter search space ----
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 1, 2)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    
    # ---- 5-Fold Cross-Validation Setup ----
    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Store validation accuracy for each fold
    fold_accuracies = []

    # Get the indices of the full dataset
    indices = np.arange(len(train_data))
    
    # Iterate through each fold
    for fold, (train_index, val_index) in enumerate(kf.split(indices)):
        print(f"Starting fold [{fold+1}/{N_SPLITS}]")

        # 1. Create Data Subsets for the current fold
        # Subset creates a view of the original dataset using the indices
        train_subset = Subset(train_data, train_index)
        val_subset = Subset(train_data, val_index)
        
        # 2. Create DataLoaders for the current fold
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        # Note: Validation batch size is typically kept constant for testing
        val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

        # 3. Model, loss, optimizer (MUST be re-initialized for each fold)
        if MODEL_NAME == "Butterfly":
            # Re-initialize the model to ensure independent training for each fold
            model = ButterflyNet(input_size=(3,224,224), dropout=dropout).to(device)
        elif MODEL_NAME == "BrainTumor":
            model = BrainTumorNet(input_size=(3,256,256), dropout=dropout).to(device)
        elif MODEL_NAME == "Animals":
            model = Animal10Net(input_size=(3,256,256), dropout=dropout).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # ---- Training + Validation Loop for the current fold ----
        for epoch in range(epochs):
            
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
            fold_epoch_accuracy = correct / total
            
            # Report the intermediate result for pruning (using the current fold's accuracy)
            # Optuna's pruning logic will treat this as a single sequential training run, 
            # which is an acceptable approximation for CV
            trial.report(fold_epoch_accuracy, epoch * N_SPLITS + fold) 

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Store the final accuracy from the last epoch of this fold
        fold_accuracies.append(fold_epoch_accuracy)
        
    # ---- Return the Final Metric ----
    # The final metric for the Optuna trial is the average accuracy across all folds
    average_accuracy = np.mean(fold_accuracies)
    
    return average_accuracy

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
        fig.write_image(f"{outdir}/{name}.png")

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

    study.optimize(objective, n_trials=2)

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
                                                )
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=256,
                                            shuffle=False,
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
    setup_tuning(Animal10Dataset(), "Animals", Animal10Net)
    setup_tuning(ButterflyDataset(), "Butterfly", ButterflyNet)
    setup_tuning(BrainTumorDataset(), "BrainTumor", BrainTumorNet)
    

