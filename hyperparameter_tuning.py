import os

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cnn_model import ButterflyNet, BrainTumorNet
from utils import ButterflyDataset, BrainTumorDataset


# Objective function for Optuna
def objective(trial):
    # ---- Hyperparameter search space ----
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 1, 2)

    # ---- Data loaders ----
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

    # ---- Model, loss, optimizer ----
    USE_GPU = True
    # This ensures PyTorch does not see any CUDA devices, forcing CPU usage.
    # Only needed for when GPU is so old that CUDA is outdated
    if not USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if MODEL_NAME == "Butterfly":
        model = ButterflyNet(input_size=dataset.input_size, dropout=dropout).to(device)
    elif MODEL_NAME == "BrainTumor":
        model = BrainTumorNet(input_size=dataset.input_size, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ---- Training + validation ----
    for epoch in range(epochs):

        # ----- Training -----
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # ----- Validation -----
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)

        accuracy = correct / total

        # Report the intermediate result for pruning
        trial.report(accuracy, epoch)

        # If trial pruned, stop early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

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

# Running the optimisation
if __name__ == "__main__":
#     # Load dataset
#     dataset = ButterflyDataset()

#     # Create train-val split
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

#     # Global variable to track which model is being tuned
#     MODEL_NAME = "Butterfly"

#     start_opt()

    # Load dataset
    dataset = BrainTumorDataset()

    # Create train-val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Global variable to track which model is being tuned
    MODEL_NAME = "BrainTumor"

    start_opt()

    

    

