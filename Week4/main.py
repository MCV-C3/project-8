import argparse
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as fnc
import torch.optim as optim
import tqdm
from models import ShuffleCNNvAttn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    Normalize,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToDtype,
    ToImage,
)
from torchviz import make_dot

import wandb


# Train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def cross_validate(
    dataset,
    model,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=50,
    num_folds=5,
    batch_size=32,
    num_workers=8,
    use_wandb=False,
):
    targets = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_train_losses = []
    fold_train_accs = []
    fold_val_losses = []
    fold_val_accs = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets)
    ):
        print(f"\nFold {fold + 1}/{num_folds}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        best_val_acc = 0.0
        early_stopping_train_loss, early_stopping_val_loss = float("inf"), float("inf")
        early_stopping_train_acc, early_stopping_val_acc = 0.0, 0.0

        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = test(model, val_loader, criterion, device)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            best_val_acc = max(best_val_acc, val_acc)
            current_lr = optimizer.param_groups[0]["lr"]

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(val_loss)
            test_accuracies.append(val_acc)

            if val_loss < early_stopping_val_loss:
                early_stopping_train_loss = train_loss
                early_stopping_val_loss = val_loss
                early_stopping_train_acc = train_acc
                early_stopping_val_acc = val_acc

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}"
            )

            if use_wandb:
                wandb.log(
                    {
                        f"fold_{fold + 1}/train_loss": train_loss,
                        f"fold_{fold + 1}/train_acc": train_acc,
                        f"fold_{fold + 1}/val_loss": val_loss,
                        f"fold_{fold + 1}/val_acc": val_acc,
                        f"fold_{fold + 1}/lr": current_lr,
                        "epoch": epoch + 1,
                    }
                )

        fold_train_losses.append(early_stopping_train_loss)
        fold_train_accs.append(early_stopping_train_acc)
        fold_val_losses.append(early_stopping_val_loss)
        fold_val_accs.append(early_stopping_val_acc)

    mean_train_loss = np.mean(fold_train_losses)
    std_train_loss = np.std(fold_train_losses)
    mean_train_acc = np.mean(fold_train_accs)
    std_train_acc = np.std(fold_train_accs)
    mean_val_loss = np.mean(fold_val_losses)
    std_val_loss = np.std(fold_val_losses)
    mean_val_acc = np.mean(fold_val_accs)
    std_val_acc = np.std(fold_val_accs)

    print("\nCross-validation Summary")
    print(f"Train Loss: {mean_train_loss:.4f} +/- {std_train_loss:.4f}")
    print(f"Train Acc:  {mean_train_acc:.4f} +/- {std_train_acc:.4f}")
    print(f"Val Loss:   {mean_val_loss:.4f} +/- {std_val_loss:.4f}")
    print(f"Val Acc:    {mean_val_acc:.4f} +/- {std_val_acc:.4f}")

    if use_wandb:
        wandb.log(
            {
                # Train metrics
                "cv/mean_train_loss": mean_train_loss,
                "cv/std_train_loss": std_train_loss,
                "cv/mean_train_acc": mean_train_acc,
                "cv/std_train_acc": std_train_acc,
                # Validation metrics
                "cv/mean_val_loss": mean_val_loss,
                "cv/std_val_loss": std_val_loss,
                "cv/mean_val_acc": mean_val_acc,
                "cv/std_val_acc": std_val_acc,
                # Additional
                "cv/best_fold_val_acc": max(fold_val_accs),
                "cv/worst_fold_val_acc": min(fold_val_accs),
                "cv/train_val_gap": mean_train_acc - mean_val_acc,
            }
        )

        fold_table = wandb.Table(
            columns=["fold", "train_loss", "train_acc", "val_loss", "val_acc"],
            data=[
                [i + 1, tl, ta, vl, va]
                for i, (tl, ta, vl, va) in enumerate(
                    zip(
                        fold_train_losses,
                        fold_train_accs,
                        fold_val_losses,
                        fold_val_accs,
                    )
                )
            ],
        )
        wandb.log({"cv/fold_results": fold_table})

    return {
        "fold_train_losses": fold_train_losses,
        "fold_train_accs": fold_train_accs,
        "fold_val_losses": fold_val_losses,
        "fold_val_accs": fold_val_accs,
        "mean_train_loss": mean_train_loss,
        "std_train_loss": std_train_loss,
        "mean_train_acc": mean_train_acc,
        "std_train_acc": std_train_acc,
        "mean_val_loss": mean_val_loss,
        "std_val_loss": std_val_loss,
        "mean_val_acc": mean_val_acc,
        "std_val_acc": std_val_acc,
    }


def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str):
    """
    Plots and saves metrics for training and testing.

    Args:
        train_metrics (Dict): Dictionary containing training metrics.
        test_metrics (Dict): Dictionary containing testing metrics.
        metric_name (str): The name of the metric to plot (e.g., "loss", "accuracy").

    Saves:
        - loss.png for loss plots
        - metrics.png for other metrics plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f"Train {metric_name.capitalize()}")
    plt.plot(test_metrics[metric_name], label=f"Test {metric_name.capitalize()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot with the appropriate name
    filename = "loss.png" if metric_name.lower() == "loss" else "metrics.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory


# Data augmentation example
def get_data_transforms(img_size: int = 224) -> Compose:
    """
    Returns a Compose object with data augmentation transformations.
    """
    return Compose(
        [
            Resize((img_size, img_size)),
            RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            RandomHorizontalFlip(p=0.2),
            RandomRotation(15),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            RandomGrayscale(p=0.1),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def plot_computational_graph(
    model: torch.nn.Module, input_size: tuple, filename: str = "computational_graph"
):
    """
    Generates and saves a plot of the computational graph of the model.

    Args:
        model (torch.nn.Module): The PyTorch model to visualize.
        input_size (tuple): The size of the dummy input tensor (e.g., (batch_size, input_dim)).
        filename (str): Name of the file to save the graph image.
    """
    model.eval()  # Set the model to evaluation mode

    # Generate a dummy input based on the specified input size
    dummy_input = torch.randn(*input_size)

    # Create a graph from the model
    _ = make_dot(
        model(dummy_input), params=dict(model.named_parameters()), show_attrs=True
    ).render(filename, format="png")

    print(f"Computational graph saved as {filename}")


def create_optimizer(params, config):
    """Helper to create optimizer from wandb config."""
    if config.optimizer == "adam":
        return optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        return optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "rmsprop":
        return optim.RMSprop(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adagrad":
        return optim.Adagrad(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adadelta":
        return optim.Adadelta(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adamax":
        return optim.Adamax(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "nadam":
        return optim.NAdam(params, lr=config.lr, weight_decay=config.weight_decay)
    else:
        return optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)


def get_criterion(loss_name: str, label_smoothing: float = 0.0):
    """
    Create loss function.
    """
    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_scheduler(optimizer, scheduler_name: str, num_epochs: int, **kwargs):
    """
    Create learning rate scheduler based on name.
    """
    if scheduler_name == "none":
        return None
    elif scheduler_name == "step":
        step_size = kwargs.get("step_size", max(1, num_epochs // 3))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.75, patience=5
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def evaluate_and_export_scores(
    model,
    dataloader,
    device,
    class_names,
    normalize_cm=True,
    cm_save_path="./plots/confusion_matrix.png",
):
    """
    Computes confusion matrix and prints raw scores for ROC curves.

    Prints two lines:
      1) Ground-truth labels
      2) Predicted probabilities (per class)

    These can be copied directly for ROC plotting.
    """

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = fnc.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)

    if normalize_cm:
        cm = cm.astype(np.float32)
        cm /= cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize_cm else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize_cm else ""))
    plt.tight_layout()
    plt.savefig(cm_save_path)
    plt.close()

    print(f"Confusion matrix saved to {cm_save_path}")

    return np.array(all_labels), np.array(all_probs)


if __name__ == "__main__":
    torch.manual_seed(42)

    config = argparse.Namespace(
        # Architecture
        data_dir="./data/2425/MIT_small_train_1",
        img_size=224,
        batch_size=64,
        num_folds=5,
        num_epochs=500,
        lr=0.01,  # 0.0075
        early_stopping=50,
        scheduler="plateau",  # plateau
        use_batchnorm=True,
        optimizer="adamw",  # adamw
        label_smoothing=0.1,  # 0.074
        weight_decay=0.0,  # 0.00045
        momentum=0.0,  # 0.108
        dropout_rate=0.1,
        use_attn=False,
    )

    train_transformation = get_data_transforms(img_size=config.img_size)
    test_transformation = Compose(
        [
            Resize((config.img_size, config.img_size)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_train = ImageFolder(config.data_dir + "/train", transform=train_transformation)
    data_test = ImageFolder(config.data_dir + "/test", transform=test_transformation)

    train_loader = DataLoader(
        data_train,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
    )
    test_loader = DataLoader(
        data_test,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
    )

    num_classes = len(data_train.classes)
    class_names = data_test.classes

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ShuffleCNNvAttn(
        num_classes=num_classes,
        dropout_rate=config.dropout_rate,
        use_attn=config.use_attn,
    ).to(device)

    criterion = get_criterion("ce", label_smoothing=config.label_smoothing)
    optimizer = create_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()), config
    )
    scheduler = get_scheduler(optimizer, config.scheduler, num_epochs=config.num_epochs)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_val_epoch = -1
    best_model_state_dict = None
    early_stopping_counter = 0

    for epoch in tqdm.tqdm(range(config.num_epochs), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if test_loss < best_val_loss:
            best_val_epoch = epoch
            best_val_loss = test_loss
            best_val_accuracy = test_accuracy
            best_model_state_dict = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= config.early_stopping:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} - [{early_stopping_counter}] "
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f} | LR: {current_lr:.6f}"
        )

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

    torch.save(best_model_state_dict, "./saved_model.pt")
    model.load_state_dict(best_model_state_dict)

    print(summary(model, (C, H, W)))
    total_params = sum(p.numel() for p in model.parameters())
    ratio = best_val_accuracy / (
        total_params / 100000
    )  # ratio: accuracy/ (number of parameters/ 100K)
    print(f"Best Test Accuracy: {best_val_accuracy:.4f}")
    print(f"Ratio (Accuracy / Number of Parameters per 100K): {ratio:.4f}")
    print("----------------------------------------------------------------")

    print("Configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    print("Classes:", data_train.classes)

    # Plot results
    plot_metrics(
        {"loss": train_losses, "accuracy": train_accuracies},
        {"loss": test_losses, "accuracy": test_accuracies},
        "loss",
    )
    plot_metrics(
        {"loss": train_losses, "accuracy": train_accuracies},
        {"loss": test_losses, "accuracy": test_accuracies},
        "accuracy",
    )

    y_true, y_score = evaluate_and_export_scores(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        normalize_cm=True,
    )
