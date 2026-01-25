import copy
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as fnc
import torch.optim as optim
import torchvision.transforms.v2 as F
from models_task4_5 import WraperModel
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

import wandb

project = "c3_week3_task4&5"


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def get_transform_no_aug():
    return F.Compose(
        [
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize(size=(224, 224)),
        ]
    )


def get_transform_aug():
    return Compose(
        [
            RandomResizedCrop(224, scale=(0.7, 1.0)),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            RandomRotation(10),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_transform_aug_configurable(
    use_random_crop: bool = True,
    crop_scale_min: float = 0.7,
    use_horizontal_flip: bool = True,
    horizontal_flip_p: float = 0.5,
    use_vertical_flip: bool = False,
    vertical_flip_p: float = 0.5,
    use_color_jitter: bool = True,
    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
    color_jitter_saturation: float = 0.2,
    color_jitter_hue: float = 0.05,
    use_rotation: bool = True,
    rotation_degrees: float = 10.0,
    use_grayscale: bool = False,
    grayscale_p: float = 0.1,
):
    """
    Configurable data augmentation transform for hyperparameter search.
    Designed for small datasets (~400 images).

    Args:
        use_random_crop: Whether to use RandomResizedCrop vs simple Resize
        crop_scale_min: Minimum scale for random crop (max is always 1.0)
        use_horizontal_flip: Enable horizontal flipping
        horizontal_flip_p: Probability of horizontal flip
        use_vertical_flip: Enable vertical flipping (useful for some scenes)
        vertical_flip_p: Probability of vertical flip
        use_color_jitter: Enable color augmentation
        color_jitter_*: ColorJitter parameters
        use_rotation: Enable random rotation
        rotation_degrees: Max rotation angle in degrees
        use_grayscale: Randomly convert to grayscale
        grayscale_p: Probability of grayscale conversion
    """
    transforms_list = []

    if use_random_crop:
        transforms_list.append(RandomResizedCrop(224, scale=(crop_scale_min, 1.0)))
    else:
        transforms_list.append(Resize((224, 224)))

    if use_horizontal_flip:
        transforms_list.append(RandomHorizontalFlip(p=horizontal_flip_p))
    if use_vertical_flip:
        transforms_list.append(RandomVerticalFlip(p=vertical_flip_p))

    if use_color_jitter:
        transforms_list.append(
            ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
            )
        )

    if use_grayscale:
        transforms_list.append(RandomGrayscale(p=grayscale_p))

    transforms_list.append(ToTensor())
    transforms_list.append(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return Compose(transforms_list)


def get_transform_test_normalized():
    # Test/val sense augmentació, amb normalize ImageNet
    return Compose(
        [
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def set_finetuning_strategy(model, strategy: str):
    """
    strategy: 'full', 'none', 'partial'
    """
    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True

    elif strategy == "none":
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True

    elif strategy == "partial":
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True

        # Unfreeze last convolutional block
        if len(list(model.backbone.features.children())) > 0:
            last_block = list(model.backbone.features.children())[-1]
            for param in last_block.parameters():
                param.requires_grad = True
        else:
            print("Warning: No features block found to unfreeze for partial strategy.")


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
            optimizer, mode="min", factor=0.5, patience=2
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def cross_validate(
    dataset,
    model_fn,
    optimizer_fn,
    criterion,
    scheduler_fn,
    device,
    num_epochs=3,
    num_folds=5,
    batch_size=16,
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

        model = model_fn().to(device)
        optimizer = optimizer_fn(filter(lambda p: p.requires_grad, model.parameters()))
        scheduler = scheduler_fn(optimizer, num_epochs) if scheduler_fn else None

        best_val_acc = 0.0
        early_stopping_train_loss, early_stopping_val_loss = float("inf"), float("inf")
        early_stopping_train_acc, early_stopping_val_acc = 0.0, 0.0

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


SWEEP_CONFIG = {
    "name": "task5_random_search",
    "method": "random",  # random search
    "metric": {
        "name": "cv/mean_val_acc",
        "goal": "maximize",
    },  # Optimize for learning curve
    "parameters": {
        # Architecture
        "arch_type": {"value": "modified"},
        "finetune_strategy": {
            "value": "partial"
        },  # conclusion get from task 0, partial finetuning fits better for small-medium datasets
        # Regularization - methodologies to improve learning curve
        "dropout_rate": {
            "distribution": "uniform",
            "max": 0.11,
            "min": 0.1,
        },  # Discrete for easier comparison
        "use_batchnorm": {"value": True},  # "values": [True, False] "value": False
        "weight_decay": {
            "distribution": "log_uniform_values",
            "max": 0.0006,
            "min": 0.0004,
        },  # Discrete for easier comparison
        "momentum": {
            "distribution": "uniform",
            "max": 0.2,
            "min": 0.1,
        },  # "values": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9] "value": 0.0
        "label_smoothing": {
            "distribution": "uniform",
            "max": 0.08,
            "min": 0.073,
        },  # "values": [0.0, 0.1, 0.2] "value": 0.0
        # Optimization
        "optimizer": {
            "value": "adamw"
        },  # "values": ["adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"] "value": "adam"
        "lr": {
            "distribution": "log_uniform_values",
            "max": 0.0079,
            "min": 0.0073,
        },  # {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
        "scheduler": {
            "values": ["cosine", "plateau"]
        },  # "values": ["none", "cosine", "step", "plateau"]
        # Training - fixed values
        "batch_size": {"value": 32},  # "values": [10, 20, 40, 60, 80, 100]
        "num_epochs": {"value": 60},  # "values": [10, 50, 100]
        "num_folds": {"value": 5},  # Always 5-fold CV
        # DATA AUGMENTATION PARAMETERS
        "aug_use_random_crop": {"values": [True, False]},
        "aug_crop_scale_min": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 0.9,
        },  # Min scale for crop (aggressive for small dataset)
        # Flipping
        "aug_use_horizontal_flip": {"value": True},
        "aug_horizontal_flip_p": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7,
        },
        "aug_use_vertical_flip": {"values": [True, False]},
        "aug_vertical_flip_p": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.5,
        },
        # Rotation
        "aug_use_rotation": {"values": [True, False]},
        "aug_rotation_degrees": {
            "distribution": "uniform",
            "min": 5.0,
            "max": 30.0,
        },  # Degrees
        # Color augmentations (important for small datasets)
        "aug_use_color_jitter": {"values": [True, False]},  # Usually helpful
        "aug_color_brightness": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.4,
        },
        "aug_color_contrast": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.4,
        },
        "aug_color_saturation": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.4,
        },
        "aug_color_hue": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.1,
        },  # Keep hue changes small
        "aug_use_grayscale": {"values": [True, False]},
        "aug_grayscale_p": {
            "distribution": "uniform",
            "min": 0.05,
            "max": 0.2,
        },
    },
}


def save_sweep_results(results: List[Dict], out_dir: str = "./plots"):
    """
    Save sweep results to JSON file for later analysis.
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(out_dir, f"sweep_results_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


def plot_methodology_comparison(
    results: List[Dict],
    methodology: str,
    out_dir: str = "./plots",
    figsize: tuple = (14, 8),
):
    """
    Plot grouped bar chart comparing train/test accuracy for a specific methodology
    (e.g., dropout_rate, use_batchnorm, weight_decay, label_smoothing, optimizer).
    Grouped by arch_type and finetune_strategy.
    """
    os.makedirs(out_dir, exist_ok=True)

    grouped_data = {}
    for r in results:
        arch = r.get("arch_type", "unknown")
        ft = r.get("finetune_strategy", "unknown")
        method_val = r.get(methodology, "unknown")
        key = (arch, ft, method_val)

        if key not in grouped_data:
            grouped_data[key] = {
                "train_accs": [],
                "val_accs": [],
                "train_losses": [],
                "val_losses": [],
                "std_val_accs": [],
            }

        grouped_data[key]["train_accs"].append(r.get("mean_train_acc", 0))
        grouped_data[key]["val_accs"].append(r.get("mean_val_acc", 0))
        grouped_data[key]["train_losses"].append(r.get("mean_train_loss", 0))
        grouped_data[key]["val_losses"].append(r.get("mean_val_loss", 0))
        grouped_data[key]["std_val_accs"].append(r.get("std_val_acc", 0))

    aggregated = {}
    for key, data in grouped_data.items():
        aggregated[key] = {
            "mean_train_acc": np.mean(data["train_accs"]),
            "std_train_acc": np.std(data["train_accs"])
            if len(data["train_accs"]) > 1
            else 0,
            "mean_val_acc": np.mean(data["val_accs"]),
            "std_val_acc": np.mean(data["std_val_accs"]),  # Use mean of CV std devs
            "mean_train_loss": np.mean(data["train_losses"]),
            "mean_val_loss": np.mean(data["val_losses"]),
        }

    arch_types = sorted(set(k[0] for k in aggregated.keys()))
    ft_strategies = sorted(set(k[1] for k in aggregated.keys()))
    method_values = sorted(set(k[2] for k in aggregated.keys()), key=lambda x: str(x))

    fig, axes = plt.subplots(
        len(arch_types), len(ft_strategies), figsize=figsize, squeeze=False
    )
    fig.suptitle(
        f"Learning Curve Analysis: {methodology}", fontsize=14, fontweight="bold"
    )

    bar_width = 0.35

    for i, arch in enumerate(arch_types):
        for j, ft in enumerate(ft_strategies):
            ax = axes[i, j]

            x_labels = []
            train_accs = []
            val_accs = []
            val_stds = []

            for mv in method_values:
                key = (arch, ft, mv)
                if key in aggregated:
                    x_labels.append(str(mv))
                    train_accs.append(aggregated[key]["mean_train_acc"])
                    val_accs.append(aggregated[key]["mean_val_acc"])
                    val_stds.append(aggregated[key]["std_val_acc"])

            if not x_labels:
                ax.set_visible(False)
                continue

            x = np.arange(len(x_labels))

            # Plot bars
            bars1 = ax.bar(
                x - bar_width / 2,
                train_accs,
                bar_width,
                label="Train Acc",
                color="#2ecc71",
                alpha=0.8,
            )
            bars2 = ax.bar(
                x + bar_width / 2,
                val_accs,
                bar_width,
                label="Val Acc",
                color="#3498db",
                alpha=0.8,
                yerr=val_stds,
                capsize=3,
            )

            ax.set_xlabel(methodology)
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{arch} | {ft}")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.legend(loc="lower right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)

            for bar in bars1:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.tight_layout()

    filepath = os.path.join(out_dir, f"methodology_{methodology}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {filepath}")
    return filepath


def plot_all_methodologies(results: List[Dict], out_dir: str = "./plots"):
    """
    Generate comparison plots for all methodologies.
    """
    methodologies = [
        "dropout_rate",
        "use_batchnorm",
        "weight_decay",
        "label_smoothing",
        "optimizer",
        "scheduler",
    ]

    for method in methodologies:
        try:
            plot_methodology_comparison(results, method, out_dir)
        except Exception as e:
            print(f"Warning: Could not plot {method}: {e}")

    plot_summary_comparison(results, out_dir)


def plot_summary_comparison(results: List[Dict], out_dir: str = "./plots"):
    """
    Create a summary plot showing all configurations sorted by validation accuracy.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not results:
        print("No results to plot.")
        return

    sorted_results = sorted(
        results, key=lambda x: x.get("mean_val_acc", 0), reverse=True
    )

    top_results = sorted_results[:20]

    labels = []
    train_accs = []
    val_accs = []
    val_stds = []

    for r in top_results:
        label = (
            f"{r.get('arch_type', '?')[:3]}_{r.get('finetune_strategy', '?')[:4]}_"
            f"do{r.get('dropout_rate', 0)}_bn{int(r.get('use_batchnorm', False))}_"
            f"wd{r.get('weight_decay', 0)}"
        )
        labels.append(label)
        train_accs.append(r.get("mean_train_acc", 0))
        val_accs.append(r.get("mean_val_acc", 0))
        val_stds.append(r.get("std_val_acc", 0))

    fig, ax = plt.subplots(figsize=(14, 10))

    y = np.arange(len(labels))
    bar_height = 0.35

    _ = ax.barh(
        y - bar_height / 2,
        train_accs,
        bar_height,
        label="Train Acc",
        color="#2ecc71",
        alpha=0.8,
    )
    _ = ax.barh(
        y + bar_height / 2,
        val_accs,
        bar_height,
        label="Val Acc",
        color="#3498db",
        alpha=0.8,
        xerr=val_stds,
        capsize=3,
    )

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Configuration")
    ax.set_title("Top 20 Configurations by Validation Accuracy")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    filepath = os.path.join(out_dir, "summary_top_configs.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved: {filepath}")


ALL_SWEEP_RESULTS = []


def model_fn(config, num_classes: int):
    model = WraperModel(
        num_classes=num_classes,
        arch_type=config.arch_type,
        dropout_rate=config.dropout_rate,
        use_batchnorm=config.use_batchnorm,
    )
    set_finetuning_strategy(model, config.finetune_strategy)
    return model


def scheduler_fn(optimizer, config, num_epochs):
    return get_scheduler(optimizer, config.scheduler, num_epochs)


def get_criterion(loss_name: str, label_smoothing: float = 0.0):
    """
    Create loss function.
    """
    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


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


def wandb_sweep_train():
    """
    Training function called by wandb sweep agent.
    Each call runs one hyperparameter configuration with cross-validation.
    """
    global ALL_SWEEP_RESULTS

    with wandb.init() as run:
        config = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data path
        base_root = "./data/2425"
        train_path = os.path.join(base_root, "MIT_small_train_1", "train")

        # Create dataset with configurable augmentation from sweep config
        train_transform = get_transform_aug_configurable(
            use_random_crop=config.aug_use_random_crop,
            crop_scale_min=config.aug_crop_scale_min,
            use_horizontal_flip=config.aug_use_horizontal_flip,
            horizontal_flip_p=config.aug_horizontal_flip_p,
            use_vertical_flip=config.aug_use_vertical_flip,
            vertical_flip_p=config.aug_vertical_flip_p,
            use_color_jitter=config.aug_use_color_jitter,
            color_jitter_brightness=config.aug_color_brightness,
            color_jitter_contrast=config.aug_color_contrast,
            color_jitter_saturation=config.aug_color_saturation,
            color_jitter_hue=config.aug_color_hue,
            use_rotation=config.aug_use_rotation,
            rotation_degrees=config.aug_rotation_degrees,
            use_grayscale=config.aug_use_grayscale,
            grayscale_p=config.aug_grayscale_p,
        )
        dataset = ImageFolder(train_path, transform=train_transform)
        num_classes = len(dataset.classes)

        print(f"SWEEP RUN: {run.name}")
        print(f"Config: {dict(config)}")

        # Criterion with label smoothing
        criterion = get_criterion("ce", label_smoothing=config.label_smoothing)

        # Run cross-validation (always 5 folds)
        results = cross_validate(
            dataset=dataset,
            model_fn=lambda: model_fn(config, num_classes),
            optimizer_fn=lambda params: create_optimizer(params, config),
            criterion=criterion,
            scheduler_fn=lambda optimizer, num_epochs: scheduler_fn(
                optimizer, config, num_epochs
            ),
            device=device,
            num_epochs=config.num_epochs,
            num_folds=5,
            batch_size=config.batch_size,
            num_workers=8,
            use_wandb=True,
        )

        run_result = {
            "run_name": run.name,
            "run_id": run.id,
            # Config parameters
            "arch_type": config.arch_type,
            "finetune_strategy": config.finetune_strategy,
            "dropout_rate": config.dropout_rate,
            "use_batchnorm": config.use_batchnorm,
            "weight_decay": config.weight_decay,
            "label_smoothing": config.label_smoothing,
            "optimizer": config.optimizer,
            "lr": config.lr,
            "scheduler": config.scheduler,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            # Data augmentation parameters
            "aug_use_random_crop": config.aug_use_random_crop,
            "aug_crop_scale_min": config.aug_crop_scale_min,
            "aug_use_horizontal_flip": config.aug_use_horizontal_flip,
            "aug_horizontal_flip_p": config.aug_horizontal_flip_p,
            "aug_use_vertical_flip": config.aug_use_vertical_flip,
            "aug_vertical_flip_p": config.aug_vertical_flip_p,
            "aug_use_rotation": config.aug_use_rotation,
            "aug_rotation_degrees": config.aug_rotation_degrees,
            "aug_use_color_jitter": config.aug_use_color_jitter,
            "aug_color_brightness": config.aug_color_brightness,
            "aug_color_contrast": config.aug_color_contrast,
            "aug_color_saturation": config.aug_color_saturation,
            "aug_color_hue": config.aug_color_hue,
            "aug_use_grayscale": config.aug_use_grayscale,
            "aug_grayscale_p": config.aug_grayscale_p,
            # Results
            "mean_train_loss": results["mean_train_loss"],
            "std_train_loss": results["std_train_loss"],
            "mean_train_acc": results["mean_train_acc"],
            "std_train_acc": results["std_train_acc"],
            "mean_val_loss": results["mean_val_loss"],
            "std_val_loss": results["std_val_loss"],
            "mean_val_acc": results["mean_val_acc"],
            "std_val_acc": results["std_val_acc"],
        }
        ALL_SWEEP_RESULTS.append(run_result)

        print("CROSS-VALIDATION SUMMARY:")
        print(
            f"  Train Loss: {results['mean_train_loss']:.4f} +/- {results['std_train_loss']:.4f}"
        )
        print(
            f"  Train Acc:  {results['mean_train_acc']:.4f} +/- {results['std_train_acc']:.4f}"
        )
        print(
            f"  Val Loss:   {results['mean_val_loss']:.4f} +/- {results['std_val_loss']:.4f}"
        )
        print(
            f"  Val Acc:    {results['mean_val_acc']:.4f} +/- {results['std_val_acc']:.4f}"
        )


def run_wandb_sweep(n_runs: int = 20, out_dir: str = "./plots"):
    """
    Initialize and run a wandb sweep.

    Args:
        n_runs: Number of random configurations to try
        out_dir: Output directory for plots and results
    """
    global ALL_SWEEP_RESULTS
    ALL_SWEEP_RESULTS = []

    os.makedirs(out_dir, exist_ok=True)

    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project=project,
    )

    print(f"\nSweep ID: {sweep_id}")
    print(f"Running {n_runs} configurations...")
    print("Metric: maximize cv/mean_val_acc")
    print("Always using 5-fold cross-validation")
    print(
        f"View at: https://wandb.ai/{wandb.api.default_entity}/{project}/sweeps/{sweep_id}"
    )

    wandb.agent(sweep_id, function=wandb_sweep_train, count=n_runs)

    if ALL_SWEEP_RESULTS:
        print("GENERATING METHODOLOGY COMPARISON PLOTS")

        save_sweep_results(ALL_SWEEP_RESULTS, out_dir)

        plot_all_methodologies(ALL_SWEEP_RESULTS, out_dir)

        print(f"\nAll plots saved to: {out_dir}")


def plot_misclassified_with_gradcam(
    model,
    dataset,
    device,
    class_names,
    num_examples: int = 3,
    save_path: str = "./plots/misclassified_gradcam.png",
):
    """
    Find misclassified images and plot them with GradCAM visualization.

    Args:
        model: Trained model
        dataset: Test dataset (ImageFolder)
        device: torch device
        class_names: List of class names
        num_examples: Number of misclassified examples to show
        save_path: Path to save the plot
    """
    model.eval()

    # ImageNet normalization values (used in transforms)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    target_layers = [model.backbone.features[-1]]

    misclassified = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)

            output = model(img_tensor)
            pred = output.argmax(dim=1).item()

            if pred != label:
                misclassified.append(
                    {
                        "idx": idx,
                        "image": img,
                        "true_label": label,
                        "pred_label": pred,
                    }
                )

            if len(misclassified) >= num_examples:
                break

    if len(misclassified) == 0:
        print("No misclassified images found!")
        return

    fig, axes = plt.subplots(
        2, len(misclassified), figsize=(5 * len(misclassified), 10)
    )

    if len(misclassified) == 1:
        axes = axes.reshape(2, 1)

    cam = GradCAMPlusPlus(model=model.backbone, target_layers=target_layers)

    for col, item in enumerate(misclassified):
        img_tensor = item["image"]
        true_label = item["true_label"]
        pred_label = item["pred_label"]

        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        input_tensor = img_tensor.unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(pred_label)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(
            img_np.astype(np.float32), grayscale_cam, use_rgb=True
        )

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(
            f"GT: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            fontsize=10,
            color="red",
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(cam_image)
        axes[1, col].set_title(
            f"GradCAM (pred: {class_names[pred_label]})", fontsize=10
        )
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("GradCAM", fontsize=12, fontweight="bold")

    plt.suptitle(
        "Misclassified Images with GradCAM Visualization",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Misclassified GradCAM plot saved to: {save_path}")

    # Clean up
    del cam


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
    import argparse

    parser = argparse.ArgumentParser(description="Training with optional W&B sweep")
    parser.add_argument(
        "--mode",
        type=str,
        default="manual",
        choices=["manual", "sweep"],
        help="Run mode: 'manual' for regular training, 'sweep' for wandb hyperparameter search",
    )
    parser.add_argument(
        "--sweep_runs",
        type=int,
        default=20,
        help="Number of sweep runs (only used with --mode sweep)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./plots",
        help="Output directory for plots and results",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.mode == "sweep":
        print("WANDB SWEEP - Random Search Hyperparameter Optimization")
        run_wandb_sweep(n_runs=args.sweep_runs, out_dir=args.out_dir)

    else:
        # MANUAL MODE
        config = argparse.Namespace(
            # Architecture
            arch_type="modified",
            finetune_strategy="partial",
            batch_size=32,
            num_folds=5,
            num_epochs=60,
            lr=0.007500331164768306,
            scheduler="plateau",
            use_batchnorm=True,
            optimizer="adamw",
            label_smoothing=0.07420576372699497,
            weight_decay=0.0004469627477065592,
            momentum=0.10830362128812387,
            dropout_rate=0.10500895322471474,
        )

        # Output folder
        out_dir = "./plots"
        os.makedirs(out_dir, exist_ok=True)

        base_root = "./data/2425"  # ajusta si cal
        train_path = os.path.join(base_root, "MIT_small_train_1", "train")
        test_path = os.path.join(base_root, "MIT_small_train_1", "test")
        train_transform = get_transform_aug()
        dataset_train = ImageFolder(train_path, transform=train_transform)
        num_classes = len(dataset_train.classes)

        best_test_loss = float("inf")
        best_test_acc = 0.0
        model_best = None

        dataset_test = ImageFolder(test_path, transform=get_transform_test_normalized())
        train_loader = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        class_names = dataset_test.classes

        criterion = get_criterion("ce", label_smoothing=config.label_smoothing)
        model = model_fn(config, num_classes).to(device)
        optimizer = create_optimizer(
            filter(lambda p: p.requires_grad, model.parameters()), config
        )
        scheduler = scheduler_fn(optimizer, config, num_epochs=config.num_epochs)

        print("Config:", config)
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")

        for epoch in range(config.num_epochs):
            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = test(model, test_loader, criterion, device)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_acc = test_acc
                model_best = copy.deepcopy(model.state_dict())

            print(
                f"Epoch {epoch + 1}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f} | LR: {current_lr:.6f}"
            )

        print(
            "Best Test Loss: {:.4f}, Best Test Acc: {:.4f}".format(
                best_test_loss, best_test_acc
            )
        )

        torch.save(model_best, "./teacher_model.pth")
        # Load best model for evaluation
        model.load_state_dict(model_best)
        print(summary(model, (3, 224, 224)))

        y_true, y_score = evaluate_and_export_scores(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            normalize_cm=True,
            cm_save_path=os.path.join(out_dir, "confusion_matrix.png"),
        )

        # Plot misclassified images with GradCAM
        plot_misclassified_with_gradcam(
            model=model,
            dataset=dataset_test,
            device=device,
            class_names=class_names,
            num_examples=4,
            save_path=os.path.join(out_dir, "misclassified_gradcam.png"),
        )
