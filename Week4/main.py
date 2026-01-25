import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as fnc
import torch.nn.utils.prune as prune
import torch.optim as optim
import tqdm
from models import (
    ShuffleCNNvAttn,
    SimpleCNNv1,
    SimpleCNNv2,
    SimpleCNNv3,
    SimpleCNNvAttn,
    TeacherModel,
)
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
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
        if "Simple" in model.__class__.__name__:
            outputs = model(inputs)
        else:
            outputs, _ = model(inputs)
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


def train_knowledge_distillation(
    teacher,
    student,
    dataloader,
    criterion,
    optimizer,
    config,
    device,
):
    teacher.eval()
    student.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs, _ = teacher(inputs)

        student_outputs, _ = student(inputs)

        soft_teacher_outputs = fnc.softmax(teacher_outputs / config.temperature, dim=-1)
        soft_student_outputs = fnc.log_softmax(
            student_outputs / config.temperature, dim=-1
        )

        soft_targets_loss = (
            torch.sum(
                soft_teacher_outputs
                * (soft_teacher_outputs.log() - soft_student_outputs)
            )
            / soft_student_outputs.size()[0]
            * (config.temperature**2)
        )
        label_loss = criterion(student_outputs, labels)

        # Weighted loss
        loss = (
            config.distillation_weight * soft_targets_loss
            + (1 - config.distillation_weight) * label_loss
        )
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = student_outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_mse_loss(
    teacher,
    student,
    dataloader,
    criterion,
    optimizer,
    config,
    device,
):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    teacher.eval()
    student.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            _, teacher_feature_map = teacher(inputs)

        outputs, regressor_feature_map = student(inputs)

        hidden_loss = mse_loss(regressor_feature_map, teacher_feature_map)
        label_loss = ce_loss(outputs, labels)

        loss = (
            config.feature_map_weight * hidden_loss
            + (1 - config.feature_map_weight) * label_loss
        )
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

            # Forward pass
            if "Simple" in model.__class__.__name__:
                outputs = model(inputs)
            else:
                outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normal_train(
    model,
    teacher_model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    config,
    device,
    train_losses,
    train_accuracies,
    test_losses,
    test_accuracies,
):
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_val_epoch = -1
    best_model_state_dict = None
    early_stopping_counter = 0

    for epoch in tqdm.tqdm(range(config.num_epochs), desc="TRAINING THE MODEL"):
        if config.knowledge_distillation:
            train_loss, train_accuracy = train_knowledge_distillation(
                teacher_model,
                model,
                train_loader,
                criterion,
                optimizer,
                config,
                device,
            )
        else:
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
    torch.save(best_model_state_dict, "./saved_model.pth")
    model.load_state_dict(best_model_state_dict)

    print(summary(model, (C, H, W)))
    total_params = count_parameters(model)
    print(
        f"Total Trainable Parameters via {count_parameters.__name__}(): {total_params}"
    )
    ratio = best_val_accuracy / (
        total_params / 100000
    )  # ratio: accuracy/ (number of parameters/ 100K)
    print(f"Best Test Accuracy: {best_val_accuracy:.4f}")
    print(f"Best Test Loss: {best_val_loss:.4f} at epoch {best_val_epoch + 1}")
    print(f"Ratio (Accuracy / Number of Parameters per 100K): {ratio:.4f}")
    print("----------------------------------------------------------------")

    print("Configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    print("Classes:", data_train.classes)


def cross_validate(
    dataset_root_path,
    teacher_model,
    config,
    device,
    use_wandb=False,
):
    fold_train_losses = []
    fold_train_accs = []
    fold_val_losses = []
    fold_val_accs = []

    for fold in range(config.num_folds):
        print(f"\nFold {fold + 1}/{config.num_folds}")

        data_train = ImageFolder(
            dataset_root_path + f"MIT_small_train_{fold + 1}/train",
            transform=train_transformation,
        )
        data_test = ImageFolder(
            dataset_root_path + f"MIT_small_train_{fold + 1}/test",
            transform=test_transformation,
        )

        train_loader = DataLoader(
            data_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        val_loader = DataLoader(
            data_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        torch.manual_seed(42)
        if config.use_model == "v1":
            model = SimpleCNNv1(
                num_classes=num_classes,
            ).to(device)
        elif config.use_model == "v2":
            model = SimpleCNNv2(
                num_classes=num_classes,
            ).to(device)
        elif config.use_model == "v3":
            model = SimpleCNNv3(
                num_classes=num_classes,
                image_input_size=config.img_size,
            ).to(device)
        elif config.use_model == "v_attn":
            model = SimpleCNNvAttn(
                num_classes=num_classes,
                dropout_rate=config.dropout_rate,
                use_attn=config.use_attn,
                knowledge_distillation=config.knowledge_distillation,
            ).to(device)
        elif config.use_model == "shuffle_v_attn":
            model = ShuffleCNNvAttn(
                num_classes=num_classes,
                dropout_rate=config.dropout_rate,
                use_attn=config.use_attn,
                knowledge_distillation=config.knowledge_distillation,
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {config.use_model}")

        criterion = get_criterion("ce", label_smoothing=config.label_smoothing)
        optimizer = create_optimizer(
            filter(lambda p: p.requires_grad, model.parameters()), config
        )
        scheduler = get_scheduler(
            optimizer, config.scheduler, num_epochs=config.num_epochs
        )

        best_val_acc = 0.0
        early_stopping_train_loss, early_stopping_val_loss = float("inf"), float("inf")
        early_stopping_train_acc, early_stopping_val_acc = 0.0, 0.0
        early_stopping_counter = 0

        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        for epoch in range(config.num_epochs):
            if config.knowledge_distillation:
                train_loss, train_acc = train_mse_loss(
                    teacher_model,
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    config,
                    device,
                )
            else:
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
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= config.early_stopping:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            print(
                f"Epoch {epoch + 1}/{config.num_epochs} - [{early_stopping_counter}] | "
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

    print(summary(model, (C, H, W)))

    print("\nCross-validation Summary")
    print(f"Train Loss: {mean_train_loss:.4f} +/- {std_train_loss:.4f}")
    print(f"Train Acc:  {mean_train_acc:.4f} +/- {std_train_acc:.4f}")
    print(f"Val Loss:   {mean_val_loss:.4f} +/- {std_val_loss:.4f}")
    print(f"Val Acc:    {mean_val_acc:.4f} +/- {std_val_acc:.4f}")

    total_params = count_parameters(model)
    print(
        f"Total Trainable Parameters via {count_parameters.__name__}(): {total_params}"
    )
    ratios = [
        val_acc / (total_params / 100000) for val_acc in fold_val_accs
    ]  # ratio: accuracy/ (number of parameters/ 100K)
    print(f"Best Test Accuracy: {max(fold_val_accs):.4f}")
    print(
        f"Ratio (Accuracy / Number of Parameters per 100K): {np.mean(ratios):.4f} +/- {np.std(ratios):.4f}"
    )

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
    if metric_name.lower() == "accuracy":
        plt.ylim(0, 1)
    else:
        plt.ylim(0, 3)
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
            RandomResizedCrop(img_size, scale=(0.5, 1.0)),  # scale=(0.6, 1.0)
            RandomHorizontalFlip(p=0.2),  # p=0.2
            RandomRotation(15),  # 15
            # ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            RandomGrayscale(p=0.1),  # p=0.1
            # RandomErasing(p=0.25, scale=(0.02, 0.2)),  # Cutout-like augmentation
            # GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
            optimizer, mode="min", factor=0.85, patience=10, min_lr=0.00025
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

            if "Simple" in model.__class__.__name__:
                outputs = model(inputs)
            else:
                outputs, _ = model(inputs)
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


def get_gradcam_target_layers(model) -> List[Tuple[str, nn.Module]]:
    """
    Returns target layers for GradCAM visualization based on model type.
    Returns a dict with layer names and their corresponding modules.
    """
    model_name = model.__class__.__name__
    layers = []

    if model_name == "SimpleCNNv1":
        layers.append(("First Block Second Conv", [model.b1[5]]))
        layers.append(("Middle Block Second Conv", [model.b2[5]]))
        layers.append(("Last Block Second Conv", [model.b3[5]]))

    elif model_name == "SimpleCNNv2":
        layers.append(("First Block Conv", [model.b1[2]]))
        layers.append(("Last Block Conv", [model.b2[2]]))

    elif model_name == "SimpleCNNv3":
        layers.append(("First Block Conv", [model.b1[2]]))
        layers.append(("Skip1", [model.skip1_relu]))
        layers.append(("Last Block Conv", [model.b2[2]]))
        layers.append(("Skip2", [model.skip2_relu]))

    elif model_name == "SimpleCNNvAttn":
        layers.append(("First Block Conv", [model.b1[2]]))
        layers.append(("Skip1", [model.skip1_relu]))
        if model.attn1 is not None:
            layers.append(("Attention1", [model.attn1]))
        layers.append(("Last Block Conv", [model.b2[2]]))
        layers.append(("Skip2", [model.skip2_relu]))
        if model.attn2 is not None:
            layers.append(("Attention2", [model.attn2]))

    elif model_name == "ShuffleCNNvAttn":
        layers.append(("First Block Conv", [model.conv1[2]]))
        layers.append(("Inverted Residual Block 1", [model.stage2]))
        if model.attn1 is not None:
            layers.append(("Attention1", [model.attn1]))
        layers.append(("Inverted Residual Block 2", [model.stage3]))
        if model.attn2 is not None:
            layers.append(("Attention2", [model.attn2]))
        layers.append(("Last Block Conv", [model.conv4[2]]))
    return layers


def find_correct_and_wrong_predictions(
    model, dataloader, device, class_names
) -> Tuple[Tuple, Tuple]:
    """
    Find one correctly predicted and one incorrectly predicted sample.
    Returns: ((correct_img, correct_label, correct_pred), (wrong_img, wrong_label, wrong_pred))
    """
    model.eval()
    correct_sample = None
    wrong_sample = None

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if "Simple" in model.__class__.__name__:
                outputs = model(inputs)
            else:
                outputs, _ = model(inputs)

            _, preds = outputs.max(1)

            for i in range(inputs.size(0)):
                if correct_sample is None and preds[i] == labels[i]:
                    correct_sample = (
                        inputs[i].unsqueeze(0),
                        class_names[labels[i].item()],
                        class_names[preds[i].item()],
                    )
                if wrong_sample is None and preds[i] != labels[i]:
                    wrong_sample = (
                        inputs[i].unsqueeze(0),
                        class_names[labels[i].item()],
                        class_names[preds[i].item()],
                    )

                if correct_sample is not None and wrong_sample is not None:
                    return correct_sample, wrong_sample

    # Fallback if no wrong prediction found
    if wrong_sample is None:
        wrong_sample = correct_sample
    if correct_sample is None:
        correct_sample = wrong_sample

    return correct_sample, wrong_sample


def find_correct_and_wrong_for_each_class(
    model, dataloader, device, class_names
) -> Dict[str, Tuple]:
    """
    Find one correctly predicted sample for each class.
    Returns a dict with class names as keys and (img, label, pred) tuples as values.
    """
    model.eval()
    correct_samples = {class_name: None for class_name in class_names}
    wrong_samples = {class_name: None for class_name in class_names}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if "Simple" in model.__class__.__name__:
                outputs = model(inputs)
            else:
                outputs, _ = model(inputs)

            _, preds = outputs.max(1)

            for i in range(inputs.size(0)):
                class_name = class_names[labels[i].item()]
                if correct_samples[class_name] is None and preds[i] == labels[i]:
                    correct_samples[class_name] = (
                        inputs[i].unsqueeze(0),
                        class_name,
                        class_name,
                    )
                if wrong_samples[class_name] is None and preds[i] != labels[i]:
                    wrong_samples[class_name] = (
                        inputs[i].unsqueeze(0),
                        class_name,
                        class_names[preds[i].item()],
                    )

            if all(sample is not None for sample in correct_samples.values()):
                if all(sample is not None for sample in wrong_samples.values()):
                    break

    return correct_samples, wrong_samples


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor from ImageNet normalization to [0, 1] range.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    img = img.cpu().numpy().transpose(1, 2, 0)
    return img


class GradCAMModelWrapper(nn.Module):
    """Wrapper to make models with tuple outputs compatible with GradCAM."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output


def generate_gradcam_visualization(
    model,
    dataloader,
    device,
    class_names,
    save_path: str = "./plots/gradcam_comparison.png",
):
    """
    Generate GradCAM visualization comparing correct and wrong predictions.
    Shows first conv, last conv, and additional layers (skip/attention) if present.
    """
    model.eval()

    # Find samples
    correct_sample, wrong_sample = find_correct_and_wrong_predictions(
        model, dataloader, device, class_names
    )

    if correct_sample is None or wrong_sample is None:
        print("Could not find suitable samples for GradCAM visualization.")
        return

    correct_img, correct_gt, correct_pred = correct_sample
    wrong_img, wrong_gt, wrong_pred = wrong_sample

    # Get target layers
    target_layers_list = get_gradcam_target_layers(model)
    num_layers = len(target_layers_list)

    # Create figure: (1 + num_layers) rows x 2 columns
    num_rows = 1 + num_layers
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))

    # Denormalize images for visualization
    correct_img_np = denormalize_image(correct_img.squeeze(0))
    wrong_img_np = denormalize_image(wrong_img.squeeze(0))

    # Row 0: Original images
    axes[0, 0].imshow(correct_img_np)
    axes[0, 0].set_title(
        f"Correct\nGT: {correct_gt} | Pred: {correct_pred}", fontsize=10
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(wrong_img_np)
    axes[0, 1].set_title(f"Wrong\nGT: {wrong_gt} | Pred: {wrong_pred}", fontsize=10)
    axes[0, 1].axis("off")

    # Wrap model if it returns tuple
    wrapped_model = (
        GradCAMModelWrapper(model)
        if not model.__class__.__name__.startswith("Simple")
        else model
    )

    # GradCAM for each layer
    for row_idx, (layer_name, target_layers) in enumerate(target_layers_list, start=1):
        try:
            with GradCAMPlusPlus(
                model=wrapped_model, target_layers=target_layers
            ) as cam:
                # For correct prediction
                grayscale_cam_correct = cam(input_tensor=correct_img, targets=None)[
                    0, :
                ]
                vis_correct = show_cam_on_image(
                    correct_img_np, grayscale_cam_correct, use_rgb=True
                )

                # For wrong prediction
                grayscale_cam_wrong = cam(input_tensor=wrong_img, targets=None)[0, :]
                vis_wrong = show_cam_on_image(
                    wrong_img_np, grayscale_cam_wrong, use_rgb=True
                )

            axes[row_idx, 0].imshow(vis_correct)
            axes[row_idx, 0].set_title(f"{layer_name}", fontsize=10)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(vis_wrong)
            axes[row_idx, 1].set_title(f"{layer_name}", fontsize=10)
            axes[row_idx, 1].axis("off")

        except Exception as e:
            print(f"Warning: Could not generate GradCAM for {layer_name}: {e}")
            axes[row_idx, 0].text(
                0.5, 0.5, f"Error: {layer_name}", ha="center", va="center"
            )
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].text(
                0.5, 0.5, f"Error: {layer_name}", ha="center", va="center"
            )
            axes[row_idx, 1].axis("off")

    plt.suptitle(f"GradCAM++ Visualization - {model.__class__.__name__}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"GradCAM visualization saved to {save_path}")


def generate_gradcam_visualization_per_class(
    model,
    dataloader,
    device,
    class_names,
    save_path: str = "./plots/gradcam_per_class.png",
):
    """
    Generate two GradCAM visualizations: one for correct predictions, one for wrong.
    Each plot has:
      - Row 0: Original images (8 columns, one per class)
      - Rows 1+: GradCAM for each target layer
    Saves to save_path with "_correct" and "_wrong" suffixes.
    """
    model.eval()

    correct_samples_dict, wrong_samples_dict = find_correct_and_wrong_for_each_class(
        model, dataloader, device, class_names
    )

    target_layers_list = get_gradcam_target_layers(model)
    num_layers = len(target_layers_list)
    num_classes = len(class_names)

    # Prepare save paths
    base_path = save_path.rsplit(".", 1)[0]
    ext = save_path.rsplit(".", 1)[1] if "." in save_path else "png"
    correct_save_path = f"{base_path}_correct.{ext}"
    wrong_save_path = f"{base_path}_wrong.{ext}"

    # Number of rows: 1 (original) + num_layers (GradCAM per layer)
    num_rows = 1 + num_layers

    # --- Plot for CORRECT predictions ---
    fig_correct, axes_correct = plt.subplots(
        num_rows, num_classes, figsize=(3 * num_classes, 3 * num_rows)
    )
    if num_rows == 1:
        axes_correct = axes_correct.reshape(1, -1)

    # --- Plot for WRONG predictions ---
    fig_wrong, axes_wrong = plt.subplots(
        num_rows, num_classes, figsize=(3 * num_classes, 3 * num_rows)
    )
    if num_rows == 1:
        axes_wrong = axes_wrong.reshape(1, -1)

    # Precompute denormalized images and store for GradCAM
    correct_images = {}
    wrong_images = {}
    correct_images_np = {}
    wrong_images_np = {}

    for class_idx, class_name in enumerate(class_names):
        correct_sample = correct_samples_dict[class_name]
        wrong_sample = wrong_samples_dict[class_name]

        # Handle correct samples
        if correct_sample is not None:
            correct_img, correct_gt, correct_pred = correct_sample
            correct_images[class_name] = correct_img
            correct_images_np[class_name] = denormalize_image(correct_img.squeeze(0))

            axes_correct[0, class_idx].imshow(correct_images_np[class_name])
            axes_correct[0, class_idx].set_title(
                f"{correct_gt}\nPred: {correct_pred}", fontsize=8
            )
            axes_correct[0, class_idx].axis("off")
        else:
            axes_correct[0, class_idx].text(
                0.5, 0.5, "No sample", ha="center", va="center"
            )
            axes_correct[0, class_idx].axis("off")

        # Handle wrong samples
        if wrong_sample is not None:
            wrong_img, wrong_gt, wrong_pred = wrong_sample
            wrong_images[class_name] = wrong_img
            wrong_images_np[class_name] = denormalize_image(wrong_img.squeeze(0))

            axes_wrong[0, class_idx].imshow(wrong_images_np[class_name])
            axes_wrong[0, class_idx].set_title(
                f"{wrong_gt}\nPred: {wrong_pred}", fontsize=8
            )
            axes_wrong[0, class_idx].axis("off")
        else:
            axes_wrong[0, class_idx].text(
                0.5, 0.5, "No sample", ha="center", va="center"
            )
            axes_wrong[0, class_idx].axis("off")

    # Wrap model if it returns tuple
    wrapped_model = (
        GradCAMModelWrapper(model)
        if not model.__class__.__name__.startswith("Simple")
        else model
    )

    # Generate GradCAM for each layer
    for row_idx, (layer_name, target_layers) in enumerate(target_layers_list, start=1):
        for class_idx, class_name in enumerate(class_names):
            # Correct predictions
            if class_name in correct_images:
                try:
                    with GradCAMPlusPlus(
                        model=wrapped_model, target_layers=target_layers
                    ) as cam:
                        grayscale_cam = cam(
                            input_tensor=correct_images[class_name], targets=None
                        )[0, :]
                        vis = show_cam_on_image(
                            correct_images_np[class_name], grayscale_cam, use_rgb=True
                        )
                    axes_correct[row_idx, class_idx].imshow(vis)
                    axes_correct[row_idx, class_idx].axis("off")
                    if class_idx == 0:
                        axes_correct[row_idx, class_idx].set_ylabel(
                            layer_name, fontsize=8, rotation=90
                        )
                except Exception:
                    axes_correct[row_idx, class_idx].text(
                        0.5, 0.5, "Error", ha="center", va="center", fontsize=6
                    )
                    axes_correct[row_idx, class_idx].axis("off")
            else:
                axes_correct[row_idx, class_idx].axis("off")

            # Wrong predictions
            if class_name in wrong_images:
                try:
                    with GradCAMPlusPlus(
                        model=wrapped_model, target_layers=target_layers
                    ) as cam:
                        grayscale_cam = cam(
                            input_tensor=wrong_images[class_name], targets=None
                        )[0, :]
                        vis = show_cam_on_image(
                            wrong_images_np[class_name], grayscale_cam, use_rgb=True
                        )
                    axes_wrong[row_idx, class_idx].imshow(vis)
                    axes_wrong[row_idx, class_idx].axis("off")
                    if class_idx == 0:
                        axes_wrong[row_idx, class_idx].set_ylabel(
                            layer_name, fontsize=8, rotation=90
                        )
                except Exception:
                    axes_wrong[row_idx, class_idx].text(
                        0.5, 0.5, "Error", ha="center", va="center", fontsize=6
                    )
                    axes_wrong[row_idx, class_idx].axis("off")
            else:
                axes_wrong[row_idx, class_idx].axis("off")

    target_layers_names = [name for name, _ in target_layers_list]
    # Add row labels on the left
    layer_names = ["Original"] + target_layers_names
    for row_idx, layer_name in enumerate(layer_names):
        axes_correct[row_idx, 0].set_ylabel(
            layer_name, fontsize=9, rotation=90, labelpad=10
        )
        axes_wrong[row_idx, 0].set_ylabel(
            layer_name, fontsize=9, rotation=90, labelpad=10
        )

    # Save correct predictions plot
    fig_correct.suptitle(
        f"GradCAM++ - Correct Predictions - {model.__class__.__name__}", fontsize=14
    )
    fig_correct.tight_layout()
    fig_correct.savefig(correct_save_path, dpi=150, bbox_inches="tight")
    plt.close(fig_correct)
    print(f"GradCAM (correct) saved to {correct_save_path}")

    # Save wrong predictions plot
    fig_wrong.suptitle(
        f"GradCAM++ - Wrong Predictions - {model.__class__.__name__}", fontsize=14
    )
    fig_wrong.tight_layout()
    fig_wrong.savefig(wrong_save_path, dpi=150, bbox_inches="tight")
    plt.close(fig_wrong)
    print(f"GradCAM (wrong) saved to {wrong_save_path}")


def global_prune_model(
    model, parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2
):
    """
    Applies global pruning to the specified parameters of the model.

    Args:
        model (torch.nn.Module): The model to be pruned.
        parameters_to_prune (list): List of tuples specifying the parameters to prune.
        pruning_method (callable): The pruning method from torch.nn.utils.prune.
        amount (float): The fraction of parameters to prune globally.
    """

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount=amount,
    )


if __name__ == "__main__":
    config = argparse.Namespace(
        # Architecture
        data_dir="./data/2425/MIT_small_train_1",  # MIT_small_train_1, MIT_large_train
        use_model="v3",  # v1, v2, v3, v_attn, shuffle_v_attn
        img_size=64,  # v1: 128, v2: 64, v3: 64, v_attn: 64, shuffle_v_attn: 224
        batch_size=32,
        load_from_trained=False,
        do_cross_validation=False,
        num_folds=4,
        num_epochs=1000,
        lr=0.0001,  # v1: 0.001, v2: 0.0001, v3: 0.0001, v_attn: 0.001, shuffle_v_attn: 0.001
        early_stopping=50,
        scheduler="none",  # v1: none, v2: none, v3: none, v_attn: none, shuffle_v_attn: plateau
        use_batchnorm=True,
        optimizer="adamw",  # v1: adam, v2: adamw, v3: adamw, v_attn: adamw, shuffle_v_attn: adamw
        label_smoothing=0.074,  # 0.074
        weight_decay=0.00045,  # 0.00045
        momentum=0.108,  # 0.108
        dropout_rate=0.1,
        use_attn=True,
        knowledge_distillation=False,
        feature_map_weight=0.5,
        temperature=4.0,
        distillation_weight=0.25,
    )

    # config = argparse.Namespace(
    #     # Architecture
    #     data_dir="./data/2425/MIT_small_train_1",
    #     use_model="v1",  # v1, v2, v3, v_attn, shuffle_v_attn
    #     img_size=128,  # v1: 128, v2: 128, v3: 224, v_attn: 224, shuffle_v_attn: 224
    #     batch_size=64,
    #     do_cross_validation=True,
    #     num_folds=5,
    #     num_epochs=500,
    #     lr=0.01,  # 0.0075
    #     early_stopping=50,
    #     scheduler="plateau",  # plateau
    #     use_batchnorm=True,
    #     optimizer="adamw",  # adamw
    #     label_smoothing=0.1,  # 0.074
    #     weight_decay=0.0,  # 0.00045
    #     momentum=0.0,  # 0.108
    #     dropout_rate=0.1,
    #     use_attn=False,
    #     knowledge_distillation=False,
    #     feature_map_weight=0.5,
    #     temperature=16.0,
    #     distillation_weight=0.25,
    # )

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

    if config.knowledge_distillation:
        torch.manual_seed(42)
        teacher_model = TeacherModel(
            num_classes=num_classes,
            arch_type="modified",
            dropout_rate=0.105,
            use_batchnorm=True,
        ).to(device)
        teacher_model.load_state_dict(torch.load("./teacher_model.pth"))
        teacher_model.eval()
    else:
        teacher_model = None

    if config.do_cross_validation:
        cross_validate(
            dataset_root_path="./data/2425/",
            teacher_model=teacher_model,
            config=config,
            device=device,
            use_wandb=False,
        )
    else:
        torch.manual_seed(42)
        if config.use_model == "v1":
            model = SimpleCNNv1(
                num_classes=num_classes,
            ).to(device)
        elif config.use_model == "v2":
            model = SimpleCNNv2(
                num_classes=num_classes,
            ).to(device)
        elif config.use_model == "v3":
            model = SimpleCNNv3(
                num_classes=num_classes,
                image_input_size=config.img_size,
            ).to(device)
        elif config.use_model == "v_attn":
            model = SimpleCNNvAttn(
                image_input_size=config.img_size,
                num_classes=num_classes,
                dropout_rate=config.dropout_rate,
                use_attn=config.use_attn,
            ).to(device)
        elif config.use_model == "shuffle_v_attn":
            model = ShuffleCNNvAttn(
                num_classes=num_classes,
                dropout_rate=config.dropout_rate,
                use_attn=config.use_attn,
                knowledge_distillation=config.knowledge_distillation,
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {config.use_model}")

        criterion = get_criterion("ce", label_smoothing=config.label_smoothing)
        optimizer = create_optimizer(
            filter(lambda p: p.requires_grad, model.parameters()), config
        )
        scheduler = get_scheduler(
            optimizer, config.scheduler, num_epochs=config.num_epochs
        )
        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        if config.load_from_trained:
            if config.use_model == "v1":
                model.load_state_dict(torch.load("./v1.pth"))
            elif config.use_model == "v2":
                model.load_state_dict(torch.load("./v2.pth"))
            elif config.use_model == "v3":
                model.load_state_dict(torch.load("./v3.pth"))
            elif config.use_model == "v_attn":
                model.load_state_dict(torch.load("./v_attn.pth"))
            elif config.use_model == "shuffle_v_attn":
                model.load_state_dict(torch.load("./shuffle_v_attn.pth"))
            print(f"Loaded model from ./{config.use_model}.pth")
        else:
            normal_train(
                model,
                teacher_model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                scheduler,
                config,
                device,
                train_losses,
                train_accuracies,
                test_losses,
                test_accuracies,
            )

        # parameters_to_prune = [
        #     (model.conv1, "weight"),
        #     (model.stage2, "weight"),
        #     (model.stage3, "weight"),
        #     (model.conv4, "weight"),
        #     (model.fc, "weight"),
        # ]
        # global_prune_model(model, parameters_to_prune)

        if not config.load_from_trained:
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
            cm_save_path=f"./plots/confusion_matrix_{config.use_model}.png",
        )

        # Generate GradCAM visualization
        generate_gradcam_visualization(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            save_path=f"./plots/gradcam_{config.use_model}.png",
        )

        generate_gradcam_visualization_per_class(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            save_path=f"./plots/gradcam_per_class_{config.use_model}.png",
        )
