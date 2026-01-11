from typing import *
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F

from sklearn.model_selection import StratifiedKFold

from models import SimpleModel, WraperModel
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as fnc


# Train function
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

    return train_loss / total, correct / total


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

    return test_loss / total, correct / total


def cross_validate(
    dataset,
    model_fn,
    optimizer_fn,
    criterion,
    device,
    num_epochs=3,
    num_folds=5,
    batch_size=16,
    num_workers=8
):
    targets = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets)
    ):
        print(f"\n========== Fold {fold + 1}/{num_folds} ==========")

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

        # NEW MODEL FOR EACH FOLD
        model = model_fn().to(device)
        optimizer = optimizer_fn(
            filter(lambda p: p.requires_grad, model.parameters())
        )

        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = test(
                model, val_loader, criterion, device
            )

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

        fold_accuracies.append(val_acc)

    print("\n========== Cross-validation Summary ==========")
    print(f"Fold Accuracies: {fold_accuracies}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std Accuracy:  {np.std(fold_accuracies):.4f}")

    return fold_accuracies


def create_model():
    model = WraperModel(num_classes=8, feature_extraction=False, remove_blocks=True, new_classification=False, change_pooling=False)
    
    # For task 0 partial FineTunning
    """
    for name, param in model.backbone.named_parameters():
        if "classifier" in name or "features.6" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    """

    return model


def create_optimizer(params):
    return optim.Adam(params, lr=0.001)



if __name__ == "__main__":

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(224, 224)),
    ])

    data_train = ImageFolder(
        "mcv/datasets/C3/2425/MIT_large_train/train",
        transform=transformation,
    )

    data_test = ImageFolder(
        "mcv/datasets/C3/2425/MIT_large_train/test",
        transform=transformation,
    )

    criterion = nn.CrossEntropyLoss()

    cross_validate(
        dataset=data_train,
        model_fn=create_model,
        optimizer_fn=create_optimizer,
        criterion=criterion,
        device=device,
        num_epochs=3,
        num_folds=5,
        batch_size=16,
        num_workers=8,
    )

    # -----------------------------------------------------
    # Final training on full training set
    # -----------------------------------------------------
    train_loader = DataLoader(
        data_train, batch_size=16, shuffle=True, num_workers=8, pin_memory=True
    )

    test_loader = DataLoader(
        data_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    model = create_model().to(device)
    optimizer = create_optimizer(
        filter(lambda p: p.requires_grad, model.parameters())
    )

    num_epochs = 3

    tic = time.perf_counter()

    for epoch in tqdm.tqdm(range(num_epochs), desc="FINAL TRAINING"):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = test(
            model, test_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

    toc = time.perf_counter()
    print(f"Total training time: {toc - tic:.2f} seconds")
    