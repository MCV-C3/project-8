import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as F
import tqdm
from models import SimpleModel
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import ImageFolder
from utils import PatchDataset


# -----------------------------
# helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def compute_mean_std(loader, max_batches=50):
    n = 0
    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        b = x.size(0)
        x = x.view(b, 3, -1)
        mean += x.mean(dim=2).sum(dim=0)
        m2 += (x**2).mean(dim=2).sum(dim=0)
        n += b
    mean /= n
    var = (m2 / n) - (mean**2)
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean.tolist(), std.tolist()


# -----------------------------
# train / test (patch-based)
# -----------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in dataloader:
        # x: [B, N, 3, H, W]
        B, N, C, H, W = x.shape
        x, y = x.to(device), y.to(device)
        x = x.view(B * N, C, H, W)

        logits = model(x)  # [B*N, K]
        logits = logits.view(B, N, -1)  # [B, N, K]
        img_logits = logits.mean(dim=1)  # [B, K]

        loss = criterion(img_logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * B
        correct += (img_logits.argmax(1) == y).sum().item()
        total += B

    return loss_sum / total, correct / total


@torch.no_grad()
def test_epoch(model, dataloader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in dataloader:
        B, N, C, H, W = x.shape
        x, y = x.to(device), y.to(device)
        x = x.view(B * N, C, H, W)

        logits = model(x)
        logits = logits.view(B, N, -1)
        img_logits = logits.mean(dim=1)

        loss = criterion(img_logits, y)
        loss_sum += loss.item() * B
        correct += (img_logits.argmax(1) == y).sum().item()
        total += B

    return loss_sum / total, correct / total


# -----------------------------
# feature extraction for SVM
# -----------------------------
@torch.no_grad()
def extract_single_feature_1d(model, dataloader, device, mode="mean", dim_idx=0):
    """
    Returns X: (num_images, 1) and y: (num_images,)
    Requires: model(x, return_features=True) -> (logits, feats)
    feats expected shape: (B*N, hidden_d)
    """
    model.eval()
    X_list, y_list = [], []

    for x, y in dataloader:
        B, N, C, H, W = x.shape
        x = x.to(device)

        x = x.view(B * N, C, H, W)
        _, feats = model(x, return_features=True)  # (B*N, hidden_d)
        feats = feats.view(B, N, -1).mean(dim=1)  # (B, hidden_d) per image

        if mode == "mean":
            one = feats.mean(dim=1, keepdim=True)  # (B,1)
        elif mode == "dim":
            one = feats[:, dim_idx : dim_idx + 1]  # (B,1)
        else:
            raise ValueError("mode must be 'mean' or 'dim'")

        X_list.append(one.cpu().numpy())
        y_list.append(y.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def train_eval_svm_1d(Xtr, ytr, Xte, yte, C=1.0):
    clf = make_pipeline(StandardScaler(), LinearSVC(C=C, dual=True, max_iter=10000))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return accuracy_score(yte, pred)


# -----------------------------
# plots
# -----------------------------
def save_loss_plot(train_losses, test_losses, outpath):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("End-to-End Train/Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved:", outpath)


def save_acc_plot(train_accs, test_accs, outpath):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(test_accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("End-to-End Train/Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved:", outpath)


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_path = os.path.expanduser("./data/2425/MIT_large_train/train")
    test_path = os.path.expanduser("./data/2425/MIT_large_train/test")

    IMG = 64
    NUM_PATCHES = 16
    EPOCHS = 50

    # Best model hyperparams (fixed)
    BASE = dict(
        n_hidden_layers=3,
        hidden_d=768,
        dropout=0.6,
        activation="gelu",
        order="linear_bn_act_do",
        input_l2norm=False,
        lr=2e-4,
        wd=1e-4,
    )

    ensure_dir("plots")

    # 1) dataset mean/std
    tf0 = F.Compose(
        [
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize((IMG, IMG)),
        ]
    )
    tmp_train = ImageFolder(train_path, transform=tf0)
    tmp_loader = DataLoader(tmp_train, batch_size=256, shuffle=True, num_workers=8)

    ds_mean, ds_std = compute_mean_std(tmp_loader, max_batches=50)
    print("Dataset mean:", ds_mean)
    print("Dataset std :", ds_std)

    # 2) final transform (dataset normalize)
    tf = F.Compose(
        [
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize((IMG, IMG)),
            F.Normalize(mean=ds_mean, std=ds_std),
        ]
    )

    # 3) datasets/loaders with 16 patches
    train_ds = PatchDataset(train_path, tf, patch_size=IMG, num_patches=NUM_PATCHES)
    test_ds = PatchDataset(test_path, tf, patch_size=IMG, num_patches=NUM_PATCHES)

    num_classes = len(ImageFolder(train_path).classes)
    print("Num classes:", num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
    )

    # 4) model
    model = SimpleModel(
        input_d=3 * IMG * IMG,
        hidden_d=BASE["hidden_d"],
        output_d=num_classes,
        n_hidden_layers=BASE["n_hidden_layers"],
        dropout=BASE["dropout"],
        activation=BASE["activation"],
        order=BASE["order"],
        input_l2norm=BASE["input_l2norm"],
    ).to(device)

    print(summary(model, (3, IMG, IMG)))
    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=BASE["lr"], weight_decay=BASE["wd"])

    # 5) train end-to-end + store curves
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_test_acc = 0.0

    for epoch in tqdm.tqdm(range(EPOCHS), desc=f"TRAIN E2E (patches={NUM_PATCHES})"):
        tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer, device)
        te_l, te_a = test_epoch(model, test_loader, criterion, device)

        train_losses.append(tr_l)
        test_losses.append(te_l)
        train_accs.append(tr_a)
        test_accs.append(te_a)
        best_test_acc = max(best_test_acc, te_a)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | train acc {tr_a:.4f} | test acc {te_a:.4f}"
            )

    print("\n[END-TO-END] best test acc:", best_test_acc)

    # save plots
    save_loss_plot(train_losses, test_losses, "plots/loss.png")
    save_acc_plot(train_accs, test_accs, "plots/test_acc.png")

    # 6) SVM with a single feature (1D)
    # choose one:
    # - mode="mean": scalar = mean over hidden vector
    # - mode="dim": scalar = one feature dimension (dim_idx)
    Xtr, ytr = extract_single_feature_1d(model, train_loader, device, mode="mean")
    Xte, yte = extract_single_feature_1d(model, test_loader, device, mode="mean")
    svm_acc = train_eval_svm_1d(Xtr, ytr, Xte, yte, C=1.0)

    print("\n[SVM-1D] test acc:", svm_acc)
    print("[COMPARISON] End-to-End best test acc:", best_test_acc)
