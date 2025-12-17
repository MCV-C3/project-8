import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as F
import tqdm
from models import SimpleModel
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import PatchDataset


def train(
    model, dataloader, criterion, optimizer, device, class_names, plot_once=False
):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in dataloader:
        # x with patches: [B, N, 3, H, W]
        B, N, C, H, W = x.shape
        x, y = x.to(device), y.to(device)
        x = x.view(B * N, C, H, W)

        logits = model(x)
        logits = logits.view(B, N, -1)
        img_logits = logits.mean(dim=1)

        if plot_once:
            # take first image in batch
            patch_probs = torch.softmax(logits[0], dim=1)  # [N, classes]
            merged_probs = torch.softmax(img_logits[0], dim=0)  # [classes]

            _plot_patch_merge(patch_probs.cpu(), merged_probs.cpu(), class_names)

            plot_once = False

        loss = criterion(img_logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (img_logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def test(model, dataloader, criterion, device):
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
        loss_sum += loss.item() * x.size(0)
        correct += (img_logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def compute_mean_std(loader, max_batches=50):
    n = 0
    mean = torch.zeros(3)
    m2 = torch.zeros(3)  # second moment
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        # x in [0,1], shape [B,3,H,W]
        b = x.size(0)
        x = x.view(b, 3, -1)
        mean += x.mean(dim=2).sum(dim=0)
        m2 += (x**2).mean(dim=2).sum(dim=0)
        n += b
    mean /= n
    var = (m2 / n) - (mean**2)
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean.tolist(), std.tolist()


def make_tf(img, normalize_mode: str, mean, std, grayscale: bool):
    ops = [
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize((img, img)),
    ]
    if grayscale:
        # manté 3 canals (compatible amb mean/std 3D)
        ops.append(F.Grayscale(num_output_channels=3))

    if normalize_mode == "imagenet":
        ops.append(F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif normalize_mode == "dataset":
        ops.append(F.Normalize(mean=mean, std=std))
    elif normalize_mode == "none":
        pass
    else:
        raise ValueError(normalize_mode)

    return F.Compose(ops)


@torch.no_grad()
def _plot_patch_merge(patch_probs, merged_probs, class_names, topk=5):
    top = torch.topk(merged_probs, topk)
    cls = top.indices.numpy()

    x = np.arange(topk)
    width = 0.15

    plt.figure(figsize=(10, 4))

    for i in range(patch_probs.shape[0]):
        plt.bar(x + i * width, patch_probs[i, cls], width, label=f"Patch {i}")

    plt.bar(
        x + patch_probs.shape[0] * width,
        merged_probs[cls],
        width,
        label="Merged",
        color="black",
    )

    plt.xticks(
        x + width * patch_probs.shape[0] / 2, [class_names[i] for i in cls], rotation=30
    )

    plt.ylabel("Probability")
    plt.title("Patch predictions → merged prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()


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


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_path = os.path.expanduser("./data/train")
    test_path = os.path.expanduser("./data/val")

    IMG = 64
    EPOCHS = 300
    NUM_PATCHES = [16]

    # 1) dataset mean/std (aprox)
    tf0 = F.Compose(
        [F.ToImage(), F.ToDtype(torch.float32, scale=True), F.Resize((IMG, IMG))]
    )
    tmp_train = ImageFolder(train_path, transform=tf0)
    tmp_loader = DataLoader(tmp_train, batch_size=256, shuffle=True, num_workers=8)
    ds_mean, ds_std = compute_mean_std(tmp_loader, max_batches=50)
    print("Dataset mean:", ds_mean)
    print("Dataset std :", ds_std)

    # 2) experiments “tot a la vegada” (16 runs)
    # Base hiperparams (els teus millors)
    base = dict(n_hidden_layers=3, hidden_d=768, dropout=0.6, lr=0.0002, wd=1e-4)

    preprocess_variants = [
        # ("imagenet", False),
        ("dataset", False),
        # ("dataset",  True),   # grayscale
        # ("imagenet", True),   # grayscale
    ]

    model_variants = [
        ("linear_bn_act_do", False),  # ordre normal, no L2
        # ("linear_bn_act_do", True),    # + input L2
        # ("do_linear_bn_act", False),   # input dropout ordre
        # ("do_linear_bn_act", True),    # input dropout + input L2
    ]

    # construeix lista configs
    experiments = []
    for norm_mode, gray in preprocess_variants:
        for order, l2 in model_variants:
            name = f"N{norm_mode}_G{int(gray)}_O{order}_L2{int(l2)}"
            experiments.append(
                dict(
                    name=name,
                    norm_mode=norm_mode,
                    grayscale=gray,
                    order=order,
                    input_l2=l2,
                )
            )
            best_model = dict(
                name=name, norm_mode=norm_mode, grayscale=gray, order=order, input_l2=l2
            )

    all_results = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for n_patches in NUM_PATCHES:
        print("\n==============================")
        print("RUN:", best_model["name"])
        print("==============================")

        train_tf = make_tf(
            IMG, best_model["norm_mode"], ds_mean, ds_std, best_model["grayscale"]
        )
        test_tf = make_tf(
            IMG, best_model["norm_mode"], ds_mean, ds_std, best_model["grayscale"]
        )

        data_train = PatchDataset(
            train_path, train_tf, patch_size=IMG, num_patches=n_patches
        )
        data_test = PatchDataset(
            test_path, test_tf, patch_size=IMG, num_patches=n_patches
        )

        num_classes = len(ImageFolder(train_path).classes)

        train_loader = DataLoader(
            data_train, batch_size=64, pin_memory=True, shuffle=True, num_workers=8
        )
        test_loader = DataLoader(
            data_test, batch_size=64, pin_memory=True, shuffle=False, num_workers=8
        )

        patches, label = data_train[0]

        C, H, W = 3, IMG, IMG
        input_d = C * H * W

        model = SimpleModel(
            input_d=input_d,
            hidden_d=base["hidden_d"],
            output_d=num_classes,
            n_hidden_layers=base["n_hidden_layers"],
            dropout=base["dropout"],
            order=best_model["order"],
            input_l2norm=best_model["input_l2"],
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=base["lr"], weight_decay=base["wd"]
        )

        class_names = data_train.dataset.classes
        best_test_l = float("inf")
        for epoch in tqdm.tqdm(range(EPOCHS), desc=f"TRAIN {n_patches}"):
            tr_l, tr_a = train(
                model, train_loader, criterion, optimizer, device, class_names
            )
            te_l, te_a = test(model, test_loader, criterion, device)

            train_losses.append(tr_l)
            test_losses.append(te_l)
            train_accuracies.append(tr_a)
            test_accuracies.append(te_a)
            if te_l < best_test_l:
                # Save the trained model
                model_save_path = f"mlp_model_{n_patches}_patches.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

            best_test_l = min(best_test_l, te_l)
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | train loss {tr_l:.4f} | train acc {tr_a:.4f} | test loss {te_l:.4f} | test acc {te_a:.4f}"
            )

        all_results.append(("TRAIN " + str(n_patches), best_test_l))

    plot_metrics({"loss": train_losses}, {"loss": test_losses}, metric_name="loss")
    plot_metrics(
        {"accuracy": train_accuracies},
        {"accuracy": test_accuracies},
        metric_name="accuracy",
    )
    print("\n===== SUMMARY (best test loss) =====")
    for name, acc in sorted(all_results, key=lambda x: x[1], reverse=False):
        print(f"{name:60s} {acc:.4f}")
