import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as F
import tqdm
from models import SimpleModel
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchviz import make_dot
from utils import PatchDataset

sys.path.append("/home/jchen/project-8")


def train(model, dataloader, criterion, optimizer, device):
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
        # x in [0,1], shape [B,N,3,H,W]
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


def plot_computational_graph(
    model: torch.nn.Module,
    input_size: tuple,
    filename: str = "computational_graph",
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
    dummy_input = torch.randn(*input_size).to("cuda")

    # Create a graph from the model
    _ = make_dot(
        model(dummy_input), params=dict(model.named_parameters()), show_attrs=True
    ).render(filename, format="png")

    print(f"Computational graph saved as {filename}")


def get_layer_output(
    model: nn.Module, input_tensor: torch.Tensor, layer_name: str
) -> torch.Tensor:
    """
    Retrieves the output of a specific layer in the model for a given input image.

    Args:
        model (nn.Module): The trained model.
        input_tensor (torch.Tensor): The input image tensor.
        layer_name (str): The name of the layer to retrieve the output from.

    Returns:
        torch.Tensor: The output of the specified layer.
    """
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Find the layer by name
    found = False
    for name, layer in model.named_modules():
        if name == layer_name:
            handle = layer.register_forward_hook(get_activation(layer_name))
            found = True
            break

    if not found:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    # Forward pass
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        model(input_tensor)

    # Remove hook
    handle.remove()

    return activation[layer_name]


def plot_activations(activations, input, gt, pred):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 20))
    axes[0, 0].axis("off")
    axes[0, 2].axis("off")
    axes[0, 1].set_title(f"Input Image, class: {gt}, predicted: {pred}", fontsize=26)
    axes[0, 1].imshow(input[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 1].axis("off")
    for i in range(3):
        act = activations[f"layer{i + 1}"][0]
        reshaped_act = act.view(16, 16, 3).cpu().numpy()
        # reshaped_act_norm = (reshaped_act - reshaped_act.min()) / (reshaped_act.max() - reshaped_act.min())
        for j in range(3):
            axes[i + 1, j].imshow(reshaped_act[:, :, j], cmap="viridis")
            axes[i + 1, j].set_title(
                f"Layer {i + 1} - From {j * 256} to {(j + 1) * 256} tensor", fontsize=26
            )
            axes[i + 1, j].axis("off")
    plt.tight_layout()
    plt.show()


def plot_classification_results(
    dict_counts: Dict[str, int], title: str = "Classification Results"
):
    """
    Plots a bar chart of classification results.

    Args:
        dict_counts (Dict[str, int]): A dictionary with class names as keys and counts as values.
        title (str): Title of the plot.
    """
    classes = list(dict_counts.keys())
    counts = list(dict_counts.values())

    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color="skyblue")
    plt.xlabel("Classes", fontsize=22)
    plt.ylabel("Number of Predictions", fontsize=22)
    plt.title(title, fontsize=22)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    print(f"Classification results plot saved as {title.replace(' ', '_').lower()}.png")
    plt.close()


def visualize_layer_output(
    output_tensor: torch.Tensor, layer_name: str = "layer_output"
):
    """
    Visualizes the output of a layer.

    Args:
        output_tensor (torch.Tensor): The output tensor from the layer.
        layer_name (str): Name of the layer for the plot title.
    """
    output_data = output_tensor.detach().cpu().numpy().flatten()

    plt.figure(figsize=(12, 5))

    # Bar plot
    plt.subplot(1, 2, 1)
    plt.bar(range(len(output_data)), output_data)
    plt.xlabel("Tensor Index", fontsize=22)
    plt.ylabel("Activation", fontsize=22)
    plt.title(f"Activation Values for {layer_name}", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(output_data, bins=30)
    plt.xlabel("Activation Value", fontsize=22)
    plt.ylabel("Frequency", fontsize=22)
    plt.title(f"Activation Distribution for {layer_name}", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    filename = f"{layer_name}_visualization.png"
    plt.savefig(filename)
    print(f"Layer visualization saved as {filename}")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_path = os.path.expanduser("mcv/datasets/C3/2526/places_reduced/train")
    test_path = os.path.expanduser("mcv/datasets/C3/2526/places_reduced/val")

    IMG = 64
    EPOCHS = 20
    NUM_PATCHES = 4

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
    base = dict(n_hidden_layers=3, hidden_d=768, dropout=0.6, lr=8e-4, wd=1e-4)

    preprocess_variants = [
        ("imagenet", False),
        ("dataset", False),
        ("dataset", True),  # grayscale
        ("imagenet", True),  # grayscale
    ]

    model_variants = [
        ("linear_bn_act_do", False),  # ordre normal, no L2
        ("linear_bn_act_do", True),  # + input L2
        ("do_linear_bn_act", False),  # input dropout ordre
        ("do_linear_bn_act", True),  # input dropout + input L2
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

    all_results = []

    for cfg in experiments:
        print("\n==============================")
        print("RUN:", cfg["name"])
        print("==============================")

        train_tf = make_tf(IMG, cfg["norm_mode"], ds_mean, ds_std, cfg["grayscale"])
        test_tf = make_tf(IMG, cfg["norm_mode"], ds_mean, ds_std, cfg["grayscale"])

        data_train = PatchDataset(
            train_path, train_tf, patch_size=IMG, num_patches=NUM_PATCHES
        )
        data_test = PatchDataset(
            test_path, test_tf, patch_size=IMG, num_patches=NUM_PATCHES
        )

        num_classes = len(ImageFolder(train_path).classes)

        train_loader = DataLoader(
            data_train, batch_size=64, pin_memory=True, shuffle=True, num_workers=8
        )
        test_loader = DataLoader(
            data_test, batch_size=64, pin_memory=True, shuffle=False, num_workers=8
        )

        C, H, W = 3, IMG, IMG
        input_d = C * H * W

        model = SimpleModel(
            input_d=input_d,
            hidden_d=base["hidden_d"],
            output_d=num_classes,
            n_hidden_layers=base["n_hidden_layers"],
            dropout=base["dropout"],
            order=cfg["order"],
            input_l2norm=cfg["input_l2"],
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=base["lr"], weight_decay=base["wd"]
        )

        best_test_acc = 0.0
        for epoch in tqdm.tqdm(range(EPOCHS), desc=f"TRAIN {cfg['name']}"):
            tr_l, tr_a = train(model, train_loader, criterion, optimizer, device)
            te_l, te_a = test(model, test_loader, criterion, device)
            best_test_acc = max(best_test_acc, te_a)
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | train acc {tr_a:.4f} | test acc {te_a:.4f}"
            )

        all_results.append((cfg["name"], best_test_acc))

    print("\n===== SUMMARY (best test acc) =====")
    for name, acc in sorted(all_results, key=lambda x: x[1], reverse=True):
        print(f"{name:60s} {acc:.4f}")
