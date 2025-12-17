import os
import sys

import numpy as np
import torch
import torchvision.transforms.v2 as F
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Add project root to path to import modules from Week1 and Week2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Week1.bovw import BOVW
from Week2.main import compute_mean_std, make_tf
from Week2.models import SimpleModel
from Week2.utils import PatchDataset


def extract_mlp_features(model, dataloader, device):
    """
    Extracts features using the trained MLP model.
    Returns a list of descriptors (one list of descriptors per image).
    Each descriptor corresponds to the output of the MLP backbone for a patch.
    """
    model.eval()
    all_descriptors = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm.tqdm(dataloader, desc="Extracting MLP features"):
            # x shape: [B, N, C, H, W]
            B, N, C, H, W = x.shape
            x = x.to(device)
            x = x.view(B * N, C, H, W)

            # Get features from the backbone (before the classification head)
            features = model(x, return_features=True)  # Shape: [B*N, hidden_d]

            # Reshape back to [B, N, hidden_d]
            features = features.view(B, N, -1)

            # Convert to numpy and store
            features_np = features.cpu().numpy()

            for i in range(B):
                all_descriptors.append(features_np[i])  # List of [N, hidden_d] arrays
                all_labels.append(y[i].item())

    return all_descriptors, all_labels


def train_bovw_classifier(descriptors, labels, bovw, map_classes):
    print("Fitting Codebook...")
    # Flatten descriptors for KMeans: List of [N, D] -> [Total_N, D]
    # But BOVW._update_fit_codebook expects a list of arrays where each array is descriptors for one image
    # Actually, looking at bovw.py: _update_fit_codebook takes descriptors: Literal["N", "T", "d"]
    # and does np.vstack(descriptors). So passing the list of [N, D] arrays is correct.

    bovw._update_fit_codebook(descriptors)

    print("Computing Histograms...")
    # Compute histograms for each image
    # BOVW.extract_bovw_histograms is not a method of BOVW class in the provided snippet,
    # but a standalone function in Week1/main.py.
    # However, we can use the internal methods or reimplement the histogram extraction here using the trained codebook.

    all_histograms = []
    for img_descs in tqdm.tqdm(descriptors, desc="Encoding images"):
        # img_descs: [N, D]
        # Predict visual words for these descriptors
        hist = bovw._compute_codebook_descriptor(img_descs, bovw.codebook_algo)
        all_histograms.append(hist)

    all_histograms = np.array(all_histograms)
    all_labels = np.array(labels)

    print("Training Classifier...")
    scaler = StandardScaler()
    all_histograms_scaled = scaler.fit_transform(all_histograms)

    classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier.fit(all_histograms_scaled, all_labels)

    return bovw, classifier, scaler


def evaluate_bovw_classifier(
    descriptors, labels, bovw, classifier, scaler, map_classes
):
    print("Evaluating...")
    all_histograms = []
    for img_descs in tqdm.tqdm(descriptors, desc="Encoding test images"):
        hist = bovw._compute_codebook_descriptor(img_descs, bovw.codebook_algo)
        all_histograms.append(hist)

    all_histograms = np.array(all_histograms)
    all_histograms_scaled = scaler.transform(all_histograms)
    all_labels = np.array(labels)

    y_pred = classifier.predict(all_histograms_scaled)

    acc = accuracy_score(all_labels, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Inverse map for classification report
    target_names = [v for k, v in map_classes.items()]

    print(classification_report(all_labels, y_pred, target_names=target_names))
    return acc


if __name__ == "__main__":
    # Configuration
    IMG = 64
    NUM_PATCHES = 16  # Same as used in MLP training
    CODEBOOK_SIZE = 1600
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Paths
    train_path = os.path.expanduser("./data/train")
    test_path = os.path.expanduser("./data/val")
    model_path = f"mlp_model_{NUM_PATCHES}_patches.pth"  # Path to the saved MLP model

    if not os.path.exists(model_path):
        print(
            f"Error: Model file {model_path} not found. Please run Week2/main.py first to train and save the MLP."
        )
        sys.exit(1)

    # 1. Setup Data and Model (Same as Week2/main.py)
    # Compute Mean/Std (or hardcode if known from previous run)
    # For consistency, we'll recompute or use standard values.
    # Let's assume we use the same normalization as the best model from Week 2.
    # Based on user's Week2/main.py, it seems 'dataset' normalization was used.

    print("Computing dataset statistics...")
    tf0 = F.Compose(
        [F.ToImage(), F.ToDtype(torch.float32, scale=True), F.Resize((IMG, IMG))]
    )
    tmp_train = ImageFolder(train_path, transform=tf0)
    tmp_loader = DataLoader(tmp_train, batch_size=256, shuffle=True, num_workers=8)
    ds_mean, ds_std = compute_mean_std(tmp_loader, max_batches=50)
    print("Dataset mean:", ds_mean)
    print("Dataset std :", ds_std)

    # Transform
    # Assuming best model used 'dataset' normalization and no grayscale
    tf = make_tf(IMG, "dataset", ds_mean, ds_std, False)

    # Datasets
    data_train = PatchDataset(train_path, tf, patch_size=IMG, num_patches=NUM_PATCHES)
    data_test = PatchDataset(test_path, tf, patch_size=IMG, num_patches=NUM_PATCHES)

    train_loader = DataLoader(
        data_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
    )
    test_loader = DataLoader(
        data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
    )

    # Load MLP Model
    num_classes = len(ImageFolder(train_path).classes)
    C, H, W = 3, IMG, IMG
    input_d = C * H * W

    # Hyperparams must match the saved model
    # Assuming the best model from Week2/main.py loop was saved.
    # The loop in Week2/main.py iterates but overwrites 'best_model' dict.
    # The last iteration's config is what we likely have if we just ran it.
    # However, the user's code saves `mlp_model_{n_patches}_patches.pth` inside the loop.
    # We need to know the architecture parameters.
    # Defaulting to the 'base' params in Week2/main.py:
    hidden_d = 768
    n_hidden_layers = 3
    dropout = 0.6
    order = "linear_bn_act_do"  # Default in main.py
    input_l2 = False  # Default in main.py

    model = SimpleModel(
        input_d=input_d,
        hidden_d=hidden_d,
        output_d=num_classes,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout,
        order=order,
        input_l2norm=input_l2,
    ).to(device)

    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 2. Extract Features
    print("Extracting training features...")
    train_descs, train_labels = extract_mlp_features(model, train_loader, device)

    print("Extracting test features...")
    test_descs, test_labels = extract_mlp_features(model, test_loader, device)

    # 3. BoVW Pipeline
    print("Initializing BoVW...")
    # We use BOVW class but we won't use its detector (SIFT/ORB) since we already have descriptors.
    # We just use it for KMeans codebook.
    bovw = BOVW(codebook_size=CODEBOOK_SIZE)

    # Map classes
    map_classes = {v: k for k, v in ImageFolder(train_path).class_to_idx.items()}

    # Train BoVW
    bovw, classifier, scaler = train_bovw_classifier(
        train_descs, train_labels, bovw, map_classes
    )

    # Test BoVW
    evaluate_bovw_classifier(
        test_descs, test_labels, bovw, classifier, scaler, map_classes
    )
