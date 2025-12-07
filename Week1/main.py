import glob
import os
from pathlib import Path
from typing import List, Literal, Tuple, Type

import numpy as np
import tqdm
from bovw import BOVW
from natsort import natsorted
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array(
        [
            bovw._compute_codebook_descriptor(
                descriptors=descriptor, kmeans=bovw.codebook_algo
            )
            for descriptor in descriptors
        ]
    )


def compute_metrics(y_true: List[int], y_pred: List[int], map_classes: dict):
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    n_classes = len(map_classes)
    false_positive_rates = []
    true_positive_rates = []
    auc_scores = []

    for i in range(n_classes):
        y_mask = [1 if label == i else 0 for label in y_true]
        y_pred_mask = [1 if label == i else 0 for label in y_pred]
        false_positive_rates[i], true_positive_rates[i], _ = roc_curve(
            y_mask, y_pred_mask
        )
        auc_scores.append(auc(false_positive_rates[i], true_positive_rates[i]))
    auc_average = np.sum(auc_scores) / n_classes

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "num_classes": n_classes,
        "false_positive_rates": false_positive_rates,
        "true_positive_rates": true_positive_rates,
        "auc_scores": auc_scores,
        "auc_average": auc_average,
    }


def test(
    dataset: List[Tuple[Type[Image.Image], int]],
    bovw: Type[BOVW],
    classifier: Type[object],
    map_classes: dict,
    batch_size: int = 50,
):
    all_histograms = []
    all_labels = []

    for i in tqdm.tqdm(
        range(0, len(dataset), batch_size), desc="Phase [Eval]: Testing"
    ):
        batch = dataset[i : i + batch_size]
        batch_descriptors = []
        batch_labels = []

        for image, label, img_path in batch:
            _, descriptors = bovw._extract_features_dense(
                image=np.array(image), image_path=Path(img_path)
            )

            if descriptors is not None and len(descriptors) > 0:
                batch_descriptors.append(descriptors)
                batch_labels.append(label)

        if batch_descriptors:
            histograms = extract_bovw_histograms(
                descriptors=batch_descriptors, bovw=bovw
            )
            all_histograms.extend(histograms)
            all_labels.extend(batch_labels)

    print("predicting the values")
    y_pred = classifier.predict(all_histograms)

    output = compute_metrics(all_labels, y_pred, map_classes=map_classes)

    aux_auc_per_class = {}
    for cls_name, cls_idx in map_classes.items():
        aux_auc_per_class[cls_name] = output["auc_scores"][cls_idx]
    print("Metrics on Phase[Eval]:")
    print(f" - Precision: {output['precision']}")
    print(f" - Recall: {output['recall']}")
    print(f" - F1 Score: {output['f1_score']}")
    print(f" - AUC: {output['auc_average']}")
    print(f" - AUC per class: {aux_auc_per_class}")

    return output


def train(
    dataset: List[Tuple[Type[Image.Image], int]],
    bovw: Type[BOVW],
    map_classes: dict,
    batch_size: int = 50,
):
    # Phase 1: Fit Codebook progressively
    print("Phase [Training]: Fitting Codebook progressively")

    for i in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Fitting Codebook"):
        batch = dataset[i : i + batch_size]
        batch_descriptors = []

        for image, label, img_path in batch:
            _, descriptors = bovw._extract_features_dense(
                image=np.array(image), image_path=Path(img_path)
            )
            if descriptors is not None and len(descriptors) > 0:
                batch_descriptors.append(descriptors)

        if batch_descriptors:
            bovw._update_fit_codebook(descriptors=batch_descriptors)

    # Phase 2: Compute Histograms for Classifier
    print("Phase [Training]: Computing Histograms")
    all_histograms = []
    all_labels = []

    for i in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Computing Histograms"):
        batch = dataset[i : i + batch_size]
        batch_descriptors = []
        batch_labels = []

        for image, label, img_path in batch:
            _, descriptors = bovw._extract_features_dense(
                image=np.array(image), image_path=Path(img_path)
            )
            if descriptors is not None and len(descriptors) > 0:
                batch_descriptors.append(descriptors)
                batch_labels.append(label)

        if batch_descriptors:
            histograms = extract_bovw_histograms(
                descriptors=batch_descriptors, bovw=bovw
            )
            all_histograms.extend(histograms)
            all_labels.extend(batch_labels)

    print("Fitting the classifier")
    classifier = LogisticRegression(class_weight="balanced").fit(
        all_histograms, all_labels
    )

    output = compute_metrics(
        all_labels, classifier.predict(all_histograms), map_classes=map_classes
    )

    auc_per_class = {}
    for cls_name, cls_idx in map_classes.items():
        auc_per_class[cls_name] = output["auc_scores"][cls_idx]
    print("Metrics on Phase[Train]:")
    print(f" - Precision: {output['precision']}")
    print(f" - Recall: {output['recall']}")
    print(f" - F1 Score: {output['f1_score']}")
    print(f" - AUC: {output['auc_average']}")
    print(f" - AUC per class: {auc_per_class}")

    return bovw, classifier


def Dataset(
    ImageFolder: str = "data/MIT_split/train",
    map_classes: dict = {},
) -> List[Tuple[Type[Image.Image], int]]:
    """
    Expected Structure:

        ImageFolder/<cls label>/xxx1.png
        ImageFolder/<cls label>/xxx2.png
        ImageFolder/<cls label>/xxx3.png
        ...

        Example:
            ImageFolder/cat/123.png
            ImageFolder/cat/nsdf3.png
            ImageFolder/cat/[...]/asd932_.png

    """
    if not map_classes:
        map_classes = {
            clsi: idx for idx, clsi in enumerate(natsorted(os.listdir(ImageFolder)))
        }

    dataset: List[Tuple] = []

    for idx, cls_folder in enumerate(natsorted(os.listdir(ImageFolder))):
        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path + "/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")

            dataset.append((img_pil, map_classes[cls_folder], img))

    return dataset, map_classes


if __name__ == "__main__":
    # /home/cboned/data/Master/MIT_split
    data_train, map_classes = Dataset(ImageFolder="./data/train")
    data_test, _ = Dataset(ImageFolder="./data/val", map_classes=map_classes)

    codebook_sizes = [50, 100, 200, 400, 800, 1600, 3200]

    for codebook_size in codebook_sizes:
        print(f"\n\nTraining with codebook size: {codebook_size}")
        bovw = BOVW(detector_type="SIFT", codebook_size=codebook_size)

        bovw, classifier = train(dataset=data_train, bovw=bovw, batch_size=100)

        test(dataset=data_test, bovw=bovw, classifier=classifier, batch_size=100)
