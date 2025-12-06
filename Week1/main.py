import glob
import os
from pathlib import Path
from typing import List, Literal, Tuple, Type

import numpy as np
import tqdm
from bovw import BOVW
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array(
        [
            bovw._compute_codebook_descriptor(
                descriptors=descriptor, kmeans=bovw.codebook_algo
            )
            for descriptor in descriptors
        ]
    )


def test(
    dataset: List[Tuple[Type[Image.Image], int]],
    bovw: Type[BOVW],
    classifier: Type[object],
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

    print(
        "Accuracy on Phase[Test]:",
        accuracy_score(y_true=all_labels, y_pred=y_pred),
    )


def train(
    dataset: List[Tuple[Type[Image.Image], int]], bovw: Type[BOVW], batch_size: int = 50
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

    print(
        "Accuracy on Phase[Train]:",
        accuracy_score(y_true=all_labels, y_pred=classifier.predict(all_histograms)),
    )

    return bovw, classifier


def Dataset(
    ImageFolder: str = "data/MIT_split/train",
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

    map_classes = {clsi: idx for idx, clsi in enumerate(os.listdir(ImageFolder))}

    dataset: List[Tuple] = []

    for idx, cls_folder in enumerate(os.listdir(ImageFolder)):
        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path + "/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")

            dataset.append((img_pil, map_classes[cls_folder], img))

    return dataset


if __name__ == "__main__":
    # /home/cboned/data/Master/MIT_split
    data_train = Dataset(ImageFolder="./data/train")
    data_test = Dataset(ImageFolder="./data/val")

    bovw = BOVW(detector_type="SIFT")

    bovw, classifier = train(dataset=data_train, bovw=bovw)

    test(dataset=data_test, bovw=bovw, classifier=classifier)
