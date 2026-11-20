"""Load camera-trap images, split the data, and prepare official CSV downloads."""

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CLASSES = (
    "antelope_duiker",
    "bird",
    "blank",
    "civet_genet",
    "hog",
    "leopard",
    "monkey_prosimian",
    "rodent",
)
BINARY_CLASSES = ("blank", "hog")


class ToRGB:
    """Convert images to RGB."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


def image_transform(image_size: int, mean=None, std=None) -> transforms.Compose:
    steps = [
        ToRGB(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if mean is not None:
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def image_folder(root: Path, image_size: int) -> datasets.ImageFolder:
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {root}. See README.md.")
    dataset = datasets.ImageFolder(root, transform=image_transform(image_size))
    if tuple(dataset.classes) not in (CLASSES, BINARY_CLASSES):
        raise ValueError("Use all eight class folders, or only blank and hog.")
    return dataset


def split_indices(dataset, seed: int, sites_csv: Path | None = None):
    """Hold out about 20% of each class, or 20% of camera sites when available."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    if sites_csv is None:
        for index, target in enumerate(dataset.targets):
            groups[target].append(index)
        training, validation = [], []
        for indices in groups.values():
            if len(indices) < 2:
                raise ValueError("Each class needs at least two images.")
            rng.shuffle(indices)
            count = min(len(indices) - 1, max(1, round(len(indices) * 0.2)))
            validation.extend(indices[:count])
            training.extend(indices[count:])
    else:
        frame = pd.read_csv(sites_csv, dtype=str)
        if not {"id", "site"}.issubset(frame.columns):
            raise ValueError("Site metadata must have id and site columns.")
        if frame[["id", "site"]].isna().any().any() or frame.id.duplicated().any():
            raise ValueError("Site metadata needs unique IDs and no missing sites.")
        sites = frame.set_index("id").site.to_dict()
        for index, (filename, _) in enumerate(dataset.samples):
            image_id = Path(filename).stem
            if image_id not in sites:
                raise ValueError(f"Missing site for {image_id}.")
            groups[sites[image_id]].append(index)
        keys = sorted(groups)
        if len(keys) < 2:
            raise ValueError("At least two camera sites are required.")
        rng.shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * 0.2)))
        validation = [i for key in keys[:count] for i in groups[key]]
        training = [i for key in keys[count:] for i in groups[key]]
    expected = set(range(len(dataset.classes)))
    if any(
        {dataset.targets[i] for i in part} != expected
        for part in (training, validation)
    ):
        raise ValueError("A split lacks a class. Try another --seed or more data.")
    return sorted(training), sorted(validation)


def channel_statistics(dataset, indices: list[int], batch_size: int):
    """Calculate pixel-weighted RGB statistics using training images only."""
    sums = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    pixels = 0
    for images, _ in DataLoader(Subset(dataset, indices), batch_size=batch_size):
        images = images.double()
        sums += images.sum(dim=(0, 2, 3))
        squares += images.square().sum(dim=(0, 2, 3))
        pixels += images.shape[0] * images.shape[2] * images.shape[3]
    mean = sums / pixels
    std = (squares / pixels - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    return mean.tolist(), std.tolist()


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image path leaves its data directory: {relative}")
    return path


class TestImages(Dataset):
    """Keep image order identical to the rows in test_features.csv."""

    def __init__(self, root: Path, features: Path, transform) -> None:
        self.frame = pd.read_csv(features, dtype={"id": str, "filepath": str})
        if not {"id", "filepath"}.issubset(self.frame.columns) or self.frame.empty:
            raise ValueError("Test metadata needs nonempty id and filepath columns.")
        if (
            self.frame[["id", "filepath"]].isna().any().any()
            or self.frame.id.duplicated().any()
        ):
            raise ValueError(
                "Test IDs must be unique; IDs and paths cannot be missing."
            )
        self.paths = [contained_path(root, path) for path in self.frame.filepath]
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Test image not found: {path}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


def prepare_data(raw_dir: Path, output: Path, binary: bool = False) -> None:
    """Copy official one-hot labeled downloads into one folder per class."""
    features = pd.read_csv(raw_dir / "train_features.csv", dtype={"id": str})
    labels = pd.read_csv(raw_dir / "train_labels.csv", dtype={"id": str})
    if not {"id", "filepath"}.issubset(features.columns):
        raise ValueError("train_features.csv needs id and filepath columns.")
    if not {"id", *CLASSES}.issubset(labels.columns):
        raise ValueError("train_labels.csv needs id and all eight class columns.")
    for frame in (features, labels):
        if frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError("Training IDs must be unique and nonempty.")
    if set(features.id) != set(labels.id) or features.empty:
        raise ValueError("Feature IDs and label IDs must match and cannot be empty.")
    targets = labels[list(CLASSES)].apply(pd.to_numeric, errors="raise")
    if not targets.isin([0, 1]).all().all() or not (targets.sum(axis=1) == 1).all():
        raise ValueError("Labels must be one-hot: exactly one 1 per row.")
    labels = labels.assign(class_name=targets.idxmax(axis=1))
    rows = features.merge(labels[["id", "class_name"]], on="id", validate="one_to_one")
    selected = BINARY_CLASSES if binary else CLASSES
    rows = rows[rows.class_name.isin(selected)]
    if set(rows.class_name) != set(selected):
        raise ValueError("The download does not contain every requested class.")
    copies = []
    for row in rows.itertuples(index=False):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row.id):
            raise ValueError(f"Image ID is not a safe filename: {row.id}")
        if not isinstance(row.filepath, str):
            raise ValueError(f"Missing filepath for {row.id}.")
        source = contained_path(raw_dir, row.filepath)
        if not source.is_file():
            raise FileNotFoundError(f"Extract the training image: {source}")
        if source.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        copies.append(
            (source, output / row.class_name / f"{row.id}{source.suffix.lower()}")
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory.")
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"Prepared {len(copies)} images at {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--train-dir", type=Path, help="Print counts from class folders"
    )
    inputs.add_argument("--raw-dir", type=Path, help="Prepare official CSV downloads")
    parser.add_argument("--output", type=Path, help="Destination for prepared images")
    parser.add_argument(
        "--binary", action="store_true", help="Prepare only blank and hog"
    )
    args = parser.parse_args()
    try:
        if args.raw_dir:
            if args.output is None:
                parser.error("--raw-dir requires --output")
            prepare_data(args.raw_dir, args.output, args.binary)
        else:
            dataset = image_folder(args.train_dir, 224)
            counts = Counter(dataset.targets)
            for index, name in enumerate(dataset.classes):
                print(f"{name:20} {counts[index]:6,}")
            print(f"{'Total':20} {len(dataset):6,}")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
