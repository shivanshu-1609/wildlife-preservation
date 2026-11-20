"""Generate an eight-class competition submission from a trained checkpoint."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import CLASSES, TestImages, image_transform
from model import load_checkpoint, select_device


@torch.inference_mode()
def predict(args):
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if args.output.exists():
        raise ValueError("The submission file already exists. Choose another --output.")
    device = select_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    classes = checkpoint["classes"]
    if len(classes) != 8 or set(classes) != set(CLASSES):
        raise ValueError(
            "Competition submissions require an eight-class model. "
            "The binary baseline has only two classes."
        )

    transform = image_transform(
        checkpoint["image_size"], checkpoint["mean"], checkpoint["std"]
    )
    dataset = TestImages(args.test_dir, args.features, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    batches = []
    for images in loader:
        logits = model(images.to(device))
        batches.append(logits.softmax(dim=1).cpu())
    probabilities = torch.cat(batches)

    if (
        not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not torch.allclose(
            probabilities.sum(dim=1), torch.ones(len(dataset)), atol=1e-5
        )
    ):
        raise ValueError("Model outputs are not valid class probabilities.")
    frame = pd.DataFrame(probabilities.numpy(), columns=classes)[list(CLASSES)]
    frame.insert(0, "id", dataset.frame.id.to_list())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} predictions to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/multiclass/model.pt")
    )
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument(
        "--features", type=Path, required=True, help="Path to test_features.csv"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    try:
        predict(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
