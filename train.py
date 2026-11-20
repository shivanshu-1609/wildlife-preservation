"""Train a CNN or dense baseline and evaluate its best validation checkpoint."""

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import channel_statistics, image_folder, image_transform, split_indices
from model import build_model, load_checkpoint, select_device


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, device):
    """Compute validation log loss and accuracy."""
    model.eval()
    total_loss = 0.0
    predictions, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        total_loss += nn.functional.cross_entropy(
            logits, labels.to(device), reduction="sum"
        ).item()
        predictions.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    predictions, targets = torch.cat(predictions), torch.cat(targets)
    metrics = {
        "log_loss": total_loss / len(targets),
        "accuracy": (predictions == targets).float().mean().item(),
    }
    return metrics, predictions, targets


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 8:
        raise ValueError(
            "Epochs/batch size must be positive; image size must be at least 8."
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory for a new experiment.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = select_device()

    dataset = image_folder(args.train_dir, args.image_size)
    training, validation = split_indices(dataset, args.seed, args.sites_csv)
    print(f"Device: {device}; training: {len(training)}; validation: {len(validation)}")
    print("Calculating training RGB mean and standard deviation...")
    mean, std = channel_statistics(dataset, training, args.batch_size)
    dataset.transform = image_transform(args.image_size, mean, std)
    train_loader = DataLoader(
        Subset(dataset, training),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(Subset(dataset, validation), batch_size=args.batch_size)

    model = build_model(args.model, len(dataset.classes), args.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss = [], float("inf")

    for epoch in range(1, args.epochs + 1):
        training_loss = train_epoch(model, train_loader, optimizer, device)
        metrics, _, _ = evaluate(model, val_loader, device)
        if not math.isfinite(training_loss) or not math.isfinite(metrics["log_loss"]):
            raise ValueError("Non-finite loss. Try a smaller learning rate.")
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": metrics["log_loss"],
                "validation_accuracy": metrics["accuracy"],
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}: train loss={training_loss:.4f}, "
            f"validation loss={metrics['log_loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.2%}"
        )
        if metrics["log_loss"] < best_loss:
            best_loss = metrics["log_loss"]
            checkpoint = {
                "format_version": 2,
                "architecture": args.model,
                "image_size": args.image_size,
                "classes": dataset.classes,
                "mean": mean,
                "std": std,
                "epoch": epoch,
                "metrics": metrics,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "split_type": "site" if args.sites_csv else "stratified",
                "training_files": [
                    Path(dataset.samples[i][0]).relative_to(args.train_dir).as_posix()
                    for i in training
                ],
                "validation_files": [
                    Path(dataset.samples[i][0]).relative_to(args.train_dir).as_posix()
                    for i in validation
                ],
                "torch_version": str(torch.__version__),
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
            }
            torch.save(checkpoint, args.output / "model.pt")
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)

    model, checkpoint = load_checkpoint(args.output / "model.pt", device)
    metrics, predictions, targets = evaluate(model, val_loader, device)
    count = len(dataset.classes)
    matrix = torch.bincount(targets * count + predictions, minlength=count**2)
    matrix = matrix.reshape(count, count)
    pd.DataFrame(matrix.numpy(), index=dataset.classes, columns=dataset.classes).to_csv(
        args.output / "confusion_matrix.csv", index_label="true_class"
    )
    print(
        f"Best epoch: {checkpoint['epoch']}; "
        f"validation accuracy: {metrics['accuracy']:.2%}; "
        f"log loss: {metrics['log_loss']:.4f}\nSaved model.pt, history.csv, "
        f"and confusion_matrix.csv in {args.output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/multiclass"))
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sites-csv", type=Path, help="Optional id/site metadata for a site holdout"
    )
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
