"""Image classifiers returning logits for cross-entropy training."""

import torch
from torch import nn


class WildlifeCNN(nn.Module):
    """Three convolution blocks followed by a regularized classification head."""

    def __init__(self, num_classes: int, image_size: int = 224) -> None:
        super().__init__()
        blocks = []
        channels = (3, 16, 32, 64)
        for incoming, outgoing in zip(channels[:-1], channels[1:], strict=True):
            blocks.extend(
                [
                    nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
        self.features = nn.Sequential(*blocks)
        flattened = 64 * (image_size // 8) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flattened, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


class DenseBaseline(nn.Module):
    """Fully connected baseline for the blank-versus-hog experiment."""

    def __init__(self, num_classes: int, image_size: int = 224) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size**2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images)


def build_model(name: str, num_classes: int, image_size: int) -> nn.Module:
    """Construct the architecture recorded in a run or checkpoint."""
    if image_size < 8 or num_classes < 2:
        raise ValueError("image_size must be at least 8 and num_classes at least 2.")
    constructors = {"cnn": WildlifeCNN, "mlp": DenseBaseline}
    if name not in constructors:
        raise ValueError(f"Unknown architecture: {name}")
    return constructors[name](num_classes, image_size)


def select_device() -> torch.device:
    """Use a GPU when available, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device):
    """Restore the architecture and weights saved during training."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Use model.pt created by this version of train.py.")
    model = build_model(
        checkpoint["architecture"], len(checkpoint["classes"]), checkpoint["image_size"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint
