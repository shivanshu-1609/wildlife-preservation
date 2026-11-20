# Wildlife Conservation in Côte d'Ivoire

Camera-trap image classification with PyTorch, completed as part of
**WorldQuant University's Applied AI Lab: Deep Learning for Computer Vision**.
The project includes a binary dense baseline and an eight-class convolutional
neural network.

## Dataset

The [DrivenData Conser-vision competition](https://www.drivendata.org/competitions/87/co/)
uses camera-trap images from Taï National Park in Côte d'Ivoire. Data is provided
by the Wild Chimpanzee Foundation and Max Planck Institute for Evolutionary
Anthropology.

Classes: `antelope_duiker`, `bird`, `blank`, `civet_genet`, `hog`, `leopard`,
`monkey_prosimian`, and `rodent`.

Obtain the images from the [official data page](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/data/)
or an authorized WQU course download. Dataset files and trained weights are
excluded from the repository.

```text
data_p1/
├── data_multiclass/
│   ├── train/<class>/<image>.jpg
│   └── test/
│       ├── test_features.csv
│       └── test_features/<image>.jpg
└── data_binary/train/
    ├── blank/
    └── hog/
```

Official CSV downloads can be converted to class folders with:

```bash
python data.py --raw-dir data/raw --output data/multiclass/train
```

`data/raw/` must contain `train_features.csv`, `train_labels.csv`, and extracted
images at the paths specified in the metadata.

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
```

Activate `.venv`: `.\.venv\Scripts\Activate.ps1` on Windows
PowerShell, or `source .venv/bin/activate` on macOS/Linux.

```bash
python -m pip install -r requirements.txt
```

## Usage

Train the multiclass CNN:

```bash
python train.py --train-dir data_p1/data_multiclass/train
```

Train the blank-versus-hog baseline:

```bash
python train.py --train-dir data_p1/data_binary/train --model mlp --output outputs/binary
```

Generate an eight-class submission:

```bash
python predict.py --test-dir data_p1/data_multiclass/test --features data_p1/data_multiclass/test/test_features.csv
```

Training saves `model.pt`, `history.csv`, and `confusion_matrix.csv` in
`outputs/multiclass/`. Prediction writes class probabilities to
`outputs/submission.csv`. Use a new `--output` directory for each experiment.

For official downloads, use `--train-dir data/multiclass/train` when training
and `--test-dir data/raw --features data/raw/test_features.csv` when predicting.

## Model

The CNN uses three convolution, ReLU, and max-pooling blocks with 16, 32, and 64
channels, followed by a 500-unit dense layer, dropout, and eight output logits.

Images are converted to RGB and resized to 224 × 224. The default training setup
uses a stratified 80/20 split, training-only normalization, Adam with learning
rate 0.001, cross-entropy loss, eight epochs, and seed 42. The checkpoint with the
lowest validation loss is retained. CUDA, MPS, or CPU is selected automatically.

Add `--sites-csv data/raw/train_features.csv` to hold out complete camera sites
instead of individual images. This better reflects the competition's unseen-site
evaluation. Class imbalance and shared camera backgrounds can affect accuracy.

## Results

| Recorded epoch | Validation loss | Validation accuracy |
| --- | ---: | ---: |
| Best: 7 | 0.52 | 84% |
| Final: 8 | 0.58 | 82% |

These values are recorded outputs from the original course notebook. They have
not been reproduced with these scripts and are not a verified competition score.
The competition evaluates multiclass log loss.

## References and license

- [WorldQuant University](https://www.wqu.edu/) — course and original learning material.
- [DrivenData problem description](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/page/483/) — labels, metadata, evaluation, and submission format.

The Python scripts and repository documentation use the [MIT license](LICENSE).
Original WQU course notebooks retain their © 2024 WQU and
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) notices.
Dataset use remains subject to the competition's terms.
