# EEG Conformer — Motor Imagery Classification

Notebook 6 of the EEG BCI series. Implements a CNN-Transformer hybrid classifier for 4-class motor imagery decoding on BCI Competition IV Dataset 2a (9 subjects, 22 EEG channels, 250 Hz).

---

## Results

| Model | Accuracy | Cohen's κ |
|---|---|---|
| CSP + SVM | 0.556 ± 0.139 | 0.408 ± 0.186 |
| EEGNet | 0.564 ± 0.134 | 0.419 ± 0.179 |
| EEG Conformer | 0.555 ± 0.115 | 0.406 ± 0.153 |

All results are mean ± std across 9 subjects, subject-dependent evaluation.

---

## Architecture

```
Input (B, 1, 22, 250)
  │
  ├── PatchEmbedding (CNN front-end)
  │     ├── Temporal Conv2d  (1, kern_len=25) → (B, 40, 22, 250)
  │     ├── BatchNorm + ELU
  │     ├── Depthwise Conv2d (22, 1), groups=40 → (B, 40, 1, 250)
  │     ├── BatchNorm + ELU + Dropout
  │     ├── Squeeze → (B, 40, 250)
  │     ├── AvgPool1d (pool_size=75, stride=15) → (B, 40, 12)
  │     └── Transpose → (B, 12, 40)  [12 tokens, d_model=40]
  │
  ├── Learnable Positional Embedding  (1, 12, 40)
  │
  ├── TransformerBlock × 6
  │     ├── Pre-LN Multi-Head Attention (10 heads, d_k=4)
  │     └── Pre-LN FFN (40 → 120 → 40, GELU)
  │
  ├── LayerNorm
  ├── Mean Pool over T'=12 → (B, 40)
  └── Linear → (B, 4)
```

**Total parameters:** ~104,000

Each of the 12 tokens represents a 300ms window of EEG, strided by 60ms. The CNN front-end handles local spatial-temporal filtering; the Transformer handles global temporal dependencies via self-attention.

---

## Project Structure

```
6.EEG Conformer/
├── eeg_conformer.ipynb   # Main notebook
├── run.py                # CLI entry point
├── src/
│   ├── train.py          # train_on_A1(), train_all_subjects()
│   ├── plot.py           # plot_results_A1(), plot_A1_confusion_matrix(), plot_bar_chart()
│   └── ablation.py       # ablation_heads(), ablation_kern_len(), ablation_layers()
├── metrics/              # Saved training metrics (.pt files)
└── models/               # Saved model weights (.pt files)
```

---

## Usage

```bash
# See all available commands
python run.py

# Train on subject A01 only (development / quick check)
python run.py train_on_A1

# Train and evaluate on all 9 subjects
python run.py train_all_subjects

# Plots (run after training)
python run.py plot_results_A1
python run.py plot_A1_confusion_matrix
python run.py plot_bar_chart

# Ablation studies
python run.py ablation_heads
python run.py ablation_kern_len
python run.py ablation_layers
```

---

## Data

BCI Competition IV Dataset 2a. Place files at:

```
../mne_data/bci_iv_2a/
├── A01T.gdf  A01E.gdf  A01T.mat  A01E.mat
├── ...
└── A09T.gdf  A09E.gdf  A09T.mat  A09E.mat
```

Preprocessing is handled by `run_pipeline()` from `../4.data preprocessing/run_pipeline.py`. The Conformer notebook does not reimplement any preprocessing.

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| F1 | 40 | Temporal filters (vs 8 in EEGNet) |
| D | 1 | Depth multiplier |
| kern_len | 25 | ~100ms temporal kernel (vs 125ms in EEGNet) |
| pool_size | 75 | ~300ms pooling window |
| pool_stride | 15 | ~60ms stride → 12 tokens |
| n_heads | 10 | d_k = 4 per head |
| n_layers | 6 | Transformer encoder depth |
| ff_ratio | 3 | FFN width: 40 → 120 → 40 |
| dropout | 0.5 | Heavy regularization for small datasets |
| optimizer | AdamW | |
| scheduler | CosineAnnealingLR | |

---

## Compute

Training all 9 subjects takes approximately 40–60 minutes on Apple Silicon (MPS). Google Colab with a T4 GPU is recommended for faster iteration.

Trained weights and metrics are cached in `models/` and `metrics/` — re-running a cell loads from cache instead of retraining.

---
