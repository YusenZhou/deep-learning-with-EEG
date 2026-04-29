# EEGNet — Motor Imagery BCI Classifier

EEGNet implementation for 4-class motor imagery classification on BCI Competition IV Dataset 2a. Part of a larger EEG/BCI learning series covering signal processing, traditional ML (CSP+LDA/SVM), and deep learning approaches.

---

## What this does

Trains a compact CNN (EEGNet) to classify EEG signals into 4 motor imagery classes — left hand, right hand, feet, and tongue — across all 9 subjects of Dataset 2a. Includes ablation experiments on key architectural hyperparameters.

---

## Project structure

```
5.EEGNet/
├── src/
│   ├── config.py          # Device, hyperparameters (LR, N_EPOCHS, BATCH_SIZE)
│   ├── data.py            # EEGDataset, DataLoader construction (default: Subject A01)
│   ├── model.py           # EEGNet architecture
│   ├── train.py           # train_one_epoch(), evaluate()
│   ├── train_models.py    # train_on_A1(), train_on_all_data()
│   ├── plot.py            # All visualization functions
│   └── ablation.py        # Ablation experiments (D, dropout, kern_len)
├── metrics/               # Cached training results (.pt files)
├── models/                # Saved model weights (.pt files)
├── run.py                 # CLI entry point
└── README.md
```

Preprocessing is handled by the sibling folder:
```
4.data preprocessing/
└── run_pipeline.py        # Returns dict with X_train, X_test, y_train, y_test
```

---

## Dataset

**BCI Competition IV Dataset 2a**
- 9 subjects (A01–A09)
- 4 classes: left hand (0), right hand (1), feet (2), tongue (3)
- 22 EEG channels, 250 Hz sampling rate
- 288 training trials per subject

Expected location: `../mne_data/bci_iv_2a/`

---

## Architecture

EEGNet is a compact CNN designed specifically for EEG. The key design choices map directly onto EEG signal structure:

```
Input (B, 1, 22, 250)
  │
  ├── Block 1: Temporal Conv (learns frequency-selective filters)
  │           → Depthwise Conv (learns spatial filters over electrodes, like CSP)
  │           → BN → ELU → AvgPool → Dropout
  │
  ├── Block 2: Separable Conv (depthwise + pointwise)
  │           → BN → ELU → AvgPool → Dropout
  │
  └── Classifier: Flatten → Linear → (B, 4)
```

Default hyperparameters for Dataset 2a:

| Parameter | Value | Meaning |
|---|---|---|
| F1 | 8 | Temporal filters |
| D | 2 | Depth multiplier (spatial filters per temporal filter) |
| F2 | 16 | Pointwise filters (F1 × D) |
| kern_len | 125 | Temporal kernel = fs // 2 |
| pool1 / pool2 | 4 / 8 | Pooling factors |
| dropout | 0.5 | Dropout rate |
| epochs | 300 | Training epochs |
| lr | 1e-3 | Adam learning rate |

Total parameters: ~2,500 (intentionally compact for small EEG datasets).

---

## Usage

All commands run through `run.py`:

```bash
# See available commands
python run.py

# Train on Subject A01 (fast, ~2 min on MPS)
python run.py train_on_A1

# Train on all 9 subjects (slow, ~20 min on MPS)
python run.py train_on_all_data

# Plot training curves and test performance for A01
python run.py plot_A1_result

# Plot confusion matrix for A01
python run.py plot_A1_confusion_matrix

# Print per-subject accuracy and kappa table (requires train_on_all_data)
python run.py show_summary

# Per-subject bar chart (requires train_on_all_data)
python run.py plot_bar_chart

# Visualize learned temporal filters for A01
python run.py visualize_filters

# Ablation: depth multiplier D ∈ {1, 2, 4}
python run.py ablation_depth_multiplier

# Ablation: dropout ∈ {0.25, 0.5, 0.75}
python run.py ablation_dropout

# Ablation: temporal kernel length ∈ {32, 64, 125}
python run.py ablation_kernel_length
```

Results are cached in `metrics/` and `models/`. Re-running a command loads from cache instead of retraining. To force retraining, delete the relevant `.pt` files.

---

## Evaluation metrics

Every run reports both **accuracy** and **Cohen's kappa**. Kappa is the primary metric because it accounts for class imbalance after artifact rejection — accuracy alone is misleading on slightly unbalanced 4-class problems.

Kappa interpretation: 0 = chance, 0.4–0.6 = moderate, >0.6 = good for motor imagery BCI.

---

