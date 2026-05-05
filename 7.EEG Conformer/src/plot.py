import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_results_A1():
    # Plot training curves
    METRICS_PATH_A01 = 'metrics/conformer_A01_metrics.pt'
    a01_metrics = torch.load(METRICS_PATH_A01)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(a01_metrics['train_losses'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')

    eval_epochs = list(range(1, len(a01_metrics['test_accs'])*10 + 1, 10))
    if len(eval_epochs) > len(a01_metrics['test_accs']):
        eval_epochs = eval_epochs[:len(a01_metrics['test_accs'])]

    axes[1].plot(eval_epochs, a01_metrics['test_accs'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Test Accuracy')
    axes[1].axhline(y=0.25, color='r', linestyle='--', label='Chance (25%)')
    axes[1].legend()

    axes[2].plot(eval_epochs, a01_metrics['test_kappas'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Kappa')
    axes[2].set_title('Cohen\'s Kappa')

    plt.suptitle('EEG Conformer — Subject A01 Training', fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_A1_confusion_matrix():
    # Confusion matrix for A01
    METRICS_PATH_A01 = 'metrics/conformer_A01_metrics.pt'
    a01_metrics = torch.load(METRICS_PATH_A01)
    _preds = a01_metrics['final_preds']
    _labels = a01_metrics['final_labels']
    cm = confusion_matrix(_labels, _preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Left', 'Right', 'Feet', 'Tongue'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'EEG Conformer — A01 (acc={a01_metrics["final_accuracy"]:.3f})')
    plt.tight_layout()
    plt.show()

def plot_bar_chart():
    METRICS_PATH_ALL = 'metrics/conformer_all_subjects.pt'
    all_results = torch.load(METRICS_PATH_ALL)
    # Per-subject bar chart
    subjects = sorted(all_results.keys())
    accs = [all_results[s]['accuracy'] for s in subjects]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(subjects))

    bars = ax.bar(x, accs, color='steelblue', alpha=0.8)
    ax.axhline(y=0.25, color='r', linestyle='--', label='Chance (25%)', alpha=0.7)
    ax.axhline(y=np.mean(accs), color='green', linestyle='--',
            label=f'Mean ({np.mean(accs):.3f})', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel('Accuracy')
    ax.set_title('EEG Conformer — Per-Subject Accuracy')
    ax.legend()
    ax.set_ylim(0, 1)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

