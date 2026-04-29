import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.model import EEGNet
from src.config import DEVICE

def plot_A1_result():
    METRICS_PATH_A01 = 'metrics/eegnet_A01_metrics.pt'
    a01_metrics = torch.load(METRICS_PATH_A01,weights_only=False)
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(a01_metrics['train_losses'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training Loss — Subject A01')
    axes[0].grid(True, alpha=0.3)

    eval_epochs = list(range(1, len(a01_metrics['test_accs'])*10 + 1, 10))
    if len(eval_epochs) > len(a01_metrics['test_accs']):
        eval_epochs = eval_epochs[:len(a01_metrics['test_accs'])]
    axes[1].plot(eval_epochs, a01_metrics['test_accs'], marker='o', markersize=3, label='Accuracy')
    axes[1].plot(eval_epochs, a01_metrics['test_kappas'], marker='s', markersize=3, label='Kappa')
    axes[1].axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Test Performance — Subject A01')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_A1_confusion_matrix():
    METRICS_PATH_A01 = 'metrics/eegnet_A01_metrics.pt'
    a01_metrics = torch.load(METRICS_PATH_A01,weights_only=False)
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    cm = confusion_matrix(a01_metrics['final_labels'], a01_metrics['final_preds'])
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f"EEGNet — Subject A01\nAcc: {a01_metrics['final_acc']:.3f}, Kappa: {a01_metrics['final_kappa']:.3f}")
    plt.tight_layout()
    plt.show()

def show_summary():
    METRICS_PATH_ALL = 'metrics/eegnet_all_subjects.pt'
    all_results = torch.load(METRICS_PATH_ALL,weights_only=False)
    # Summary statistics
    accs   = [all_results[s]['accuracy'] for s in sorted(all_results.keys())]
    kappas = [all_results[s]['kappa']    for s in sorted(all_results.keys())]

    print(f"{'Subject':<10} {'Accuracy':>10} {'Kappa':>10}")
    print('-' * 32)
    for subj in sorted(all_results.keys()):
        r = all_results[subj]
        print(f"{subj:<10} {r['accuracy']:>10.4f} {r['kappa']:>10.4f}")
    print('-' * 32)
    print(f"{'Mean':<10} {np.mean(accs):>10.4f} {np.mean(kappas):>10.4f}")
    print(f"{'Std':<10} {np.std(accs):>10.4f} {np.std(kappas):>10.4f}")
    print(f"\nAccuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Kappa    : {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")

def plot_bar_chart():
    # Per-subject bar chart
    METRICS_PATH_ALL = 'metrics/eegnet_all_subjects.pt'
    all_results = torch.load(METRICS_PATH_ALL,weights_only=False)
    subjects = sorted(all_results.keys())
    accs_plot   = [all_results[s]['accuracy'] for s in subjects]
    kappas_plot = [all_results[s]['kappa']    for s in subjects]

    x = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, accs_plot,   width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, kappas_plot, width, label='Kappa',    color='coral')

    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
    ax.axhline(y=np.mean(accs_plot), color='steelblue', linestyle=':', alpha=0.7, label=f'Mean acc ({np.mean(accs_plot):.3f})')

    ax.set_xlabel('Subject')
    ax.set_ylabel('Score')
    ax.set_title('EEGNet — Per-Subject Performance on Dataset 2a')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_filters():
    # Load the trained A01 model
    MODEL_PATH_A01   = 'models/eegnet_A01.pt'
    model_viz = EEGNet().to(DEVICE)
    model_viz.load_state_dict(torch.load(MODEL_PATH_A01, map_location=DEVICE))
    model_viz.eval()

    # Extract temporal conv weights
    # Shape: (F1, 1, 1, kern_len) = (8, 1, 1, 125)
    temporal_weights = model_viz.block1[0].weight.detach().cpu().numpy()
    print(f"Temporal conv weight shape: {temporal_weights.shape}")

    fs = 250  # Hz
    F1 = temporal_weights.shape[0]

    fig, axes = plt.subplots(F1, 2, figsize=(12, 2 * F1))

    for i in range(F1):
        # Filter in time domain
        filt = temporal_weights[i, 0, 0, :]  # shape: (kern_len,)
        
        # Time domain
        t = np.arange(len(filt)) / fs
        axes[i, 0].plot(t, filt, color='steelblue', linewidth=0.8)
        axes[i, 0].set_ylabel(f'Filter {i}')
        axes[i, 0].set_xlim(0, t[-1])
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title('Time Domain')
        if i == F1 - 1:
            axes[i, 0].set_xlabel('Time (s)')
        
        # Frequency response
        freqs = np.fft.rfftfreq(len(filt), d=1/fs)
        fft_mag = np.abs(np.fft.rfft(filt))
        
        axes[i, 1].plot(freqs, fft_mag, color='coral', linewidth=0.8)
        axes[i, 1].axvspan(8, 13, alpha=0.15, color='green', label='Mu' if i == 0 else '')
        axes[i, 1].axvspan(13, 30, alpha=0.15, color='orange', label='Beta' if i == 0 else '')
        axes[i, 1].set_xlim(0, 60)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title('Frequency Response')
            axes[i, 1].legend(fontsize=8)
        if i == F1 - 1:
            axes[i, 1].set_xlabel('Frequency (Hz)')

    plt.suptitle('EEGNet Learned Temporal Filters — Subject A01', y=1.01, fontsize=14)
    plt.tight_layout()
    plt.show()