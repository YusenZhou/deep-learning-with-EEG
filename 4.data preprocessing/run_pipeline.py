import numpy as np
import mne
from braindecode.datasets import HGD
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')


# Dataset constants
FS = 500           # Sampling rate (Hz)
N_CHANNELS = 128   # EEG channels
N_CLASSES = 4      # left hand, right hand, feet, rest
CLASS_NAMES = ['left_hand', 'right_hand', 'feet', 'rest']

def load_subject_epochs(subject_id):
    """
    Load and epoch one subject's data from the High-Gamma Dataset (HGD)
    via braindecode/MOABB.
    
    Args:
        subject_id: int, 1–14
    
    Returns:
        epochs_data: np.ndarray, shape (n_trials, 128, n_times)
        labels:      np.ndarray, shape (n_trials,), values in {0, 1, 2, 3}
                     0=left hand, 1=right hand, 2=feet, 3=rest
    """
    dataset = HGD(subject_ids=[subject_id])

    all_epochs_data = []
    all_labels = []

    for ds in dataset.datasets:
        raw = ds.raw.copy()

        raw.pick_types(eeg=True)
        raw.filter(4.0, 38.0, fir_design='firwin')
        raw.set_eeg_reference('average', projection=False)

        events, event_id_full = mne.events_from_annotations(raw)

        target_events = {}
        label_map = {}
        for key, val in event_id_full.items():
            key_lower = key.lower().replace(' ', '_')
            if 'left' in key_lower and 'hand' in key_lower:
                target_events[key] = val
                label_map[val] = 0
            elif 'right' in key_lower and 'hand' in key_lower:
                target_events[key] = val
                label_map[val] = 1
            elif 'feet' in key_lower or 'foot' in key_lower:
                target_events[key] = val
                label_map[val] = 2
            elif 'rest' in key_lower:
                target_events[key] = val
                label_map[val] = 3

        if not target_events:
            continue

        epochs = mne.Epochs(
            raw, events, event_id=target_events,
            tmin=0.5, tmax=2.5,
            baseline=None,
            preload=True,
            proj=False
        )

        epochs_data = epochs.get_data()
        event_codes = epochs.events[:, 2]
        labels = np.array([label_map[c] for c in event_codes])

        all_epochs_data.append(epochs_data)
        all_labels.append(labels)

    epochs_data = np.concatenate(all_epochs_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Baseline correction: subtract mean of first 250 samples (0.5s at 500 Hz)
    baseline = epochs_data[:, :, :250].mean(axis=2, keepdims=True)
    epochs_data = epochs_data - baseline

    return epochs_data, labels

def detect_artifacts(epochs_data, threshold_uv=100.0):
    """
    Detect artifact-contaminated trials using peak-to-peak thresholding.
    
    Args:
        epochs_data:  np.ndarray, shape (n_trials, n_channels, n_times)
        threshold_uv: float, PTP threshold in microvolts
    
    Returns:
        clean_mask: np.ndarray, shape (n_trials,), dtype=bool
                    True = clean, False = artifact
    """
    # YOUR CODE HERE
    # mask = []
    # n_trials, n_channels, n_times = epochs_data.shape
    # for i in range(n_trials):
    #     clean = True
    #     for j in range(n_channels):
    #         if np.max(epochs_data[i,j,:]) - np.min(epochs_data[i,j,:]) >threshold_uv:
    #             clean = False
    #             break
    #     mask.append(clean)
    # return np.array(mask)
    ptp_per_channel = np.max(epochs_data, axis=2) - np.min(epochs_data, axis=2)
    max_ptp_per_trial = np.max(ptp_per_channel, axis=1) *1e6
    clean_mask = max_ptp_per_trial <= threshold_uv
    return clean_mask

def reject_artifacts(epochs_data, labels, threshold_uv=100.0):
    """
    Reject artifact trials and their corresponding labels.
    
    Args:
        epochs_data:  np.ndarray, shape (n_trials, n_channels, n_times)
        labels:       np.ndarray, shape (n_trials,)
        threshold_uv: float, PTP threshold in microvolts
    
    Returns:
        clean_data:   np.ndarray, shape (n_clean, n_channels, n_times)
        clean_labels: np.ndarray, shape (n_clean,)
        n_rejected:   int
    """
    # YOUR CODE HERE
    mask = detect_artifacts(epochs_data,threshold_uv)
    clean_data = epochs_data[mask]
    clean_labels = labels[mask]
    n_rejected = (~mask).sum()
    return clean_data,clean_labels,n_rejected

def split_epochs(epochs_data, labels, test_size=0.2, random_state=42):
    """
    Stratified train/test split at the epoch level.
    
    Args:
        epochs_data:  np.ndarray, shape (n_trials, n_channels, n_times)
        labels:       np.ndarray, shape (n_trials,)
        test_size:    float
        random_state: int
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # YOUR CODE HERE
    sss = StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
    train_idx,test_idx = next(sss.split(epochs_data,labels))
    X_train,X_test = epochs_data[train_idx],epochs_data[test_idx]
    y_train,y_test = labels[train_idx],labels[test_idx]
    return X_train, X_test, y_train, y_test


def sliding_window(X, y, window_size=500, step_size=250):
    """
    Extract overlapping sliding windows from epochs.
    
    Args:
        X:           np.ndarray, shape (n_epochs, n_channels, n_times)
        y:           np.ndarray, shape (n_epochs,)
        window_size: int, samples per window
        step_size:   int, step between windows
    
    Returns:
        X_windows: np.ndarray, shape (n_windows, n_channels, window_size)
        y_windows: np.ndarray, shape (n_windows,)
    """
    # YOUR CODE HERE
    n_epochs,n_channels,n_times = X.shape
    windows = []
    labels = []

    for i in range(n_epochs):
        for start in range(0,n_times-window_size+1,step_size):
            windows.append(X[i,:,start:start+window_size])
            labels.append(y[i])
    windows,labels = map(np.array,[windows,labels])
    return windows,labels

def compute_normalization_stats(X_train):
    """
    Compute per-channel mean and std from training data.
    
    Args:
        X_train: np.ndarray, shape (n_windows, n_channels, n_times)
    
    Returns:
        mu:    np.ndarray, shape (n_channels,)
        sigma: np.ndarray, shape (n_channels,)
    """
    # YOUR CODE HERE
    mu = np.mean(X_train,axis=(0,2))
    sigma = np.std(X_train,axis =(0,2))
    return mu,sigma


def normalize(X, mu, sigma):
    """
    Apply z-score normalization using precomputed statistics.
    
    Args:
        X:     np.ndarray, shape (n_windows, n_channels, n_times)
        mu:    np.ndarray, shape (n_channels,)
        sigma: np.ndarray, shape (n_channels,)
    
    Returns:
        X_norm: np.ndarray, same shape as X
    """
    # YOUR CODE HERE
    mu = mu.reshape(1,-1,1)
    sigma = sigma.reshape(1,-1,1)
    X = (X-mu)/sigma
    return X

def run_pipeline(subject_id, artifact_threshold=100.0,
                 test_size=0.2, window_size=500, step_size=250, random_state=42):
    """
    Complete preprocessing pipeline: load → reject → split → window → normalize.
    
    Args:
        subject_id:          int, 1–14 (HGD subject)
        artifact_threshold:  float, PTP threshold in µV
        test_size:           float, fraction for test set
        window_size:         int, samples per window
        step_size:           int, step between windows
        random_state:        int, random seed
    
    Returns:
        dict with 'X_train', 'X_test', 'y_train', 'y_test',
             'n_rejected', 'mu', 'sigma'
    """
    # YOUR CODE HERE
    data, labels = load_subject_epochs(subject_id)
    clean_data,clean_labels,n_rejected = reject_artifacts(data,labels,threshold_uv=artifact_threshold)
    X_train, X_test, y_train, y_test = split_epochs(clean_data,clean_labels,test_size=test_size,random_state=random_state)
    train_windows, train_labels = sliding_window(X_train,y_train,window_size=window_size,step_size=step_size)
    test_windows,test_labels = sliding_window(X_test,y_test,window_size=window_size,step_size=step_size)
    mu,sigma = compute_normalization_stats(train_windows)
    X_fin_train = normalize(train_windows,mu,sigma)
    X_fin_test = normalize(test_windows,mu,sigma)
    return {'X_train':X_fin_train, 'X_test':X_fin_test, 'y_train':train_labels, 'y_test':test_labels,
             'n_rejected':n_rejected, 'mu':mu, 'sigma':sigma}
    

