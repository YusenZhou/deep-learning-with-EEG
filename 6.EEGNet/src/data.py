import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

import sys
sys.path.insert(0, '../4.data preprocessing')
from run_pipeline import run_pipeline

DATA_DIR = '../mne_data/bci_iv_2a'

data = run_pipeline('A01', DATA_DIR)
X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']

# print(f"X_train: {X_train.shape}")
# print(f"X_test:  {X_test.shape}")
# print(f"y_train: {y_train.shape}, classes: {np.unique(y_train)}")
# print(f"y_test:  {y_test.shape},  classes: {np.unique(y_test)}")

class EEGDataset(Dataset):
    """
    Wraps EEG data for PyTorch DataLoader.
    
    Args:
        X: np.ndarray, shape (N, C, T) — preprocessed EEG windows
        y: np.ndarray, shape (N,) — integer class labels 0–3
    
    __getitem__ returns:
        x: FloatTensor, shape (1, C, T) — one trial with singleton dim
        label: LongTensor, scalar
    """
    def __init__(self, X, y):
        # YOUR CODE HERE
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)
    
    def __len__(self):
        # YOUR CODE HERE
        return len(self.X)
    
    def __getitem__(self, idx):
        # YOUR CODE HERE
        return self.X[idx].unsqueeze(0),self.y[idx]
    
BATCH_SIZE = 64

train_ds = EEGDataset(X_train, y_train)
test_ds  = EEGDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

