import torch
import sys

# Import preprocessing pipeline
sys.path.insert(0, '../4.data preprocessing')
from run_pipeline import run_pipeline

DATA_DIR = '../mne_data/bci_iv_2a'

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')