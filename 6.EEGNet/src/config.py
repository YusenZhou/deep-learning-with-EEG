import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


N_EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 64