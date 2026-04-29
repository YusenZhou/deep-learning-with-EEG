import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet: A compact convolutional neural network for EEG-based BCIs.
    
    Architecture:
        Block 1: Temporal Conv → BN → Depthwise Conv → BN → ELU → AvgPool → Dropout
        Block 2: Separable Conv (Depthwise + Pointwise) → BN → ELU → AvgPool → Dropout
        Classifier: Flatten → Linear
    
    Args:
        n_channels: int, number of EEG channels (22)
        n_timepoints: int, time samples per window (250)
        n_classes: int, number of output classes (4)
        F1: int, temporal filters (8)
        D: int, depth multiplier for spatial filters (2)
        F2: int, pointwise filters (default F1*D=16)
        kern_len: int, temporal kernel length (125)
        sep_kern: int, separable conv kernel (16)
        pool1: int, first pooling factor (4)
        pool2: int, second pooling factor (8)
        dropout: float, dropout rate (0.5)
    
    forward(x):
        x: (B, 1, C, T)
        returns: (B, n_classes) — logits
    """
    def __init__(self, n_channels=22, n_timepoints=250, n_classes=4,
                 F1=8, D=2, F2=None, kern_len=125, sep_kern=16,
                 pool1=4, pool2=8, dropout=0.5):
        super().__init__()
        if F2 is None:
            F2 = F1 * D
        
        # Store config for later use
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # ============================================================
        # Block 1 — Temporal + Spatial filtering
        # YOUR CODE HERE: define the layers for Block 1
        #   self.temporal_conv  = ...
        #   self.bn1            = ...
        #   self.depthwise_conv = ...
        #   self.bn2            = ...
        #   self.pool1          = ...
        #   self.drop1          = ...
        # ============================================================
        self.block1 = nn.Sequential(
            nn.Conv2d(1,F1,kernel_size=(1,kern_len), padding=(0, kern_len//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1,F1*D,(n_channels,1),groups = F1,bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1,pool1)),
            nn.Dropout2d(dropout)
        )
        
        # ============================================================
        # Block 2 — Separable convolution
        # YOUR CODE HERE: define the layers for Block 2
        #   self.sep_depthwise = ...
        #   self.sep_pointwise = ...
        #   self.bn3           = ...
        #   self.pool2         = ...
        #   self.drop2         = ...
        # ============================================================
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D,F1*D,kernel_size=(1,sep_kern),groups=F1*D, padding=(0,sep_kern//2), bias=False),
            nn.Conv2d(F1*D,F2,(1,1),bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,pool2)),
            nn.Dropout2d(dropout)
        )
        
        # ============================================================
        # Classifier
        # YOUR CODE HERE: compute flattened size, define Linear layer
        #   self.classifier = ...
        # ============================================================
        _flat = F2 * (n_timepoints // (pool1 * pool2))

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(_flat,n_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, C, T)
        Returns:
            logits: (B, n_classes)
        """
        # ============================================================
        # YOUR CODE HERE: wire the layers defined above
        # Block 1: temporal_conv → bn1 → depthwise_conv → bn2 → elu → pool1 → drop1
        # Block 2: sep_depthwise → sep_pointwise → bn3 → elu → pool2 → drop2
        # Classifier: flatten → classifier
        # ============================================================
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)