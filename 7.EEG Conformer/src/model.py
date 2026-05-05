import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score


class PatchEmbedding(nn.Module):
    """
    CNN front-end that converts raw EEG into a sequence of patch tokens.
    
    Args:
        n_channels: int, number of EEG channels (22)
        F1: int, number of temporal filters (40)
        D: int, depth multiplier (1)
        kern_len: int, temporal kernel size (25)
        pool_size: int, average pooling kernel (75)
        pool_stride: int, pooling stride (15)
        dropout: float, dropout rate (0.5)
    
    forward(x):
        x: (B, 1, C, T)
        returns: (B, T', d_model) where d_model = F1 * D
    """
    def __init__(self, n_channels=22, F1=40, D=1, kern_len=25,
                 pool_size=75, pool_stride=15, dropout=0.5):
        super().__init__()
        self.d_model = F1 * D
        
        # ============================================================
        # YOUR CODE HERE: define the CNN layers and pooling
        # ============================================================
        self.cnn = nn.Sequential(
            nn.Conv2d(1,F1,(1,kern_len),padding = (0,kern_len//2),bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.Conv2d(F1,F1*D,(n_channels,1),groups=F1,bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AvgPool1d(pool_size,stride=pool_stride)

        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.d_model))
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, C, T) — e.g. (B, 1, 22, 250)
        Returns:
            tokens: (B, T', d_model) — e.g. (B, 12, 40)
        """
        # ============================================================
        # YOUR CODE HERE:
        # 1. CNN layers: (B,1,C,T) → (B, F1*D, 1, T)
        # 2. Squeeze dim 2: → (B, F1*D, T)
        # 3. AvgPool1d: → (B, F1*D, T')
        # 4. Transpose: → (B, T', F1*D)
        # ============================================================
        x = self.cnn(x)
        x = x.squeeze(2)
        x = self.pool(x) 
        x = x.transpose(1,2)
        seq_len = x.size(1)
        return x + self.pos_embed[:, :seq_len, :]
    
class TransformerBlock(nn.Module):
    """
    Single Transformer encoder layer with Pre-LN.
    
    Args:
        d_model: int, embedding dimension (40)
        n_heads: int, number of attention heads (10)
        ff_ratio: int, FFN expansion ratio (3)
        dropout: float, dropout rate (0.5)
    
    forward(x):
        x: (B, T', d_model)
        returns: (output, attn_weights)
            output: (B, T', d_model)
            attn_weights: (B, T', T')
    """
    def __init__(self, d_model=40, n_heads=10, ff_ratio=3, dropout=0.5):
        super().__init__()
        
        # ============================================================
        # YOUR CODE HERE: define LayerNorms, MultiheadAttention, FFN
        # ============================================================
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model,n_heads,dropout,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_model*ff_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*ff_ratio,d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        """
        Args:
            x: (B, T', d_model)
        Returns:
            out: (B, T', d_model)
            attn_weights: (B, T', T') — attention weights for visualization
        """
        # ============================================================
        # YOUR CODE HERE:
        # Pre-LN attention block with residual
        # Pre-LN FFN block with residual
        # Return both output and attention weights
        # ============================================================
        residual = x
        x = self.ln1(x)
        x,attn_weights = self.attn(x,x,x)
        x = self.dropout(x)
        x = x+ residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x += residual
        return x, attn_weights
    
class EEGConformer(nn.Module):
    """
    EEG Conformer: CNN front-end + Transformer encoder + classifier.
    
    Args:
        n_channels: int (22)
        n_timepoints: int (250)
        n_classes: int (4)
        F1: int, temporal filters (40)
        D: int, depth multiplier (1)
        kern_len: int, temporal kernel (25)
        pool_size: int, pooling window (75)
        pool_stride: int, pooling stride (15)
        n_heads: int, attention heads (10)
        n_layers: int, Transformer layers (6)
        ff_ratio: int, FFN expansion (3)
        dropout: float (0.5)
    
    forward(x):
        x: (B, 1, C, T)
        returns: logits (B, n_classes)
    """
    def __init__(self, n_channels=22, n_timepoints=250, n_classes=4,
                 F1=40, D=1, kern_len=25,
                 pool_size=75, pool_stride=15,
                 n_heads=10, n_layers=6, ff_ratio=3, dropout=0.5):
        super().__init__()
        
        d_model = F1 * D
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Compute sequence length after pooling
        self.seq_len = (n_timepoints - pool_size) // pool_stride + 1
        
        # ============================================================
        # YOUR CODE HERE:
        # 1. self.patch_embed = PatchEmbedding(...)
        # 2. self.pos_embed  = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        # 3. self.pos_drop   = nn.Dropout(dropout)
        # 4. self.blocks     = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
        # 5. self.norm       = nn.LayerNorm(d_model)   — final norm before classifier
        # 6. self.classifier = nn.Linear(d_model, n_classes)
        # ============================================================
        self.patch_embed = PatchEmbedding(n_channels,F1,D,kern_len,pool_size,pool_stride,dropout)
        self.pos_embed = nn.Parameter(torch.zeros((1,self.seq_len,d_model)))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model,n_heads,ff_ratio,dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model,n_classes)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, C, T)
        Returns:
            logits: (B, n_classes)
        """
        # ============================================================
        # YOUR CODE HERE:
        # 1. tokens = self.patch_embed(x)          → (B, T', d_model)
        # 2. tokens = tokens + self.pos_embed      → add positional
        # 3. tokens = self.pos_drop(tokens)
        # 4. for block in self.blocks:
        #        tokens, _ = block(tokens)          → pass through each layer
        # 5. tokens = self.norm(tokens)             → final layer norm
        # 6. pooled = tokens.mean(dim=1)            → mean pool over T'
        # 7. logits = self.classifier(pooled)       → (B, n_classes)
        # ============================================================
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        for block in self.blocks:
            tokens,_ = block(tokens)
        tokens = self.norm(tokens)
        pooled =tokens.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
    
    def get_attention_weights(self, x):
        """
        Run forward pass and collect attention weights from all layers.
        Used for visualization — not for training.
        
        Args:
            x: (B, 1, C, T)
        Returns:
            attn_list: list of (B, T', T') tensors, one per layer
        """
        self.eval()
        with torch.no_grad():
            tokens = self.patch_embed(x)
            tokens = tokens + self.pos_embed
            tokens = self.pos_drop(tokens)
            
            attn_list = []
            for block in self.blocks:
                tokens, attn_w = block(tokens)
                attn_list.append(attn_w.cpu())
        
        return attn_list
    

def train_one_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: nn.Module
        loader: DataLoader
        optimizer: torch.optim.Optimizer
        device: torch.device
    
    Returns:
        avg_loss: float — mean cross-entropy loss over all batches
    """
    # YOUR CODE HERE
    model.train()
    total_loss = 0
    for X,y in loader:
        X,y = X.to(device),y.to(device)
        logits = model(X)
        optimizer.zero_grad()
        loss = F.cross_entropy(logits,y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(loader)



def evaluate(model, loader, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: nn.Module
        loader: DataLoader
        device: torch.device
    
    Returns:
        accuracy: float
        kappa: float — Cohen's kappa
        all_preds: np.ndarray — predicted labels
        all_labels: np.ndarray — true labels
    """
    # YOUR CODE HERE
    model.eval()
    all_preds = []
    all_labels =[]
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device),y.to(device)
            logits = model(X)
            preds = torch.argmax(logits,dim= 1)
            all_preds.append(preds)
            all_labels.append(y)
    all_preds  = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    correct = (all_preds == all_labels).sum().item()
    total = len(all_preds)
    accuracy = correct/total
    kappa = cohen_kappa_score(all_labels,all_preds)
    return accuracy,kappa,all_preds,all_labels