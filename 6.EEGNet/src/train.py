import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

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
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)


def evaluate(model, loader, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: nn.Module
        loader: DataLoader
        device: torch.device
    
    Returns:
        accuracy: float — fraction correct
        kappa: float — Cohen's kappa
        all_preds: np.ndarray — predicted labels
        all_labels: np.ndarray — true labels
    """
    # YOUR CODE HERE
    model.eval()
    #correct = 0
    #total = 0
    all_preds = []
    all_labels =[]
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device),y.to(device)
            logits = model(X)
            preds = torch.argmax(logits,dim= 1)
            # correct += (preds == y).sum().item()
            # total += len(X)
            all_preds.append(preds)
            all_labels.append(y)
    all_preds  = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    correct = (all_preds == all_labels).sum().item()
    total = len(all_preds)
    accuracy = correct/total
    kappa = cohen_kappa_score(all_labels,all_preds)
    return accuracy,kappa,all_preds,all_labels
