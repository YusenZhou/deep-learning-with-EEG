import torch
import os
from src.config import DEVICE
from src.data import train_loader,test_loader
from src.model import EEGConformer,train_one_epoch,evaluate

N_EPOCHS = 300
LR = 5e-4
BATCH_SIZE = 64
WEIGHT_DECAY = 0.01

def ablation_layers():
    # Ablation 1: Number of Transformer layers
# Test on Subject A01 only (for speed)

    ABLATION1_PATH = 'metrics/conformer_ablation_layers.pt'

    if os.path.exists(ABLATION1_PATH):
        ablation1_results = torch.load(ABLATION1_PATH)
        print("Loaded cached ablation results.")
    else:
        layer_configs = [2, 4, 6, 8]
        ablation1_results = {}
        
        for n_layers in layer_configs:
            print(f"\nTraining with n_layers={n_layers}...")
            model = EEGConformer(n_layers=n_layers).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
            
            for epoch in range(N_EPOCHS):
                train_one_epoch(model, train_loader, optimizer, DEVICE)
                scheduler.step()
            
            acc, kappa, _, _ = evaluate(model, test_loader, DEVICE)
            n_params = sum(p.numel() for p in model.parameters())
            ablation1_results[n_layers] = {
                'accuracy': acc, 'kappa': kappa, 'n_params': n_params
            }
            print(f"  n_layers={n_layers}: acc={acc:.4f}, kappa={kappa:.4f}, params={n_params:,}")
        
        os.makedirs('metrics', exist_ok=True)
        torch.save(ablation1_results, ABLATION1_PATH)

    # Display results
    print(f"\n{'n_layers':<12} {'Accuracy':>10} {'Kappa':>10} {'Params':>12}")
    print('-' * 46)
    for n_layers in sorted(ablation1_results.keys()):
        r = ablation1_results[n_layers]
        print(f"{n_layers:<12} {r['accuracy']:>10.4f} {r['kappa']:>10.4f} {r['n_params']:>12,}")



def ablation_heads():
    # Ablation 2: Number of attention heads

    ABLATION2_PATH = 'metrics/conformer_ablation_heads.pt'

    if os.path.exists(ABLATION2_PATH):
        ablation2_results = torch.load(ABLATION2_PATH)
        print("Loaded cached ablation results.")
    else:
        head_configs = [1, 2, 5, 8, 10]  # all divide 40
        ablation2_results = {}
        
        for n_heads in head_configs:
            print(f"\nTraining with n_heads={n_heads} (d_k={40//n_heads})...")
            model = EEGConformer(n_heads=n_heads).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
            
            for epoch in range(N_EPOCHS):
                train_one_epoch(model, train_loader, optimizer, DEVICE)
                scheduler.step()
            
            acc, kappa, _, _ = evaluate(model, test_loader, DEVICE)
            ablation2_results[n_heads] = {'accuracy': acc, 'kappa': kappa, 'd_k': 40 // n_heads}
            print(f"  n_heads={n_heads}: acc={acc:.4f}, kappa={kappa:.4f}")
        
        os.makedirs('metrics', exist_ok=True)
        torch.save(ablation2_results, ABLATION2_PATH)

    # Display results
    print(f"\n{'n_heads':<10} {'d_k':>6} {'Accuracy':>10} {'Kappa':>10}")
    print('-' * 38)
    for n_heads in sorted(ablation2_results.keys()):
        r = ablation2_results[n_heads]
        print(f"{n_heads:<10} {r['d_k']:>6} {r['accuracy']:>10.4f} {r['kappa']:>10.4f}")

def ablation_kern_len():
    # Ablation 2: Number of attention heads

    ABLATION2_PATH = 'metrics/conformer_ablation_heads.pt'

    if os.path.exists(ABLATION2_PATH):
        ablation2_results = torch.load(ABLATION2_PATH)
        print("Loaded cached ablation results.")
    else:
        head_configs = [1, 2, 5, 8, 10]  # all divide 40
        ablation2_results = {}
        
        for n_heads in head_configs:
            print(f"\nTraining with n_heads={n_heads} (d_k={40//n_heads})...")
            model = EEGConformer(n_heads=n_heads).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
            
            for epoch in range(N_EPOCHS):
                train_one_epoch(model, train_loader, optimizer, DEVICE)
                scheduler.step()
            
            acc, kappa, _, _ = evaluate(model, test_loader, DEVICE)
            ablation2_results[n_heads] = {'accuracy': acc, 'kappa': kappa, 'd_k': 40 // n_heads}
            print(f"  n_heads={n_heads}: acc={acc:.4f}, kappa={kappa:.4f}")
        
        os.makedirs('metrics', exist_ok=True)
        torch.save(ablation2_results, ABLATION2_PATH)

    # Display results
    print(f"\n{'n_heads':<10} {'d_k':>6} {'Accuracy':>10} {'Kappa':>10}")
    print('-' * 38)
    for n_heads in sorted(ablation2_results.keys()):
        r = ablation2_results[n_heads]
        print(f"{n_heads:<10} {r['d_k']:>6} {r['accuracy']:>10.4f} {r['kappa']:>10.4f}")