import torch
import os
from src.config import DEVICE,LR,N_EPOCHS
from src.model import EEGNet
from src.data import test_loader,train_loader
from src.train import train_one_epoch,evaluate



def ablation_depth_multiplier():
    # Ablation 1: Depth multiplier D
    # We run on Subject A01 only for speed

    ABLATION_D_PATH = 'metrics/ablation_depth_multiplier.pt'

    if os.path.exists(ABLATION_D_PATH):
        ablation_d_results = torch.load(ABLATION_D_PATH,weights_only=False)
        print("Loaded cached ablation results.")
    else:
        d_values = [1, 2, 4]
        ablation_d_results = {}
        
        for d_val in d_values:
            print(f"\nD={d_val}:")
            model = EEGNet(D=d_val).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            n_params = sum(p.numel() for p in model.parameters())
            
            for epoch in range(N_EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
            
            acc, kappa, _, _ = evaluate(model, test_loader, DEVICE)
            ablation_d_results[d_val] = {'accuracy': acc, 'kappa': kappa, 'n_params': n_params}
            print(f"  D={d_val}: acc={acc:.4f}, kappa={kappa:.4f}, params={n_params:,}")
        
        os.makedirs('metrics', exist_ok=True)
        torch.save(ablation_d_results, ABLATION_D_PATH)

    print(f"\n{'D':>5} {'Accuracy':>10} {'Kappa':>10} {'Params':>10}")
    print('-' * 38)
    for d_val in sorted(ablation_d_results.keys()):
        r = ablation_d_results[d_val]
        print(f"{d_val:>5} {r['accuracy']:>10.4f} {r['kappa']:>10.4f} {r['n_params']:>10,}")

def ablation_dropout():
    # Ablation 2: Dropout rate

    ABLATION_DROP_PATH = 'metrics/ablation_dropout.pt'

    if os.path.exists(ABLATION_DROP_PATH):
        ablation_drop_results = torch.load(ABLATION_DROP_PATH,weights_only=False)
        print("Loaded cached dropout ablation results.")
    else:
        drop_values = [0.25, 0.5, 0.75]
        ablation_drop_results = {}
        
        for drop_val in drop_values:
            print(f"\nDropout={drop_val}:")
            model = EEGNet(dropout=drop_val).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            
            train_losses = []
            for epoch in range(N_EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
                train_losses.append(loss)
            
            acc, kappa, _, _ = evaluate(model, test_loader, DEVICE)
            ablation_drop_results[drop_val] = {
                'accuracy': acc, 'kappa': kappa,
                'final_train_loss': train_losses[-1]
            }
            print(f"  Dropout={drop_val}: acc={acc:.4f}, kappa={kappa:.4f}, final_loss={train_losses[-1]:.4f}")
        
        os.makedirs('metrics', exist_ok=True)
        torch.save(ablation_drop_results, ABLATION_DROP_PATH)

    print(f"\n{'Dropout':>10} {'Accuracy':>10} {'Kappa':>10} {'Final Loss':>12}")
    print('-' * 45)
    for drop_val in sorted(ablation_drop_results.keys()):
        r = ablation_drop_results[drop_val]
        print(f"{drop_val:>10.2f} {r['accuracy']:>10.4f} {r['kappa']:>10.4f} {r['final_train_loss']:>12.4f}")

def ablation_kernel_length():
    # Ablation 3: Temporal kernel length

    ABLATION_KERN_PATH = 'metrics/ablation_kern_len.pt'

    if os.path.exists(ABLATION_KERN_PATH):
        ablation_kern_results = torch.load(ABLATION_KERN_PATH,weights_only=False)
        print("Loaded cached kernel ablation results.")
    else:
        kern_values = [32, 64, 125]
        ablation_kern_results = {}
        
        for kern_val in kern_values:
            print(f"\nkern_len={kern_val}:")
            model = EEGNet(kern_len=kern_val).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            n_params = sum(p.numel() for p in model.parameters())
            
            for epoch in range(N_EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
            
            acc, kappa, _, _ = evaluate(model, test_loader, DEVICE)
            ablation_kern_results[kern_val] = {'accuracy': acc, 'kappa': kappa, 'n_params': n_params}
            print(f"  kern_len={kern_val}: acc={acc:.4f}, kappa={kappa:.4f}, params={n_params:,}")
        
        os.makedirs('metrics', exist_ok=True)
        torch.save(ablation_kern_results, ABLATION_KERN_PATH)

    print(f"\n{'kern_len':>10} {'Accuracy':>10} {'Kappa':>10} {'Params':>10}")
    print('-' * 43)
    for kern_val in sorted(ablation_kern_results.keys()):
        r = ablation_kern_results[kern_val]
        print(f"{kern_val:>10} {r['accuracy']:>10.4f} {r['kappa']:>10.4f} {r['n_params']:>10,}")