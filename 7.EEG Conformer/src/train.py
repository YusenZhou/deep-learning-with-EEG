import os
import torch
from src.config import DEVICE,run_pipeline,DATA_DIR
from src.data import train_loader,test_loader,DataLoader,EEGDataset
from src.model import EEGConformer,train_one_epoch,evaluate

# Training hyperparameters
N_EPOCHS = 300
LR = 5e-4
BATCH_SIZE = 64
WEIGHT_DECAY = 0.01

def train_on_A1():
    # Train on Subject A01 with caching
    METRICS_PATH_A01 = 'metrics/conformer_A01_metrics.pt'
    MODEL_PATH_A01   = 'models/conformer_A01.pt'

    if os.path.exists(METRICS_PATH_A01) and os.path.exists(MODEL_PATH_A01):
        a01_metrics = torch.load(METRICS_PATH_A01)
        model_a01 = EEGConformer().to(DEVICE)
        model_a01.load_state_dict(torch.load(MODEL_PATH_A01, map_location=DEVICE))
        print("Loaded cached A01 results.")
    else:
        model_a01 = EEGConformer().to(DEVICE)
        optimizer = torch.optim.AdamW(model_a01.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
        
        train_losses = []
        test_accs = []
        test_kappas = []
        
        for epoch in range(N_EPOCHS):
            loss = train_one_epoch(model_a01, train_loader, optimizer, DEVICE)
            scheduler.step()
            train_losses.append(loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc, kappa, _, _ = evaluate(model_a01, test_loader, DEVICE)
                test_accs.append(acc)
                test_kappas.append(kappa)
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{N_EPOCHS} | loss: {loss:.4f} | "
                        f"acc: {acc:.4f} | kappa: {kappa:.4f} | "
                        f"lr: {scheduler.get_last_lr()[0]:.6f}")
        
        # Final evaluation
        final_acc, final_kappa, final_preds, final_labels = evaluate(model_a01, test_loader, DEVICE)
        print(f"\nFinal A01: acc={final_acc:.4f}, kappa={final_kappa:.4f}")
        
        a01_metrics = {
            'train_losses': train_losses,
            'test_accs': test_accs,
            'test_kappas': test_kappas,
            'final_accuracy': final_acc,
            'final_kappa': final_kappa,
            'final_preds': final_preds,
            'final_labels': final_labels
        }
        
        os.makedirs('metrics', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        torch.save(a01_metrics, METRICS_PATH_A01)
        torch.save(model_a01.state_dict(), MODEL_PATH_A01)
        print("Training complete. Saved.")

def train_all_subjects():
    METRICS_PATH_ALL = 'metrics/conformer_all_subjects.pt'
    if os.path.exists(METRICS_PATH_ALL):
        all_results = torch.load(METRICS_PATH_ALL)
        print("Loaded cached 9-subject results.")
        for subj in sorted(all_results.keys()):
            r = all_results[subj]
            print(f"  {subj}: acc={r['accuracy']:.4f}, kappa={r['kappa']:.4f}")
    else:
        all_results = {}
        subjects = [f'A0{i}' for i in range(1, 10)]
        
        for subj in subjects:
            print(f"\n{'='*50}")
            print(f"Training Subject {subj}")
            print(f"{'='*50}")
            
            # Load data
            data = run_pipeline(subj, DATA_DIR)
            X_tr = data['X_train']
            X_te = data['X_test']
            y_tr = data['y_train']
            y_te = data['y_test']
            
            # Create loaders
            tr_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                                shuffle=True, drop_last=False)
            te_loader = DataLoader(EEGDataset(X_te, y_te), batch_size=BATCH_SIZE,
                                shuffle=False, drop_last=False)
            
            # Train
            model = EEGConformer().to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
            
            for epoch in range(N_EPOCHS):
                loss = train_one_epoch(model, tr_loader, optimizer, DEVICE)
                scheduler.step()
                if (epoch + 1) % 100 == 0:
                    acc, kappa, _, _ = evaluate(model, te_loader, DEVICE)
                    print(f"  Epoch {epoch+1}/{N_EPOCHS} | loss: {loss:.4f} | "
                        f"acc: {acc:.4f} | kappa: {kappa:.4f}")
            
            # Final eval
            acc, kappa, preds, labels = evaluate(model, te_loader, DEVICE)
            all_results[subj] = {'accuracy': acc, 'kappa': kappa}
            print(f"  Final: acc={acc:.4f}, kappa={kappa:.4f}")
            
            # Save per-subject model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/conformer_{subj}.pt')
        
        os.makedirs('metrics', exist_ok=True)
        torch.save(all_results, METRICS_PATH_ALL)
        print("\nAll subjects trained and saved.")