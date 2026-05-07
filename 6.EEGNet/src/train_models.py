import os
import torch
from torch.utils.data import DataLoader
from src.model import EEGNet
from src.data import train_loader,test_loader,run_pipeline,EEGDataset
from src.train import train_one_epoch,evaluate
from src.config import DEVICE,N_EPOCHS,LR,BATCH_SIZE


def train_on_A1():
    # Train on Subject A01 with caching
    METRICS_PATH_A01 = 'metrics/eegnet_A01_metrics.pt'
    MODEL_PATH_A01   = 'models/eegnet_A01.pt'

    if os.path.exists(METRICS_PATH_A01) and os.path.exists(MODEL_PATH_A01):
        a01_metrics = torch.load(METRICS_PATH_A01,weights_only=False)
        model_a01 = EEGNet().to(DEVICE)
        model_a01.load_state_dict(torch.load(MODEL_PATH_A01, map_location=DEVICE))
        print("Loaded cached A01 results.")
    else:
        model_a01 = EEGNet().to(DEVICE)
        optimizer = torch.optim.Adam(model_a01.parameters(), lr=LR)
        
        train_losses = []
        test_accs = []
        test_kappas = []
        
        for epoch in range(N_EPOCHS):
            loss = train_one_epoch(model_a01, train_loader, optimizer, DEVICE)
            train_losses.append(loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc, kappa, _, _ = evaluate(model_a01, test_loader, DEVICE)
                test_accs.append(acc)
                test_kappas.append(kappa)
                print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | loss: {loss:.4f} | test acc: {acc:.4f} | kappa: {kappa:.4f}")
        
        # Final evaluation
        final_acc, final_kappa, final_preds, final_labels = evaluate(model_a01, test_loader, DEVICE)
        
        a01_metrics = {
            'train_losses': train_losses,
            'test_accs': test_accs,
            'test_kappas': test_kappas,
            'final_acc': final_acc,
            'final_kappa': final_kappa,
            'final_preds': final_preds,
            'final_labels': final_labels
        }
        
        os.makedirs('metrics', exist_ok=True)
        os.makedirs('models',  exist_ok=True)
        torch.save(a01_metrics,              METRICS_PATH_A01)
        torch.save(model_a01.state_dict(),   MODEL_PATH_A01)
        print(f"\nTraining complete. Final acc: {final_acc:.4f}, kappa: {final_kappa:.4f}")

def train_on_all_data():
    METRICS_PATH_ALL = 'metrics/eegnet_all_subjects.pt'

    if os.path.exists(METRICS_PATH_ALL):
        all_results = torch.load(METRICS_PATH_ALL,weights_only=False)
        print("Loaded cached 14-subject results.")
        for subj in sorted(all_results.keys()):
            r = all_results[subj]
            print(f"  {subj}: acc={r['accuracy']:.4f}, kappa={r['kappa']:.4f}")
    else:
        all_results = {}
        subjects = list(range(1, 15))
        
        for subj in subjects:
            print(f"\n{'='*50}")
            print(f"Training Subject {subj}")
            print(f"{'='*50}")
            
            # Load data
            data = run_pipeline(subj)
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
            model = EEGNet().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            
            for epoch in range(N_EPOCHS):
                loss = train_one_epoch(model, tr_loader, optimizer, DEVICE)
                if (epoch + 1) % 100 == 0:
                    acc, kappa, _, _ = evaluate(model, te_loader, DEVICE)
                    print(f"  Epoch {epoch+1}/{N_EPOCHS} | loss: {loss:.4f} | acc: {acc:.4f} | kappa: {kappa:.4f}")
            
            # Final evaluation
            acc, kappa, preds, labels = evaluate(model, te_loader, DEVICE)
            all_results[subj] = {
                'accuracy': acc,
                'kappa': kappa,
                'preds': preds,
                'labels': labels
            }
            print(f"  Final: acc={acc:.4f}, kappa={kappa:.4f}")
            
            # Save model weights per subject
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/eegnet_{subj}.pt')
        
        # Save all results
        os.makedirs('metrics', exist_ok=True)
        torch.save(all_results, METRICS_PATH_ALL)
        print("\nAll results saved.")
