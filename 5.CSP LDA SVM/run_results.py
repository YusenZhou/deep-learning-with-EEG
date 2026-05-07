import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from function import run_csp_classifier
from sklearn.base import clone
import sys
sys.path.insert(0, '../4.data preprocessing')
from run_pipeline import run_pipeline

np.random.seed(42)

plt.rcParams['figure.figsize'] = (10, 5)

methods = {
    'CSP+LDA':        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    'CSP+linearSVM':  SVC(kernel='linear', C=1.0),
    'CSP+rbfSVM':     SVC(kernel='rbf',    C=1.0, gamma='scale'),
}

subjects = list(range(1, 15))
results = {m: {} for m in methods}

for subj in subjects:
    data = run_pipeline(subj)

    Xtr = data['X_train']
    Xte  = data['X_test']
    ytr = data['y_train']
    yte  = data['y_test']
    for name, clf in methods.items():
        res = run_csp_classifier(Xtr, Xte, ytr, yte, clone(clf))
        results[name][subj] = res
    print(f"Subject {subj:>2d}: " + "  ".join(
        f"{name} κ={results[name][subj]['kappa']:.3f}" for name in methods))
    
# Summary table
print(f"{'Method':<18} {'Acc mean ± std':<22} {'Kappa mean ± std':<22}")
print("-" * 62)
for name in methods:
    accs = [results[name][s]['accuracy'] for s in subjects]
    kapps = [results[name][s]['kappa'] for s in subjects]
    print(f"{name:<18} {np.mean(accs):.3f} ± {np.std(accs):.3f}       "
          f"{np.mean(kapps):.3f} ± {np.std(kapps):.3f}")
    
# Per-subject kappa bar chart
fig, ax = plt.subplots(figsize=(11, 4.5))
subject_labels = [f'S{s:02d}' for s in subjects]
x = np.arange(len(subjects))
width = 0.27
for i, name in enumerate(methods):
    kapps = [results[name][s]['kappa'] for s in subjects]
    ax.bar(x + (i - 1) * width, kapps, width, label=name)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='kappa=0.5 (strong)')
ax.set_xticks(x); ax.set_xticklabels(subject_labels)
ax.set_ylabel("Cohen's kappa"); ax.set_title('HGD — per-subject test kappa')
ax.legend()
plt.tight_layout()
plt.show()