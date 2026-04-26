# Stage 1 — CSP + LDA/SVM Classifier

Motor imagery classification on BCI Competition IV Dataset 2a using Common Spatial Patterns (CSP) with LDA and SVM classifiers.

---

## Files

| File | Purpose |
|------|---------|
| `csp_lda_svm.ipynb` | Step-by-step notebook: concept teaching, implementation, sanity checks, single-subject results |
| `function.py` | All CSP and classification functions: `trial_covariances`, `fit_binary_csp`, `fit_csp_ovr`, `csp_features`, `run_csp_classifier` |
| `run_results.py` | Runs all 9 subjects, prints summary table, generates per-subject kappa bar chart |

---

## Dependencies

```
numpy
scipy
scikit-learn
matplotlib
```

Plus the shared preprocessing pipeline from `../4. data preprocessing/run_pipeline.py`.

---

### Reproduce all results
Set your data directory at the top of `run_results.py`:

```python
DATA_DIR = '../mne_data/bci_iv_2a'   # adjust to your local path
```

Then run:

```bash
python run_results.py
```

This will:
1. Load and preprocess each subject via `run_pipeline()`
2. Run CSP+LDA, CSP+LinearSVM, and CSP+RBF SVM on all 9 subjects
3. Print a summary table of accuracy and Cohen's kappa (mean ± std)
4. Display a per-subject kappa bar chart

---

## Expected output

```
Method             Acc mean ± std         Kappa mean ± std
--------------------------------------------------------------
CSP+LDA            0.xxx ± 0.xxx          0.xxx ± 0.xxx
CSP+linearSVM      0.xxx ± 0.xxx          0.xxx ± 0.xxx
CSP+rbfSVM         0.xxx ± 0.xxx          0.xxx ± 0.xxx
```

CSP results serve as the performance baseline for all subsequent deep learning models (EEGNet, EEG Conformer).
