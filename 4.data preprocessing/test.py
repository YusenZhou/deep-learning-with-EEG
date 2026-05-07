from run_pipeline import run_pipeline,split_epochs,load_subject_epochs,reject_artifacts,N_CHANNELS
from fit_csp import fit_csp
import sys
import numpy as np

if sys.argv[1] == 'run_pipeline':
    all_results = {}
    print("Running pipeline on all subjects...\n")

    for subj in range(1, 15):
        try:
            result = run_pipeline(subj)
            all_results[subj] = result
            print(f"  Subject {subj:>2d}: train={result['X_train'].shape[0]:>4d} windows, "
                f"test={result['X_test'].shape[0]:>4d} windows, "
                f"rejected={result['n_rejected']:>3d} trials")
        except Exception as e:
            print(f"  Subject {subj:>2d}: FAILED — {e}")

    print(f"\nSuccessfully processed {len(all_results)}/14 subjects")

if sys.argv[1]=='fit_csp':
    test_data, test_labels = load_subject_epochs(1)
    _clean_data, _clean_labels, _n_rej = reject_artifacts(test_data, test_labels, threshold_uv=60.0)
    _X_tr, _X_te, _y_tr, _y_te = split_epochs(_clean_data, _clean_labels)
    _csp_tr, _csp_te, _csp_obj = fit_csp(_X_tr, _y_tr, _X_te, n_components=4)

    assert _csp_tr.ndim == 2, f"Expected 2D, got {_csp_tr.ndim}D"
    assert _csp_tr.shape[0] == _X_tr.shape[0], "Number of training samples changed"
    assert _csp_te.shape[0] == _X_te.shape[0], "Number of test samples changed"
    assert _csp_tr.shape[1] == _csp_te.shape[1], "Feature dims don't match between train and test"

    _class_means = np.array([_csp_tr[_y_tr == c].mean(axis=0) for c in range(4)])
    assert not np.allclose(_class_means[0], _class_means[1], atol=0.1), \
        "CSP features are identical across classes — something is wrong"

    print(f"✓ fit_csp: train shape {_csp_tr.shape}, test shape {_csp_te.shape}")
    print(f"  Feature dimensionality: {_csp_tr.shape[1]}")
    print(f"  Class 0 mean features (first 4): {_class_means[0][:4].round(3)}")
    print(f"  Class 1 mean features (first 4): {_class_means[1][:4].round(3)}")

    print(f"\n  (CSP topomap visualization skipped — {N_CHANNELS} channels)")