from scipy.linalg import eigh
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import numpy as np

def trial_covariances(X):
    """
    Compute normalized spatial covariance for each trial.

    Args:
        X: np.ndarray, shape (n_trials, C, T)
           Band-passed EEG epochs. C=22, T=250 for Dataset 2a.

    Returns:
        covs: np.ndarray, shape (n_trials, C, C)
              Each (C, C) matrix is (X_i @ X_i.T) / trace(X_i @ X_i.T).
    """
    # YOUR CODE HERE
    covs = X @ X.transpose(0,2,1)
    traces = np.trace(covs,axis1=1,axis2=2)
    return covs / traces[:,None,None]

def fit_binary_csp(X0, X1, n_components=4):
    """
    Fit binary CSP between two classes.

    Args:
        X0: np.ndarray, shape (n_trials_0, C, T). Trials of class 0.
        X1: np.ndarray, shape (n_trials_1, C, T). Trials of class 1.
        n_components: int, must be even. Number of spatial filters to keep
                      (half from top eigenvalues, half from bottom).

    Returns:
        W: np.ndarray, shape (n_components, C)
           Each row is one spatial filter. Top n_components//2 rows
           maximize variance for class 0; bottom n_components//2 rows
           maximize variance for class 1.
    """
    assert n_components % 2 == 0, "n_components must be even"
    cov_0 = trial_covariances(X0).mean(axis=0)
    cov_1 = trial_covariances(X1).mean(axis=0)
    composite = cov_0+cov_1
    composite += np.eye(composite.shape[0]) * 1e-6
    eigenvalues,eigenvectors = eigh(cov_0,composite)
    eigenvalues  = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    k = n_components // 2
    selected = np.concatenate([eigenvectors[:, :k], eigenvectors[:, -k:]], axis=1)  # (C, n_components)
    W = selected.T
    return W
    

def fit_csp_ovr(X, y, n_components=4, n_classes=4):
    """
    Fit multi-class CSP using one-vs-rest.

    Args:
        X: np.ndarray, shape (n_trials, C, T)
        y: np.ndarray, shape (n_trials,). Integer labels in {0, ..., n_classes-1}.
        n_components: int, filters kept per binary problem.
        n_classes: int.

    Returns:
        W: np.ndarray, shape (n_classes * n_components, C)
           Stacked spatial filters from all n_classes binary CSPs.
    """
    # YOUR CODE HERE
    filters = []
    
    for c in range(n_classes):
        X_c    = X[y == c]           # trials for class c
        X_rest = X[y != c]           # all other trials
        W_c    = fit_binary_csp(X_c, X_rest, n_components)  # (n_components, C)
        filters.append(W_c)
    
    W = np.concatenate(filters, axis=0)  # (n_classes * n_components, C)
    return W

def csp_features(X, W):
    """
    Apply CSP filters and extract log-variance features.

    Args:
        X: np.ndarray, shape (n_trials, C, T)
        W: np.ndarray, shape (n_filters, C)

    Returns:
        features: np.ndarray, shape (n_trials, n_filters)
                  features[i, j] = log(var_t(W[j] @ X[i])).
    """
    # YOUR CODE HERE
    Y = X.transpose(0, 2, 1) @ W.T 
    Y = Y.transpose(0, 2, 1)
    var = np.var(Y,axis = 2)
    features = np.log(var)
    return features

def run_csp_classifier(X_train, X_test, y_train, y_test, classifier, n_components=4, n_classes=4):
    """
    End-to-end CSP + classifier pipeline.

    Args:
        X_train, X_test: np.ndarray, shape (n_trials, C, T)
        y_train, y_test: np.ndarray, shape (n_trials,)
        classifier: sklearn-style estimator with .fit() and .predict()
        n_components: CSP components per class (one-vs-rest)

    Returns:
        dict with 'accuracy' and 'kappa' on the test set.
    """
    # YOUR CODE HERE
    W = fit_csp_ovr(X_train, y_train, n_components=n_components, n_classes=n_classes)
    feat_train = csp_features(X_train, W)
    feat_test  = csp_features(X_test,  W)
    
    classifier.fit(feat_train, y_train)
    y_pred = classifier.predict(feat_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'kappa':    cohen_kappa_score(y_test, y_pred)
    }