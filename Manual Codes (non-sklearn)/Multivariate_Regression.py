def fit_multi(X, y):
    X_b = np.c_[np.ones((X.shape[0],1)), X]
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta

def predict_multi(X, beta):
    X_b = np.c_[np.ones((X.shape[0],1)), X]
    return X_b @ beta
