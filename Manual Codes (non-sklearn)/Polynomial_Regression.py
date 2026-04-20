def fit_poly(X, y):
    x = X[:,0]
    X_poly = np.c_[np.ones(len(x)), x, x**2]
    beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return beta

def predict_poly(X, beta):
    x = X[:,0]
    X_poly = np.c_[np.ones(len(x)), x, x**2]
    return X_poly @ beta
