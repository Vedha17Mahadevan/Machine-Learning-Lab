def fit_linear(X, y):
    x = X[:,0]
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    m = np.sum((x-x_mean)*(y-y_mean)) / np.sum((x-x_mean)**2)
    c = y_mean - m*x_mean

    return m,c

def predict_linear(X, m, c):
    x = X[:,0]
    return m*x + c
