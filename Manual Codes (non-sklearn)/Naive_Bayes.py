import numpy as np

def train_nb(X, y):
    classes = np.unique(y)
    mean = {}
    var = {}
    prior = {}

    for c in classes:
        X_c = X[y == c]
        mean[c] = np.mean(X_c, axis=0)
        var[c] = np.var(X_c, axis=0)
        prior[c] = len(X_c) / len(X)

    return mean, var, prior

def gaussian(x, mean, var):
    return (1 / np.sqrt(2*np.pi*var)) * np.exp(-(x-mean)**2 / (2*var))

def predict_nb(X, mean, var, prior):
    preds = []
    for x in X:
        probs = {}
        for c in mean:
            probs[c] = np.sum(np.log(gaussian(x, mean[c], var[c]))) + np.log(prior[c])
        preds.append(max(probs, key=probs.get))
    return np.array(preds)
