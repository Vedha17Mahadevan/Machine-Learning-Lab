def knn_reg(X_train, y_train, X_test, k=5):
    preds = []
    for test in X_test:
        dists = []
        for i in range(len(X_train)):
            d = euclidean(test, X_train[i])
            dists.append((d, y_train[i]))
        dists.sort(key=lambda x:x[0])
        neighbors = dists[:k]
        values = [val for _, val in neighbors]
        preds.append(np.mean(values))
    return np.array(preds)
