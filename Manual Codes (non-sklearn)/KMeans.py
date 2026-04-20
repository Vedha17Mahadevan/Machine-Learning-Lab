def kmeans(X, k=3, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]

        for point in X:
            dists = [np.linalg.norm(point-c) for c in centroids]
            idx = np.argmin(dists)
            clusters[idx].append(point)

        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    labels = []
    for point in X:
        dists = [np.linalg.norm(point-c) for c in centroids]
        labels.append(np.argmin(dists))

    return np.array(labels), centroids
