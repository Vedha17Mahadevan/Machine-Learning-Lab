import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# distance function
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# KNN function
def knn(X_train, y_train, X_test, k=3):
    predictions = []

    for test_point in X_test:
        distances = []

        for i in range(len(X_train)):
            dist = euclidean_distance(test_point, X_train[i])
            distances.append((dist, y_train[i]))

        distances.sort(key=lambda x: x[0])

        k_neighbors = distances[:k]
        labels = [label for _, label in k_neighbors]

        most_common = Counter(labels).most_common(1)[0][0]
        predictions.append(most_common)

    return np.array(predictions)

# predict
y_pred = knn(X_train, y_train, X_test, k=3)

# accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)

print("Predictions:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy)
