import numpy as np
from tmp_PA import data_ready, repartition_data, train_test_split
from collections import Counter


class KNNFromScratch:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def predict(self, x_test):
        y_pred = []
        for test_point in x_test:
            distances = []
            for train_point in self.X_train:
                distance = self.euclidean_distance(test_point, train_point)
                distances.append(distance)
            sorted_indices = np.argsort(distances)
            k_nearest_indices = sorted_indices[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            y_pred.append(most_common[0][0])
        return y_pred
