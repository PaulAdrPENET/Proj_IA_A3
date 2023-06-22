import numpy as np
from tmp_PA import data_ready, repartition_data, train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from math import radians, cos, sin, asin, sqrt


# Classification KNN from scratch : Gabriel Lefèvre


class KNNFromScratch:
    def __init__(self, nb_neighbor):
        # Initialisation des variables (nombre de voisins, données d'entrainement)
        self.nb_neighbor = nb_neighbor
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    @staticmethod
    def haversine_distance(coord1, coord2):
        r = 6371  # Rayon de la Terre en km
        # On converti les coordonnées en radian
        coord1[1], coord1[0], coord2[1], coord2[0] = map(radians, [coord1[1], coord1[0], coord2[1], coord2[0]])
        dlon = coord2[1] - coord1[1]
        dlat = coord2[0] - coord1[0]
        # On applique la formule de calcul de la fonction de haversine
        dist_haver = 2 * asin(sqrt(sin(dlat / 2) ** 2 + cos(coord1[0]) * cos(coord2[0]) * sin(dlon / 2) ** 2))
        return dist_haver * r

    # Fonction de calcul de distance : norme L1 (valeurs absolues)
    @staticmethod
    def dist_L1(coord1, coord2):
        dlat_abs = abs(coord1[0] - coord2[0])
        dlong_abs = abs(coord1[1] - coord2[1])
        distance = dlat_abs + dlong_abs
        return distance

    # Fonction de calcul de distance : norme L2 (euclidienne)
    @staticmethod
    def dist_L2(coord1, coord2):
        dlat_squared = (coord1[0] - coord2[0]) ** 2
        dlong_squared = (coord1[1] - coord2[1]) ** 2
        distance = sqrt(dlat_squared + dlong_squared)
        return distance

    def classification_knn(self, x_test):
        prediction = []
        # On parcourt chaque point des données test

        for _, row1 in x_test.iterrows():
            distances = []
            # On parcourt chaque point des données d'entrainement pour les comparer à celle de test
            for _, row2 in x_train.iterrows():
                coord1 = (row1['latitude'], row1['longitude'])
                coord2 = (row2['latitude'], row2['longitude'])
                # On calcule et on récupère chaque distance
                distance = self.dist_L2(coord1, coord2)
                distances.append(distance)
            # On transtype la liste distance en un tableau array
            distances = np.array(distances)
            # On tri le tableau pour placer le voisin le plus proche en début de liste
            idx_sorted = np.argsort(distances)
            print(idx_sorted)
            # On récupère les indices des k voisins les plus proches
            # (dépend de la valeur voulue lors de l'instanciation)
            nearest_neighbor_idx = idx_sorted[:self.nb_neighbor]
            # On récupère les labels de ces voisins les plus proches
            nearest_neighbor_lbls = [self.y_train.values[i] for i in nearest_neighbor_idx]
            # On compte la récurrence de chaque voisins
            # Counter compte les récurrences et les tri par ordre du plus récurrent au moins récurrent
            # .most_common(1) retourne le premier plus récurrent
            most_common_neighbor = Counter(nearest_neighbor_lbls).most_common(1)
            prediction.append(most_common_neighbor[0][0])
        return prediction


x, y = repartition_data(data_ready)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

# Création et entraînement de l'instance KNNFromScratch
knn = KNNFromScratch(nb_neighbor=5)
knn.fit(x_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = knn.classification_knn(x_test)

# Calcul de l'exactitude (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Score_accuracy:", accuracy)
