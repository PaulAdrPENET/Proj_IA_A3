from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tmp_PA import data_ready, repartition_data, train_test_split


# Classification KNN avec la bibliothèque scikit-learn : Gabriel Lefèvre


x, y = repartition_data(data_ready)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

# Création d'une instance du classifieur KNN
inst_knn = KNeighborsClassifier(n_neighbors=5)
# Entrainement du modèle
inst_knn.fit(x_train, y_train)
# Test de prédiction
y_pred = inst_knn.predict(x_test)
# Calcul la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
# Affichage du résultat
print("Précision : ", accuracy)



