import pandas as pd
from Preparation import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import json

data = rewrite_data()

""" Réduction des dimensions Paul-Adrien PENET
On choisit de supprimer certaines classes (classes ayant une corrélation faible).
"""
def reduction_dim(data):
    numeric = data.select_dtypes(include=['float64', 'int64'])
    cmatrice = numeric.corr()
    # print(cmatrice)
    # On choisit de supprimer certaines classes (classes ayant une corrélation faible).
    # Ainsi que les classes non-utile.
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cmatrice, annot=True, cmap='coolwarm')
    # plt.show()
    data = data.drop("Num_Acc", axis=1)
    data = data.drop("num_veh", axis=1)
    data = data.drop("id_usa", axis=1)
    data = data.drop("ville", axis=1)
    data = data.drop("an_nais", axis=1)
    data = data.drop("place", axis=1)
    data = data.drop("week", axis=1)
    data = data.drop("departement", axis=1)
    # On vérifie de nouveau la corrélation entre les classes.
    numeric = data.select_dtypes(include=['float64', 'int64'])
    cmatrice = numeric.corr()
    # print(cmatrice)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cmatrice, annot=True, cmap='coolwarm')
    # plt.show()
    data = data.drop("hours", axis=1)
    data = data.drop("descr_agglo", axis=1)
    data = data.drop("descr_motif_traj", axis=1)
    data = data.drop("description_intersection", axis=1)
    data = data.drop("descr_type_col", axis=1)
    data = data.drop("descr_athmo", axis=1)
    data = data.drop("age", axis=1)
    data = data.drop("month", axis=1)
    # data = data.drop("descr_dispo_secu",axis=1)
    # On supprime également les colonnes non numériques n'apparaissant pas dans la matrice :
    data = data.drop("date", axis=1)
    data = data.drop("id_code_insee", axis=1)
    data = data.drop("time", axis=1)
    numeric = data.select_dtypes(include=['float64', 'int64'])
    cmatrice = numeric.corr()
    # print(cmatrice)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(cmatrice, annot=True, cmap='coolwarm')
    # print(data)
    # plt.show()
    # On peut par ailleurs utiliser l'algorithme random forest afin de comparer les résultats.
    return data


""" Reduction data Paul-Adrien PENET
Notre dataframe possède 73000 données environ, afin de réduire
le temps d'execution des algorithmes nous conservons uniquement 4000 données.
"""
def reduction_data(data):
    data_filtre = data.groupby('descr_grav').head(1000)
    # print(data_filtre)
    return data_filtre

# Interface des autres fonctions de préparation des données.
def data():
    data = rewrite_data()
    data_reduc_dim = reduction_dim(data)
    data_ready = reduction_data(data_reduc_dim)
    return data_ready

""" Repartition data Paul-Adrien PENET
"""
def repartition_data(data):
    # Dans la dataframe data_ready les données sont ordonnées à cause du groupby.
    # On répartit les données de façon aléatoire.
    #data = data()
    data_unsorted = data.sample(frac=1, random_state=0).reset_index(drop=True)
    # Séparation des données.
    X = data_unsorted
    y = data['descr_grav']
    return X, y


# data_reduc_dim = reduction_dim(data)
# print(data_reduc_dim)
# data_ready = reduction_data(data_reduc_dim)
# print(data_ready)
X, y = repartition_data(data())
# X = data_reduc_dim -> caractéristique
# y = data_reduc_dim['descr_grav'] -> variable cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# On choisit une valeur aléatoire pour random_state.

# ------------- Holdout ------------- :

#on initialise un objet de type LogisticRegression :
ho = LogisticRegression(max_iter=1000)
#On ajuste le modèle au donnée d'apprentissage :
ho.fit(X_train, y_train)

# On calcule les scores d'apprentissage et de test
Ho_apprentissage_score = ho.score(X_train, y_train)
HO_test_score = ho.score(X_test, y_test)

print("------Holdout-----")
print("Score apprentissage :", np.mean(Ho_apprentissage_score))
print("Score de test :", np.mean(HO_test_score))

# ------------- LeaveOneOut ------------- :

# On met à l'échelle la caractéristique X :
echelle = StandardScaler()
X_scaled = echelle.fit_transform(X)

# On initialise l'objet LeaveOneOut :
LeaveOO = LeaveOneOut()

# On initialise les listes de stockages des scores :
train_scores = []
test_scores = []

# On boucle N fois sur chaque K
for train_index, test_index in LeaveOO.split(X):
    X_train = X_scaled[train_index]
    X_test = X_scaled[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    #On instancie le modèle de régression logistique et on l'ajuste à nos données.
    loo = LogisticRegression(max_iter=1000)
    loo.fit(X_train, y_train)

    train_score = loo.score(X_train, y_train)
    test_score = loo.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

print("------LeaveOneOut-----")
print("score apprentissage :", np.mean(train_scores))
print("score test :", np.mean(test_scores))

# Classification avec trois algorithmes de "haut niveau" :

""" Classification : Paul-Adrien PENET
"""
def classification(type_methode,accident_info):

    #accident_info = json.loads(accident_info)
    accident_info_list = list(accident_info.values())
    print(accident_info_list)

    X, y = repartition_data(data())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #SVM :
    if type_methode == 0:
        clf = SVC(C=0.1)
        clf.fit(X_train, y_train)
        predictions = clf.predict([accident_info_list])
        resultat = {
            "SVM": predictions.tolist()
        }

    #Random Forest :
    elif type_methode == 1:
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        predictions = clf.predict([accident_info_list])
        resultat = {
            "Random Forest": predictions.tolist()
        }

    #MLP :
    elif type_methode == 2:
        clf = MLPClassifier(random_state=0)  # Paramètres au choix
        clf.fit(X_train, y_train)
        predictions = clf.predict([accident_info_list])
        resultat = {
            "Multiplayer Perceptron": predictions.tolist()
        }

    fichier_json = json.dumps(resultat)
    print(fichier_json)

    return fichier_json

# Evaluation quantitative des résultats "supervisé" : Taux d'apprentissage :

""" Support Vector Machine (SVM) : Paul-Adrien PENET 
"""
def classification_SVM():
    # On défini le paramètre de régularisation sur 1 comme point de départ
    # cette valeur peut être ajusté en fonction des résultats.
    # On crée une instance du modèle SVM puis on l'entraine.
    clf = SVC(C=0.1)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    score_precision = accuracy_score(y_test, prediction)
    # On utilise gridSearch pour trouver les meilleurs hyper-paramètres :
    # Il existe plusieurs paramètres pour le SVM, le paramètre C permet de réduire le bruit dans les observations.
    # parametre = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    # grid_search_result = GridSearchCV(clf, parametre, scoring='accuracy')
    # grid_search_result.fit(X_train, y_train)
    # print('Grid search resultat pour le paramètre C')
    # print(grid_search_result.cv_results_)
    # best_params = grid_search_result.best_params_
    # print('Meilleur paramètre', best_params)
    # best_score = grid_search_result.best_score_
    # print('Best score', best_score)
    return score_precision

X, y = repartition_data(data())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

""" Random Forest : Paul-Adrien PENET
"""
def classification_Random_Forest():
    clf2 = RandomForestClassifier(n_estimators=100, random_state=0)
    clf2.fit(X_train, y_train)
    prediction = clf2.predict(X_test)
    score_precision = accuracy_score(y_test, prediction)
    # On utilise gridSearch pour trouver les meilleurs hyper-paramètres :
    # parametre = {'n_estimators':[100, 200, 300]} # par défaut le nombre de n_estimators est égale à 100.
    # grid_search_result = GridSearchCV(param_grid=parametre)
    # grid_search_result.fit(X_train, y_train)
    # print('Grid search résultat pour le paramètre n_estimators :')
    # print(grid_search_result.cv_results_)
    # print('Meilleur paramètre', grid_search_result.best_params_)
    return score_precision

""" Multilayer Perceptron (MLP) : Paul-Adrien PENET
"""
def classification_MLP():
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=0)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    score_precision = accuracy_score(y_test, y_pred)
    # On utilise gridSearch pour trouver les meilleurs hyper-paramètres :
    # parametre = {'n_estimators':[100, 200, 300]} # par défaut le nombre de n_estimators est égale à 100.
    # grid_search_result = GridSearchCV(param_grid=parametre)
    # grid_search_result.fit(X_train, y_train)
    # print('Grid search résultat pour le paramètre n_estimators :')
    # print(grid_search_result.cv_results_)
    # print('Meilleur paramètre', grid_search_result.best_params_)
    return score_precision

""" Vote majoritaire : Paul-Adrien PENET
"""
def vote_majoritaire():
    # On initialise les classifieurs :
    svm = SVC()
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    mlp = MLPClassifier()

    # On initialise le vote majoritaire :
    vote = VotingClassifier(
        estimators=[('svm', svm), ('rf', rf), ('mlp', mlp)],
        voting='hard'
    )

    # On calcule et affiche les scores pour chaque classfieur :
    scores_svm = cross_val_score(svm, X, y, scoring='accuracy', cv=5)
    scores_rf = cross_val_score(rf, X, y, scoring='accuracy', cv=5)
    scores_mlp = cross_val_score(mlp, X, y, scoring='accuracy', cv=5)
    scores_vote = cross_val_score(vote, X, y, scoring='accuracy', cv=5)

    # Affichage des scores
    print("--------Vote Majoritaire--------")
    print("Précision SVM: %0.3f (+/- %0.3f)" % (scores_svm.mean(), scores_svm.std()))
    print("Précision Random Forest: %0.3f (+/- %0.3f)" % (scores_rf.mean(), scores_rf.std()))
    print("Précision Multilayer Perceptron: %0.3f (+/- %0.3f)" % (scores_mlp.mean(), scores_mlp.std()))
    print("Précision Ensemble: %0.3f (+/- %0.3f)" % (scores_vote.mean(), scores_vote.std()))
    return

# -------- TEST -------- :
score_precision = classification_SVM()
print("------SVM------")
print("Score précision :", score_precision)
# Pour un score des scores d'apprentissages et de test ~ 0.26
# On obtient un score de précision = 0.28375 (avant optimisation via gridsearch)

score_precision = classification_Random_Forest()
print("-------Random Forest------")
print("score précision :", score_precision)
# Pour un score des scores d'apprentissages et de test ~ 0.26
# On obtient un score de précision = 0.22 (avant optimisation via gridsearch)

score_precision = classification_MLP()
print("-------Multilayer Perceptron------")
print("score précision :", score_precision)
# Pour un score des scores d'apprentissages et de test ~ 0.26
# On obtient un score de précision = 0.265 (avant optimisation via gridsearch)

vote_majoritaire()

# Test de la classification :
print(data())
accident_info = {
  "latitude": 48.8566,
  "longitude": 2.3522,
  "descr_cat_veh": 1,
  "descr_lum": 1,
  "descr_etat_surf": 1,
  "descr_dispo_secu": 1,
  "descr_grav": 0
}
prediction_gravite = classification(1, accident_info)
print(prediction_gravite)