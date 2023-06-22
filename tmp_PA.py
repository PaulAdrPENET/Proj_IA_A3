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

data = rewrite_data()

""" Réduction des dimensions Paul-Adrien PENET
On choisit de supprimer certaines classes (classes ayant une corrélation faible).
"""
def reduction_dim(data):
   numeric = data.select_dtypes(include=['float64', 'int64'])
   cmatrice = numeric.corr()
   #print(cmatrice)
   # On choisit de supprimer certaines classes (classes ayant une corrélation faible).
   # Ainsi que les classes non-utile.
   #plt.figure(figsize=(12, 10))
   #sns.heatmap(cmatrice, annot=True, cmap='coolwarm')
   #plt.show()
   data = data.drop("Num_Acc", axis=1)
   data = data.drop("num_veh", axis=1)
   data = data.drop("id_usa", axis=1)
   data = data.drop("ville", axis=1)
   data = data.drop("an_nais", axis=1)
   data = data.drop("place", axis=1)
   data = data.drop("week", axis=1)
   data = data.drop("departement", axis=1)
   #On vérifie de nouveau la corrélation entre les classes.
   numeric = data.select_dtypes(include=['float64', 'int64'])
   cmatrice = numeric.corr()
   #print(cmatrice)
   #plt.figure(figsize=(12, 10))
   #sns.heatmap(cmatrice, annot=True, cmap='coolwarm')
   #plt.show()
   data = data.drop("hours", axis=1)
   data = data.drop("descr_agglo", axis=1)
   data = data.drop("descr_motif_traj", axis=1)
   data = data.drop("description_intersection", axis=1)
   data = data.drop("descr_type_col", axis=1)
   data = data.drop("descr_lum", axis=1)
   data = data.drop("age", axis=1)
   data = data.drop("month", axis=1)
   #On supprime également les colonnes non numériques n'apparaissant pas dans la matrice :
   data = data.drop("date", axis = 1)
   data = data.drop("id_code_insee", axis = 1)
   data = data.drop("time", axis = 1)
   numeric = data.select_dtypes(include=['float64', 'int64'])
   cmatrice = numeric.corr()
   #print(cmatrice)
   #plt.figure(figsize=(12, 10))
   #sns.heatmap(cmatrice, annot=True, cmap='coolwarm')
   #print(data)
   #plt.show()
   # Forest test (comparer les résultats).
   return data

""" Notre dataframe possède 73000 données environ, afin de réduire
le temps d'execution des algorithmes nous conservons uniquement 4000 données.
"""
def reduction_data(data):
    data_filtre = data.groupby('descr_grav').head(1000)
    #print(data_filtre)
    return data_filtre
# Cependant les données étant regroupées par description grave, il faut les mélanger.

def repartition_data(data):
   #Dans la dataframe data_ready les données sont ordonnées à cause du groupby.
   #On répartit les données de façon aléatoire.
   data_unsorted = data.sample(frac=1,random_state=0).reset_index(drop=True)
   #Séparation des données.
   X = data_unsorted
   y = data['descr_grav']
   return X,y


data_reduc_dim = reduction_dim(data)
#print(data_reduc_dim)
data_ready = reduction_data(data_reduc_dim)
print(data_ready)
X, y = repartition_data(data_ready)
#X = data_reduc_dim
#y = data_reduc_dim['descr_grav']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#On choisit une valeur aléatoire pour random_state.
#Holdout :

ho = LogisticRegression(max_iter=1000)
ho.fit(X_train, y_train)

#On calcule les scores d'apprentissage et de test
Ho_apprentissage_score = ho.score(X_train, y_train)
HO_test_score = ho.score(X_test, y_test)

print("------Holdout-----")
print("Score apprentissage :", np.mean(Ho_apprentissage_score))
print("Score de test :", np.mean(HO_test_score))

#LeaveOneOut :

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

LeaveOO = LeaveOneOut()

train_scores = []
test_scores = []

for train_index, test_index in LeaveOO.split(X):
    X_train = X_scaled[train_index]
    X_test = X_scaled[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    loo = LogisticRegression(max_iter=1000)
    loo.fit(X_train, y_train)

    train_score = loo.score(X_train, y_train)
    test_score = loo.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

print("------LeaveOneOut-----")
print("score apprentissage :", np.mean(train_scores))
print("score test :", np.mean(test_scores))

#Classification avec trois algorithmes de "haut niveau" :

#Support Vector Machine (SVM) :

svm = SVC(kernel='linear', C=1.0)
#On défini le paramètre de régularisation sur 1 comme point de départ
#cette valeur peut être ajusté en fonction des résultats.
X, y = repartition_data(data_ready)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = SVC()
clf.fit(X_train, y_train)
#On crée une instance du modèle SVM puis on l'entraine.
prediction = clf.predict(X_test)
score_precision = accuracy_score(y_test, prediction)
#On utilise gridSearch pour trouver les meilleurs hyper-paramètres :
#Il existe plusieurs paramètres pour le SVM, le paramètre C permet de réduire le bruit dans les observations.

parametre = { 'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
grid_search_result = GridSearchCV(clf, parametre, scoring='accuracy')
grid_search_result.fit(X_train, y_train)
print('Grid search resultat pour le paramètre C')
print(grid_search_result.cv_results_)
best_params = grid_search_result.best_params_
print('Meilleur paramètre', best_params)
best_score = grid_search_result.best_score_
print('Best score', best_score)
print("------SVM------")
print("Score précision :", score_precision)
#Pour un score des scores d'apprentissages et de test ~ 0.26
#On obtient un score de précision = 0.28375 (avant optimisation via gridsearch)

#Ransom Forest :

clf2 = RandomForestClassifier(n_estimators=100, random_state=0)
clf2.fit(X_train, y_train)
prediction = clf2.predict(X_test)
score_precision = accuracy_score(y_test, prediction)
#On utilise gridSearch pour trouver les meilleurs hyper-paramètres :
#parametre = {'n_estimators':[100, 200, 300]} # par défaut le nombre de n_estimators est égale à 100.
#grid_search_result = GridSearchCV(param_grid=parametre)
#grid_search_result.fit(X_train, y_train)
#print('Grid search résultat pour le paramètre n_estimators :')
#print(grid_search_result.cv_results_)
#print('Meilleur paramètre', grid_search_result.best_params_)

print("-------Random Forest------")
print("score précision :", score_precision)
#Pour un score des scores d'apprentissages et de test ~ 0.26
#On obtient un score de précision = 0.22 (avant optimisation via gridsearch)


#Multilayer Perceptron (MLP) :

mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=0)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
score_precision = accuracy_score(y_test, y_pred)

print("-------Multilayer Perceptron------")
print("score précision :", score_precision)
#Pour un score des scores d'apprentissages et de test ~ 0.26
#On obtient un score de précision = 0.265 (avant optimisation via gridsearch)
