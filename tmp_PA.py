import pandas as pd
from Preparation import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
   data = data.drop("descr_dispo_secu", axis=1)
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
   data_unsorted = data.sample(frac=1,random_state=1).reset_index(drop=True)
   # Holdout
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
#On choisit une valeur aléatoire pour random_state.
#Holdout :

holdout = LogisticRegression(max_iter=20000)
holdout.fit(X_train, y_train)

#On calcule les scores d'apprentissage et de test
holdout_train_score = holdout.score(X_train, y_train)
holdout_test_score = holdout.score(X_test, y_test)

print("------Holdout-----")
print("Score apprentissage :", np.mean(holdout_train_score))
print("Score de test :", np.mean(holdout_test_score))

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

    loo = LogisticRegression(max_iter=20000)
    loo.fit(X_train, y_train)

    train_score = loo.score(X_train, y_train)
    test_score = loo.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

print("------LeaveOneOut-----")
print("score apprentissage :", np.mean(train_scores))
print("score test :", np.mean(test_scores))



