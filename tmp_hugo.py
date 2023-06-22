import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from tmp_PA import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve


# Préparation des données Paul-Adrien PENET.
data_reduc = reduction_dim(data)

#On écrit le fichier data.csv dans un dataframe pandas.
data = rewrite_data()

#fonction qui représente la valeur cible, c'est à dire le nombre d'accident par gravité
def graphique_cible():
    counts = data_reduc['descr_grav'].value_counts()
    plt.bar(counts.index, counts.values)
    plt.xlabel('Description de la gravité')
    plt.ylabel('Nombre accidents')
    plt.show()
#graphique_cible()


#fonction qui affiche Nombre d’instances
def graphique_nombre_instances():
    data_reduc.hist(bins=50, figsize=(20, 15))
    plt.show()

#graphique_nombre_instances()
#fonction qui affiche Nombre d’instances par classe avec en parametre les deux colonnes à comparer
def graphique_nombre_instances_par_classe(x,y):
    table=pd.crosstab(x,y)
    table.plot.bar(stacked=True)
    plt.show()
#graphique_nombre_instances_par_classe(data_reduc['descr_grav'], data_reduc['descr_cat_veh']) #descr_dispo_secu, descr_etat_surf, descr_athmo, descr_cat_veh

def graphique_taille_feature():
    features = data_reduc.columns.tolist()
    taille = [len(data_reduc[col]) for col in features]
    plt.bar(features,taille)
    plt.show()

#graphique_taille_feature()

#carte :
def affiche_heatmap():
    data_grav_change = data_reduc
    data_grav_change['descr_grav'] = data_grav_change['descr_grav'].replace({1: 4, 2: 1, 4: 2})

    fig = px.density_mapbox(data_grav_change, lat='latitude', lon='longitude', z='descr_grav', radius=8,
                            center=dict(lat=46.603354, lon=1.888334), zoom=4,
                            mapbox_style="stamen-terrain", color_continuous_scale='Viridis')
    fig.update_layout(coloraxis_colorbar=dict(title='gravité des accidents <br> de tué à indemne'))

    fig.show()



#affichage des 4000 données
data_filtre = data.groupby('descr_grav').head(1000)
#print(data_filtre['descr_grav'])

data_ready = reduction_data(data_reduc_dim)
X, y = repartition_data(data_ready)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

knn = KNeighborsClassifier(n_neighbors=7)
scores = cross_val_score(knn, X_train, y_train, cv=5)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#print("Précision : {:.2f}%".format(accuracy * 100))
#print("Précision moyenne : {:.2f}%".format(scores.mean() * 100))#affichage du pourcentage

#étape 4 :
#Évaluation quantitative des résultats « non supervisé » :
#calcul coefficient silhouette
def silhouette():
    clusters_silhouette = []
    for i in range(2, 15):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        labels = kmeans.labels_
        silhouette = silhouette_score(X_train, labels)
        clusters_silhouette.append(silhouette)
    plt.plot(range(2, 15), clusters_silhouette, 'x-', color="green")
    plt.title("coefficient silhouette")
    plt.show()
#print("Coefficient Silhouette : {:.4f}".format(silhouette))
silhouette()
#calcul Calinski-Harabasz Index
def Calinski_Harabasz():
    clusters_Calinski_Harabasz = []
    for i in range(2, 15):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        labels = kmeans.labels_
        calinski_harabasz = calinski_harabasz_score(X_train, labels)
        clusters_Calinski_Harabasz.append(calinski_harabasz)
    plt.plot(range(2, 15), clusters_Calinski_Harabasz, 'x-', color="green")
    plt.title("Indice de Calinski-Harabasz")
    plt.show()

#print("Indice de Calinski-Harabasz : {:.4f}".format(calinski_harabasz))

#calcul Davies-Bouldin Index
def Davies_Bouldin():
    clusters_Davies_Bouldin = []
    for i in range(2, 15):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X_train)
        labels = kmeans.labels_
        davies_bouldin = davies_bouldin_score(X_train, labels)
        clusters_Davies_Bouldin.append(davies_bouldin)
    plt.plot(range(2, 15), clusters_Davies_Bouldin, 'x-', color="green")
    plt.title("Indice de Davies-Bouldin")
    plt.show()

#print("Indice de Davies-Bouldin : {:.4f}".format(davies_bouldin))


#Évaluation quantitative des résultats « supervisé » :
#calcul taux d'apprentissage
def taux_apprentissage():
    taux_apprentissage = accuracy_score(y_test, y_pred)
    print("Taux d'apprentissage : {:.2f}%".format(taux_apprentissage * 100))

def Matrice_de_confusion():
    matrice_confusion = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :")
    print(matrice_confusion)

def precision_rappel():
    precision = precision_score(y_test, y_pred)
    rappel = recall_score(y_test, y_pred)

    print("Précision : {:.2f}".format(precision))
    print("Rappel : {:.2f}".format(rappel))

def Courbe_ROC():
    fpr, tpr, seuils = roc_curve(y_test, scores)# score correspond au score de précision du model
    #fpr : taux de faux positif
    #tpr : taux de vrais positif

    #affichage du graph
    plt.plot(fpr, tpr)
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.show()#