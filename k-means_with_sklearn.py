from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from tmp_PA import data_ready
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

# Clustering k-means avec la bibliothèque scikit-learn : Gabriel Lefèvre


def k_means_with_sklearn(nb_ofcentroid):
    # On crée un modèle contenant les méthodes de clustering k-means
    k_means = KMeans(n_clusters=nb_ofcentroid)
    # Entrainement du modèle avec nos données
    k_means.fit(data_ready)
    # On récupère les attributions des points
    list_ofassign = k_means.labels_
    # On récupère les coordonnées de chaque centroïdes
    coord_centroids = k_means.cluster_centers_

    #calcul des coéfficients
    silhouette = silhouette_score(data_ready, list_ofassign)
    calinski_harabasz = calinski_harabasz_score(data_ready, list_ofassign)
    davies_bouldin = davies_bouldin_score(data_ready, list_ofassign)


    return list_ofassign, coord_centroids, silhouette, calinski_harabasz, davies_bouldin


def aff_k_means_with_sklearn(nb_ofcentroid):
    # On réalise le clustering avec la bibliothèque scikit-learn
    list_ofassign, coord_centroids = k_means_with_sklearn(nb_ofcentroid)
    # Affichage
    # On crée une nouvelle colonne dans le fichier data pour avoir les attributions de chaque accidents
    data_ready['centroid'] = list_ofassign
    fig = px.scatter_mapbox(data_ready, lat="latitude", lon="longitude", color="centroid",
                            mapbox_style="carto-positron", zoom=5, center={"lat": data_ready['latitude'].mean(),
                                                                           "lon": data_ready['longitude'].mean()})

    # Ajout des centroïdes à la carte
    lat_ofcentroids = [centroid[0] for centroid in coord_centroids]
    lon_ofcentroids = [centroid[1] for centroid in coord_centroids]
    fig.add_trace(go.Scattermapbox(lat=lat_ofcentroids, lon=lon_ofcentroids, mode='markers',
                                   marker=dict(color='#00FF00', size=10)))
    fig.update_layout(title="Représentation des accidents de la route après clustering k-means avec sklearn")
    fig.show()


# aff_k_means_with_sklearn(5)


#fonction qui affiche le coefficient silhouette dans un graph avec le nombre de centroïde variant de 2 à 15
def silhouette():
    clusters_silhouette = []
    for i in range(2, 15):
        list_ofassign, coord_centroids, silhouette, calinski_harabasz, davies_bouldin = k_means_with_sklearn(i)# on récupère le coefficient silhouette
        clusters_silhouette.append(silhouette)
    plt.plot(range(2, 15), clusters_silhouette, 'x-', color="green") #affichage du graph
    plt.title("coefficient silhouette")
    plt.show()
silhouette()

#silhouette() # appel pour le graph du coefficient silhouette

def Calinski_Harabasz():
    clusters_Calinski_Harabasz = []
    for i in range(2, 15):
        list_ofassign, coord_centroids, silhouette, calinski_harabasz, davies_bouldin = k_means_with_sklearn(i)  # on récupère la valeure de calinski_harabasz
        clusters_Calinski_Harabasz.append(calinski_harabasz)
    plt.plot(range(2, 15), clusters_Calinski_Harabasz, 'x-', color="green")
    plt.title("Indice de Calinski-Harabasz")
    plt.show()
Calinski_Harabasz()

def Davies_Bouldin():
    clusters_Davies_Bouldin = []
    for i in range(2, 15):
        list_ofassign, coord_centroids, silhouette, calinski_harabasz, davies_bouldin = k_means_with_sklearn(i)  # on récupère la valeure de Davies_Bouldin
        clusters_Davies_Bouldin.append(davies_bouldin)
    plt.plot(range(2, 15), clusters_Davies_Bouldin, 'x-', color="green")
    plt.title("Indice de Davies-Bouldin")
    plt.show()

Davies_Bouldin()