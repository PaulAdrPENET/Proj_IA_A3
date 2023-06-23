from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from tmp_PA import data_ready


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
    return list_ofassign, coord_centroids


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


aff_k_means_with_sklearn(8)
