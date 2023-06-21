import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import plotly.express as px
import plotly.graph_objects as go
from tmp_PA import data_ready


# Clustering k-means from scratch : Gabriel Lefèvre


# Fonction de calcul de distance : Haversine
def dist_haversine(long1, lat1, long2, lat2):
    r = 6371  # Rayon de la Terre en km
    # On converti les coordonnées en radian
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
    dlon = long2 - long1
    dlat = lat2 - lat1
    # On applique la formule de calcul de la fonction de haversine
    dist_haver = 2 * asin(sqrt(sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2))
    return dist_haver * r


# Fonction de calcul de distance : norme L1 (valeurs absolues)
def dist_L1(coord1, coord2):
    dlat_abs = abs(coord1['latitude']-coord2['latitude'])
    dlong_abs = abs(coord1['longitude']-coord2['longitude'])
    distance = dlat_abs + dlong_abs
    return distance


# Fonction de calcul de distance : norme L2 (euclidienne)
def dist_L2(coord1, coord2):
    dlat_squared = (coord1['latitude'] - coord2['latitude'])**2
    dlong_squared = (coord1['longitude'] - coord2['longitude'])**2
    distance = sqrt(dlat_squared + dlong_squared)
    return distance


def initialize_cluster_centers(data, nb_ofcluster):
    # Récupération aléatoire d'indice entre 0 et le dernier indice de data
    rand_idx = np.random.choice(range(len(data)), size=nb_ofcluster, replace=False)
    # Avec les indices récupérés, on obtient les coordonnées des centroïdes
    coord_centroids = data.iloc[rand_idx][['latitude', 'longitude']].values
    return coord_centroids


def assign_tocluster(data, coord_centroids):
    list_ofassign = []
    for i in range(len(data)):
        # On récupère les coordonnées de l'accident
        crash_coord = data.iloc[i][['latitude', 'longitude']].values
        distances = []
        for centroid in coord_centroids:
            # Calcul de la distance entre l'accident et chaque centroïde
            distances.append(dist_haversine(crash_coord[0], crash_coord[1], centroid[0], centroid[1]))
        # Ajout à une liste de l'indice de la plus petite distance correspondant au centroid de l'accident
        list_ofassign.append(np.argmin(distances))
    return list_ofassign


def upt_new_centroid(data, list_ofassign, nb_ofcluster):
    new_centroids = np.zeros((nb_ofcluster, 2))
    counts = np.zeros(nb_ofcluster)
    for i in range(len(data)):
        # On récupère les coordonnées de l'accident
        point = data.iloc[i][['latitude', 'longitude']].values
        # On récupère l'indexe du centroïde de l'accident
        cluster_index = list_ofassign[i]
        # On ajoute l'indexe et les coordonnées du points au tableau de mise à jour
        new_centroids[cluster_index] += point
        # On compte le changement à l'indice correspondant à celui du centroïde
        counts[cluster_index] += 1
    for i in range(nb_ofcluster):
        if counts[i] > 0:
            # On divise les données des nouveaux centroïdes par le nombre de point relié à ce dernier
            # Pour obtenir la moyenne des coordonnées reliées au centroïde
            new_centroids[i] /= counts[i]
    return new_centroids


def has_converged(old_centroids, new_centroids, factor_ofconvergence):
    # On calcule la différence entre les anciens et les nouveaux centroïdes
    dcentroid = np.abs(old_centroids - new_centroids)

    # On récupère la plus grande différence entre les anciens et les nouveaux centroïdes
    max_difference = np.max(dcentroid)
    # On teste si la différence maximale est inférieur au facteur de convergence
    # pour savoir si le programme doit se terminer
    if max_difference < factor_ofconvergence:
        return True
    else:
        return False


def k_means_from_scratch(data, nb_ofcluster):
    """
    Dans cette fonction on va réalisé les étapes d'un clustering k-means from scratch
    """
    # Step 1 Initialisation : On récupère les coordonnées des premiers centroïdes
    coord_centroids = initialize_cluster_centers(data, nb_ofcluster)
    # Step 2 Attribution : Chaque points de data est attribués à un centroïde
    list_ofassign = assign_tocluster(data, coord_centroids)
    # Step 3 Update : On boucle afin de réaliser le clustering jusqu'à que les données aient convergé
    end_loop = False
    while not end_loop:
        new_centroids = upt_new_centroid(data, list_ofassign, nb_ofcluster)
        list_ofassign = assign_tocluster(data, coord_centroids)
        # On vérifie la convergence afin de savoir si la boucle doit continuer
        end_loop = has_converged(coord_centroids, new_centroids, 0.001)
        coord_centroids = new_centroids
    return list_ofassign, coord_centroids


def aff_k_means_from_scratch(data, nb_ofcluster):
    # Réalisation du clustering
    list_ofassign, coord_centroids = k_means_from_scratch(data_ready, nb_ofcluster)

    # Affichage
    # On crée une nouvelle colonne dans le fichier data pour avoir les attributions de chaque accidents
    data['centroid'] = list_ofassign
    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", color="centroid",
                            mapbox_style="carto-positron", zoom=5, center={"lat": data['latitude'].mean(),
                                                                           "lon": data['longitude'].mean()})

    # Ajout des centroïdes à la carte
    lat_ofcentroids, lon_ofcentroids = zip(*coord_centroids)
    fig.add_trace(go.Scattermapbox(lat=lat_ofcentroids, lon=lon_ofcentroids, mode='markers',
                                   marker=dict(color='black', size=10)))
    fig.update_layout(title="Représentation des accidents de la route après clustering k-means from scratch")
    fig.show()


aff_k_means_from_scratch(data_ready, 14)