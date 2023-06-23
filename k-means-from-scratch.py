import numpy as np
from math import radians, cos, sin, asin, sqrt
import plotly.express as px
import plotly.graph_objects as go
from tmp_PA import data_ready


# Clustering k-means from scratch : Gabriel Lefèvre

class KMeansFromScratch:
    def __init__(self, nb_of_clusters):
        # Initialisation des variables
        self.nb_of_clusters = nb_of_clusters
        self.coord_centroids = None
        self.list_of_assign = None

    # Fonction de calcul de distance : Haversine
    @staticmethod
    def haversine_distance(long1, lat1, long2, lat2):
        r = 6371  # Rayon de la Terre en km
        # On converti les coordonnées en radian
        long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
        dlon = long2 - long1
        dlat = lat2 - lat1
        # On applique la formule de calcul de la fonction de haversine
        dist_haver = 2 * asin(sqrt(sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2))
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

    def initialize_cluster_centers(self, data):
        # Récupération aléatoire d'indice entre 0 et le dernier indice de data
        rand_idx = np.random.choice(range(len(data)), size=self.nb_of_clusters, replace=False)
        # Avec les indices récupérés, on obtient les coordonnées des centroïdes
        self.coord_centroids = data.iloc[rand_idx][['latitude', 'longitude']].values

    def assign_to_cluster(self, data):
        self.list_of_assign = []
        for i in range(len(data)):
            # On récupère les coordonnées de l'accident
            crash_coord = data.iloc[i][['latitude', 'longitude']].values
            distances = []
            for centroid in self.coord_centroids:
                # Calcul de la distance entre l'accident et chaque centroïde
                distances.append(self.dist_L2((crash_coord[0], crash_coord[1]), (centroid[0], centroid[1])))
            # Ajout à une liste de l'indice de la plus petite distance correspondant au centroid de l'accident
            self.list_of_assign.append(np.argmin(distances))

    def update_new_centroids(self, data):
        new_centroids = np.zeros((self.nb_of_clusters, 2))
        counts = np.zeros(self.nb_of_clusters)
        for i in range(len(data)):
            # On récupère les coordonnées de l'accident
            point = data.iloc[i][['latitude', 'longitude']].values
            # On récupère l'indexe du centroïde de l'accident
            cluster_index = self.list_of_assign[i]
            # On ajoute l'indexe et les coordonnées du points au tableau de mise à jour
            new_centroids[cluster_index] += point
            # On compte le changement à l'indice correspondant à celui du centroïde
            counts[cluster_index] += 1
        for i in range(self.nb_of_clusters):
            if counts[i] > 0:
                # On divise les données des nouveaux centroïdes par le nombre de point relié à ce dernier
                # Pour obtenir la moyenne des coordonnées reliées au centroïde
                new_centroids[i] /= counts[i]
        return new_centroids

    @staticmethod
    def has_converged(old_centroids, new_centroids, convergence_factor):
        # On calcule la différence entre les anciens et les nouveaux centroïdes
        d_centroid = np.abs(old_centroids - new_centroids)
        # On récupère la plus grande différence entre les anciens et les nouveaux centroïdes
        max_difference = np.max(d_centroid)
        # On teste si la différence maximale est inférieur au facteur de convergence
        # pour savoir si le programme doit se terminer
        return max_difference < convergence_factor

    def k_means_from_scratch(self, data):
        """
            Dans cette fonction on va réaliser les étapes d'un clustering k-means from scratch
        """
        # Step 1 Initialisation : On récupère les coordonnées des premiers centroïdes
        self.initialize_cluster_centers(data)
        # Step 2 Attribution : Chaque points de data est attribués à un centroïde
        self.assign_to_cluster(data)
        # Step 3 Update : On boucle afin de réaliser le clustering jusqu'à que les données aient convergé
        end_loop = False
        while not end_loop:
            new_centroids = self.update_new_centroids(data)
            self.assign_to_cluster(data)
            # On vérifie la convergence afin de savoir si la boucle doit continuer
            end_loop = self.has_converged(self.coord_centroids, new_centroids, 0.001)
            self.coord_centroids = new_centroids

    def aff_k_means_from_scratch(self, data):
        # On crée une nouvelle colonne dans le fichier data pour avoir les attributions de chaque accidents
        data['centroid'] = self.list_of_assign
        fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", color="centroid",
                                mapbox_style="carto-positron", zoom=5,
                                center={"lat": data['latitude'].mean(), "lon": data['longitude'].mean()})

        # Ajout des centroïdes à la carte
        lat_of_centroids, lon_of_centroids = zip(*self.coord_centroids)
        fig.add_trace(go.Scattermapbox(lat=lat_of_centroids, lon=lon_of_centroids, mode='markers',
                                       marker=dict(color='#00FF00', size=10)))
        fig.update_layout(title="Représentation des accidents de la route après clustering k-means from scratch")
        fig.show()


# Instanciation de l'objet KMeansFromScratch
kmeans = KMeansFromScratch(nb_of_clusters=9)
# Utilisation de la méthode pour réaliser le clustering
kmeans.k_means_from_scratch(data_ready)
# Utilisation de la méthode pour afficher le clustering sur une carte
kmeans.aff_k_means_from_scratch(data_ready)
#
