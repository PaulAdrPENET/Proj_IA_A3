import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Préparation des données Paul-Adrien PENET.


def rewrite_data(data):
    types = data.dtypes
    # print(types)
    # On remarque que les dates sont au format object, et les numéros des départements au format object également.
    data['date'] = pd.to_datetime(data['date'])
    types = data.dtypes
    # print(types)
    # print(data['date'])
    # On sépare la date et le temps en deux colonnes :
    data['time'] = data['date'].dt.time
    data['date'] = data['date'].dt.date
    # print(data['date'], data['time'])
    # On regarde si il y a des valeurs manquantes dans notre dataframe.
    # print(data.isna().any())
    # print(data.isnull().any())
    # On corrige la valeur de l'âge (en diminuant tout par 14 ans) :
    data['age'] = data['age'] - 14
    # print(data['age'].mean())

# Préparation Gabriel Lefèvre


"""
print(data['descr_grav'])
cath_quali = [data['descr_cat_veh'], data['descr_agglo'],
              data['descr_athmo'], data['descr_lum'],
              data['descr_etat_surf'], data['description_intersection'],
              data['descr_dispo_secu'], data['descr_grav'],
              data['descr_motif_traj'], data['descr_type_col']]

fig, (ax1, ax2) = plt.subplots(2, 5)
for cl in cath_quali:
    recurrence = Counter(cl)
    name = list(recurrence.keys())
    nb = list(recurrence.values())
    plt.bar(name, nb)
plt.xlabel("Instances")
plt.ylabel("Nombre d'accident ")
plt.title('Récurrences de chaque instances')
plt.tight_layout()
plt.show()
"""
