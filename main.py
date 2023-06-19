import pandas as pd

# Préparation des données Paul-Adrien PENET.

#On écrit le fichier data.csv dans un dataframe pandas.
data = pd.read_csv('data.csv')
#print(data.head())
types = data.dtypes
#print(types)
#On remarque que les dates sont au format object, et les numéros des départements au format object également.
data['date'] = pd.to_datetime(data['date'])
types = data.dtypes
#print(types)
print(data['date'])
#On sépare la date et le temps en deux colonnes :
data['time'] = data['date'].dt.time
data['date'] = data['date'].dt.date
print(data['date'],data['time'])
# On regarde si il y a des valeurs manquantes dans notre dataframe.
print(data.isna().any())
print(data.isnull().any())
# On corrige la valeur de l'âge (en diminuant tout par 14 ans) :
data['age'] = data['age'] - 14
print(data['age'].mean())


