import pandas as pd

#On écrit le fichier data.csv dans un dataframe pandas.
data = pd.read_csv('data.csv')
print(data.head())
