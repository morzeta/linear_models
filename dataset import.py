from ucimlrepo import fetch_ucirepo

# fetch dataset
air_quality = fetch_ucirepo(id=360)

# data (as pandas dataframes)
X = air_quality.data.features
y = air_quality.data.targets

# metadata
print(air_quality.metadata)

# variable information
print(air_quality.variables)

import pandas as pd

filename = "AirQualityUCI.csv"
try:
    datas = pd.read_csv(filename)  # Utilisez read_csv pour les fichiers CSV
    print(datas.head(10))  # Affiche les 10 premières lignes
except FileNotFoundError as e:
    print(f"Le fichier {filename} n'a pas été trouvé. {e}")
except pd.errors.EmptyDataError:
    print(f"Le fichier {filename} est vide.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")

