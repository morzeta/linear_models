import pandas as pd
import numpy as np
filename = "AirQualityUCI.csv"
try:
    datas = pd.read_csv(filename, sep = ";")  # Utilisez read_csv pour les fichiers CSV
    rand = np.random.rand(900)
    print(rand)# Affiche les 10 premières lignes
    datas.head(900).to_csv("date_test", sep = ";" ,index= True)
except FileNotFoundError as e:
    print(f"Le fichier {filename} n'a pas été trouvé. {e}")
except pd.errors.EmptyDataError:
    print(f"Le fichier {filename} est vide.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")




