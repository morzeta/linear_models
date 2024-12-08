import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
import data_prep
x, y = data_prep.read_data()

# Ajouter une constante pour le modèle
x_with_const = sm.add_constant(x)

# Réaliser la régression initiale pour détecter l'hétéroskédasticité
model = sm.OLS(y, x_with_const).fit()
residuals = model.resid

# Test de White pour l'hétéroskédasticité
white_test = het_white(residuals, x_with_const)
labels = ['Statistique LM', 'p-valeur LM', 'Statistique F', 'p-valeur F']
white_results = dict(zip(labels, white_test))
print("\nRésultats du test de White :")
for key, value in white_results.items():
    print(f"{key}: {value:.4f}")

if white_results['p-valeur LM'] < 0.05:
    print("\nConclusion : Hétéroskédasticité détectée. Transformation des données en cours...")

    # Transformation logarithmique (par exemple sur y)
    y_log = np.log1p(y)  # log(1 + y) pour éviter les valeurs négatives

    # Régression avec les données transformées
    model_log = sm.OLS(y_log, x_with_const).fit()
    print("\nRésumé après transformation logarithmique :")
    print(model_log.summary())

    # Exporter les données transformées
    transformed_data = pd.concat([x, y_log.rename("y_log")], axis=1)
    transformed_data.to_csv("transformed_data.csv", index=False)
    print("\nDonnées transformées exportées vers 'transformed_data.csv'.")
else:
    print("\nConclusion : Pas d'hétéroskédasticité détectée.")
