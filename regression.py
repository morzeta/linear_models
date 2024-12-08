import data_prep
import statsmodels.api as sm

# Ajouter une constante pour l'interception
X = sm.add_constant(data_prep.read_data())

# Créer le modèle
model = sm.OLS(data_prep.read_data(), X)
results = model.fit()

# Résumé des résultats
print(results.summary())