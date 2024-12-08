import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_prep import read_data
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.stats.diagnostic import het_breuschpagan
# Charger et nettoyer les données
x, y = read_data()

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Créer un modèle de régression linéaire
model = LinearRegression()

# Ajuster le modèle sur les données d'entraînement
model.fit(x_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(x_test)

# Évaluer les performances du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Évaluation du modèle :")
print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
print(f"Score R² : {r2:.2f}")

# Afficher un aperçu des prédictions
predictions = pd.DataFrame({'Réel': y_test.values.flatten(), 'Prédictions': y_pred.flatten()})
print(predictions)



# Calculer les résidus
residuals = y_test.values.flatten() - y_pred.flatten()
print(f"Residuals size: {len(residuals)}")
print(f"x_test size: {len(x_test)}")
# Tracer les résidus contre une ou plusieurs variables explicatives

for column in x_test.columns:
    plt.scatter(x_test[column], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel(column)
    plt.ylabel("résidus")
    plt.title(f"Résidus vs {column}")
    plt.show()



# Fit the model
x_with_const = sm.add_constant(x_train)
model = sm.OLS(y_train, x_with_const).fit()

# Summary of the model
print(model.summary())
#b) outliers
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
# Example: Breusch-Pagan Test for heteroskedasticity
residuals = model.resid
bp_test = het_breuschpagan(residuals, x_with_const)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
result = dict(zip(labels, bp_test))

# Afficher les résultats
print("\nRésultats du test de Breusch-Pagan pour l'homoskédasticité :")
for key, value in result.items():
    print(f"{key}: {value:.4f}")

# Interprétation claire
p_value = result['p-value']
if p_value < 0.05:
    print("\nConclusion : Les résultats suggèrent une hétéroscédasticité (p-valeur < 0.05).")
    print("Cela signifie que la variance des résidus n'est pas constante.")
else:
    print("\nConclusion : Les résultats suggèrent une homoskédasticité (p-valeur >= 0.05).")
    print("Cela signifie que la variance des résidus est constante.")