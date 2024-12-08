import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data_prep import read_data
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import plotly.graph_objects as go
from IPython.core.display import HTML

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

# Calculer les résidus
residuals = y_test.values.flatten() - y_pred.flatten()

# Résumé avec statsmodels
x_with_const = sm.add_constant(x_train)
model_ols = sm.OLS(y_train, x_with_const).fit()

# Test de Breusch-Pagan
bp_test = het_breuschpagan(model_ols.resid, x_with_const)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
result_bp = dict(zip(labels, bp_test))

# Résultat et interprétation du test de Breusch-Pagan
bp_interpretation = ""
if result_bp['p-value'] < 0.05:
    bp_interpretation = "Les résultats suggèrent une hétéroscédasticité (p-valeur < 0.05). La variance des résidus n'est pas constante."
else:
    bp_interpretation = "Les résultats suggèrent une homoskédasticité (p-valeur >= 0.05). La variance des résidus est constante."

# Créer les graphiques interactifs
residuals_plot = []
for column in x_test.columns:
    trace = go.Scatter(x=x_test[column], y=residuals, mode='markers', name=f"Résidus vs {column}")
    residuals_plot.append(trace)

layout = go.Layout(title="Graphiques des résidus", xaxis_title="Variable explicative", yaxis_title="Résidus",
                   height=500, width=900)
fig = go.Figure(data=residuals_plot, layout=layout)

# Sauvegarder les graphiques et résultats dans un fichier HTML
html_content = f"""
<html>
<head>
    <title>Rapport de régression</title>
</head>
<body>
    <h1>Rapport de Modélisation</h1>
    <h2>Évaluation du Modèle</h2>
    <p><b>Erreur quadratique moyenne (MSE) :</b> {mse:.2f}</p>
    <p><b>Score R² :</b> {r2:.2f}</p>

    <h2>Test de Breusch-Pagan</h2>
    <p><b>Lagrange multiplier statistic :</b> {result_bp['Lagrange multiplier statistic']:.4f}</p>
    <p><b>p-value :</b> {result_bp['p-value']:.4f}</p>
    <p><b>f-value :</b> {result_bp['f-value']:.4f}</p>
    <p><b>f p-value :</b> {result_bp['f p-value']:.4f}</p>
    <p><b>Interprétation :</b> {bp_interpretation}</p>

    <h2>Graphiques des Résidus</h2>
    <div>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
</body>
</html>
"""

# Sauvegarder dans un fichier HTML
with open("rapport_regression.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Rapport généré : rapport_regression.html")