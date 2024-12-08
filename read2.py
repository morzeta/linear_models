import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import plotly.graph_objects as go
import statsmodels.api as sm
# Charger les données
from data_prep import read_data

x, y = read_data()

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modèle de régression
x_train_const = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train_const).fit()
y_pred = model.predict(add_constant(x_test))

# Calcul des résidus
residuals = y_test.to_numpy().flatten() - y_pred.to_numpy().flatten()

# (a) Vérification de la non-linéarité (graphiques des résidus)
residuals_fig = []
for column in x_test.columns:
    residuals_fig.append(go.Scatter(x=x_test[column], y=residuals, mode='markers', name=f"Résidus vs {column}"))

residuals_layout = go.Layout(title="Graphiques des Résidus", xaxis_title="Variable explicative", yaxis_title="Résidus")
residuals_plot = go.Figure(data=residuals_fig, layout=residuals_layout)

# (b) Valeurs aberrantes (Distance de Cook)
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

# (c) Multicolinéarité (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = x_train.columns
vif_data["VIF"] = [variance_inflation_factor(x_train_const.values, i + 1) for i in range(len(x_train.columns))]

# (d) Hétéroscédasticité (White)
white_test = het_white(model.resid, model.model.exog)
white_results = dict(zip(['Statistique LM', 'p-value', 'Statistique F', 'p-value F'], white_test))

# (e) Autocorrélation (Durbin-Watson)
dw_stat = durbin_watson(model.resid)

# Interprétation des tests
white_interpretation = (
    "Hétéroscédasticité détectée." if white_results['p-value'] < 0.05 else "Homoskédasticité détectée (p > 0.05)."
)
dw_interpretation = (
    "Pas d'autocorrélation détectée." if 1.5 < dw_stat < 2.5 else "Autocorrélation détectée."
)

# Générer le fichier HTML
html_content = f"""
<html>
<head>
    <title>Rapport de Vérification de Modèle</title>
</head>
<body>
    <h1>Rapport de Modélisation</h1>

    <h2>(a) Non-linéarité</h2>
    <div>{residuals_plot.to_html(full_html=False, include_plotlyjs='cdn')}</div>

    <h2>(b) Valeurs aberrantes</h2>
    <p>Distance de Cook : Les 5 plus élevées :</p>
    <pre>{pd.Series(cooks_d).sort_values(ascending=False).head(5).to_string()}</pre>

    <h2>(c) Multicolinéarité</h2>
    <p>Facteurs de variance d'inflation (VIF) :</p>
    <pre>{vif_data.to_string(index=False)}</pre>

    <h2>(d) Hétéroscédasticité</h2>
    <p>Résultats du test de White :</p>
    <pre>{white_results}</pre>
    <p>Interprétation : {white_interpretation}</p>

    <h2>(e) Autocorrélation</h2>
    <p>Statistique de Durbin-Watson : {dw_stat:.2f}</p>
    <p>Interprétation : {dw_interpretation}</p>
</body>
</html>
"""

with open("rapport_modelisation.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Rapport généré : rapport_modelisation.html")
