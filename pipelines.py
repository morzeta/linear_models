import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from statsmodels.api import add_constant
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from preparation import read_data


def ordinary_linear_regression_v1():
    # Load and clean data
    x, y = read_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit the model
    x_with_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_with_const).fit()

    # Summary of the model
    print(model.summary())
    # outliers
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    # Example: Breusch-Pagan Test for heteroskedasticity
    residuals = model.resid
    bp_test = het_breuschpagan(residuals, x_with_const)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    result = dict(zip(labels, bp_test))

    # Show results
    print("\nResults of the Breusch-Pagan test for homoskedasticity :")
    for key, value in result.items():
        print(f"{key}: {value:.4f}")

    # Clear interpretation
    p_value = result['p-value']
    if p_value < 0.05:
        print("\nConclusion: The results suggest homoskedasticity (p-value < 0.05).")
        print("This means that the variance of the residuals is not constant.")
    else:
        print("\nConclusion: The results suggest homoskedasticity (p-value >= 0.05).")
        print("This means that the variance of the residuals is constant.")


def linear_regression_v1():
    # Load and clean data
    x, y = read_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(x_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model evaluation :")
    print(f"Mean Square Error (MSE) : {mse:.2f}")
    print(f"Score R² : {r2:.2f}")

    # Show Preview of Predictions
    predictions = pd.DataFrame({'Target': y_test.values.flatten(), 'Predictions': y_pred.flatten()})
    print(predictions)

    # Calculate the residuals
    residuals = y_test.values.flatten() - y_pred.flatten()
    print(f"Residuals size: {len(residuals)}")
    print(f"x_test size: {len(x_test)}")

    # Plot the residuals against one or more explanatory variables
    for column in x_test.columns:
        plt.scatter(x_test[column], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        plt.xlabel(column)
        plt.ylabel("residues")
        plt.title(f"Residues vs {column}")
        plt.show()


def linear_regression_v2():
    # Load and clean data
    x, y = read_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(x_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate the residuals
    residuals = y_test.values.flatten() - y_pred.flatten()

    # Create interactive charts
    residuals_plot = []
    for column in x_test.columns:
        trace = go.Scatter(x=x_test[column], y=residuals, mode='markers', name=f"Residues vs {column}")
        residuals_plot.append(trace)

    layout = go.Layout(title="Residual graphs", xaxis_title="Explanatory variable", yaxis_title="Residues",
                       height=500, width=900)
    fig = go.Figure(data=residuals_plot, layout=layout)

    # Summary with statsmodels
    x_with_const = sm.add_constant(x_train)
    model_ols = sm.OLS(y_train, x_with_const).fit()

    # Test the Breusch-Pagan
    bp_test = het_breuschpagan(model_ols.resid, x_with_const)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    result_bp = dict(zip(labels, bp_test))

    # Result and interpretation of the Breusch-Pagan test
    bp_interpretation = ""
    if result_bp['p-value'] < 0.05:
        bp_interpretation = "The results suggest heteroskedasticity (p-value < 0.05)." \
                            " The variance of the residuals is not constant."
    else:
        bp_interpretation = "The results suggest homoskedasticity (p-value >= 0.05)." \
                            " The variance of the residuals is constant."

    # Save graphs and results to an HTML file
    html_content = f"""
    <html>
    <head>
        <title>Regression report</title>
    </head>
    <body>
        <h1>Modeling Report</h1>
        <h2>Model Evaluation</h2>
        <p><b>Mean Square Error (MSE) :</b> {mse:.2f}</p>
        <p><b>Score R² :</b> {r2:.2f}</p>

        <h2>Test the Breusch-Pagan</h2>
        <p><b>Lagrange multiplier statistic :</b> {result_bp['Lagrange multiplier statistic']:.4f}</p>
        <p><b>p-value :</b> {result_bp['p-value']:.4f}</p>
        <p><b>f-value :</b> {result_bp['f-value']:.4f}</p>
        <p><b>f p-value :</b> {result_bp['f p-value']:.4f}</p>
        <p><b>Interpretation :</b> {bp_interpretation}</p>

        <h2>Residual Graphs</h2>
        <div>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
    </body>
    </html>
    """

    # Save to HTML file
    with open("regression report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Report generated : regression report.html")


def linear_regression_v3():
    # Load data
    x, y = read_data()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Regression model
    x_train_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_train_const).fit()
    y_pred = model.predict(add_constant(x_test))

    # Calculation of residuals
    residuals = y_test.to_numpy().flatten() - y_pred.to_numpy().flatten()

    # Checking for non-linearity (residual plots)
    residuals_fig = []
    for column in x_test.columns:
        residuals_fig.append(go.Scatter(x=x_test[column], y=residuals, mode='markers', name=f"Residues vs {column}"))

    residuals_layout = go.Layout(title="Residual Graphs", xaxis_title="Explanatory variable", yaxis_title="Residues")
    residuals_plot = go.Figure(data=residuals_fig, layout=residuals_layout)

    # Outliers (Cook Distance)
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Multi-co-linearity (VIF)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = x_train.columns
    vif_data["VIF"] = [variance_inflation_factor(x_train_const.values, i + 1) for i in range(len(x_train.columns))]

    # Heteroskedasticity (White)
    white_test = het_white(model.resid, model.model.exog)
    white_results = dict(zip(['Statistical LM', 'p-value', 'Statistical F', 'p-value F'], white_test))

    # Auto-correlation (Durbin-Watson)
    dw_stat = durbin_watson(model.resid)

    # Interpretation of tests
    white_interpretation = (
        "Heteroskedasticity detected." if white_results['p-value'] < 0.05 else "Homoskedasticity detected (p > 0.05)."
    )
    dw_interpretation = (
        "No auto-correlation detected." if 1.5 < dw_stat < 2.5 else "Auto-correlation detected."
    )

    # Generate HTML file
    html_content = f"""
    <html>
    <head>
        <title>Model Verification Report</title>
    </head>
    <body>
        <h1>Modeling Report</h1>

        <h2>(a) Non-linearity</h2>
        <div>{residuals_plot.to_html(full_html=False, include_plotlyjs='cdn')}</div>

        <h2>(b) Outliers</h2>
        <p>Cook's Distance: Top 5 :</p>
        <pre>{pd.Series(cooks_d).sort_values(ascending=False).head(5).to_string()}</pre>

        <h2>(c) Multi-collinearity</h2>
        <p>Inflation variance factors (VIF) :</p>
        <pre>{vif_data.to_string(index=False)}</pre>

        <h2>(d) Heteroskedasticity</h2>
        <p>White's test results :</p>
        <pre>{white_results}</pre>
        <p>Interpretation : {white_interpretation}</p>

        <h2>(e) Auto-correlation</h2>
        <p>Durbin-Watson statistic : {dw_stat:.2f}</p>
        <p>Interpretation : {dw_interpretation}</p>
    </body>
    </html>
    """

    with open("modeling report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Report generated : modeling report.html")
