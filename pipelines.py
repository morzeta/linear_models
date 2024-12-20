import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from statsmodels.api import add_constant
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from preparation import read_data, get_x_y, data_columns
from visualize import residual_subplots
from html_report import regression_report


def ordinary_linear_regression_v1():
    # Load and clean data
    data = read_data()
    x, y = get_x_y(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    # Fit the model
    x_with_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_with_const).fit()

    pred_summary = model.get_prediction(sm.add_constant(x_test)).summary_frame(alpha=0.05)

    print(pred_summary.head())

    # Summary of the model
    print(model.summary())
    # outliers
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Breusch-Pagan Test for heteroskedasticity
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
        print("\nConclusion: The results suggest heteroskedasticity (p-value < 0.05).")
        print("This means that the variance of the residuals is not constant.")
    else:
        print("\nConclusion: The results suggest homoskedasticity (p-value >= 0.05).")
        print("This means that the variance of the residuals is constant.")


def ordinary_linear_regression_v2():
    # Load and clean data
    data = read_data()
    x, y = get_x_y(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Fit the model
    x_with_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_with_const).fit()

    residuals = model.resid
    cov_matrix = np.diag(residuals ** 2)

    gls_model = sm.GLS(y_train, x_with_const, sigma=cov_matrix).fit()
    print(gls_model.summary())

    # Breusch-Pagan Test for heteroskedasticity
    gls_residuals = gls_model.resid
    bp_test = het_breuschpagan(gls_residuals, x_with_const)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    result = dict(zip(labels, bp_test))

    # Show results
    print("\nResults of the Breusch-Pagan test for homoskedasticity :")
    for key, value in result.items():
        print(f"{key}: {value:.4f}")

    # Clear interpretation
    p_value = result['p-value']
    if p_value < 0.05:
        print("\nConclusion: The results suggest heteroskedasticity (p-value < 0.05).")
        print("This means that the variance of the residuals is not constant.")
    else:
        print("\nConclusion: The results suggest homoskedasticity (p-value >= 0.05).")
        print("This means that the variance of the residuals is constant.")


def ordinary_linear_regression_v3():
    # Load and clean data
    data = read_data()
    x, y = get_x_y(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Fit the model
    x_with_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_with_const, cov_type='HC3').fit()

    # Summary of the model
    print(model.summary())

    # outliers
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Breusch-Pagan Test for heteroskedasticity
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
        print("\nConclusion: The results suggest heteroskedasticity (p-value < 0.05).")
        print("This means that the variance of the residuals is not constant.")
    else:
        print("\nConclusion: The results suggest homoskedasticity (p-value >= 0.05).")
        print("This means that the variance of the residuals is constant.")


def ordinary_linear_regression_v4():
    # Load and clean data
    data = read_data()
    x, y = get_x_y(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Fit the model
    x_with_const = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_with_const, cov_type='HC3').fit()

    # Summary of the model
    print(model.summary())

    print("\nf-test for linear combination of CO")
    f_test_result = model.f_test(
        data_columns[0] + " + " +
        data_columns[1] + " = 0")
    print(f_test_result)
    print("\nf-test for linear combination of NOx")
    f_test_result = model.f_test(
        data_columns[5] + " + " +
        data_columns[6] + " = 0")
    print(f_test_result)

    print("\nf-test for linear combination of coefficient for NOx and CO concentration")
    f_test_result = model.f_test(
        data_columns[1] + " = 0, " + data_columns[6] + " = 0")
    print(f_test_result)

    print("\nf-test for linear combination of of coefficient for NOx and CO sensor measure")
    f_test_result = model.f_test(
        data_columns[0] + " = 0, " + data_columns[5] + " = 0")
    print(f_test_result)

    # outliers
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Breusch-Pagan Test for heteroskedasticity
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
        print("\nConclusion: The results suggest heteroskedasticity (p-value < 0.05).")
        print("This means that the variance of the residuals is not constant.")
    else:
        print("\nConclusion: The results suggest homoskedasticity (p-value >= 0.05).")
        print("This means that the variance of the residuals is constant.")


def linear_regression_v1():
    # Load and clean data
    data = read_data()
    x, y = get_x_y(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(x_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Intercept:", model.intercept_)
    print("Coefficients:", dict(zip(data_columns, model.coef_[0])))

    print("Model evaluation :")
    print(f"Mean Square Error (MSE) : {mse:.2f}")
    print(f"Score R² : {r2:.2f}")

    # Calculate the residuals
    residuals = y_test.values.flatten() - y_pred.flatten()
    print(f"Residuals size: {len(residuals)}")
    print(f"x_test size: {len(x_test)}")

    # residual_subplots(x_test, residuals)
    # plt.show()


def linear_regression_v2():
    # Load and clean data
    data = read_data()
    x, y = get_x_y(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

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

    # regression_report(result_bp, mse, r2, fig.to_html(full_html=False, include_plotlyjs='cdn'))

    fig.show()


def linear_regression_v3():
    # Load data
    data = read_data()
    x, y = get_x_y(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

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


def heteroskedasticity_test():
    data = read_data()
    x, y = get_x_y(data)

    # Add a constant for the model
    x_with_const = sm.add_constant(x)

    # Perform the initial regression to detect heteroskedasticity
    model = sm.OLS(y, x_with_const).fit()
    residuals = model.resid

    # White's test for heteroskedasticity
    white_test = het_white(residuals, x_with_const)
    labels = ['LM Statistics', 'LM p-value', 'F statistic', 'F p-value']
    white_results = dict(zip(labels, white_test))
    print("\nWhite's test results :")
    for key, value in white_results.items():
        print(f"{key}: {value:.4f}")

    if white_results['LM p-value'] < 0.05:
        print("\nConclusion: Heteroskedasticity detected. Data transformation in progress...")

        # Logarithmic transformation (e.g. on y)
        y_log = np.log1p(y)  # log(1 + y) to avoid negative values

        # Regression with transformed data
        model_log = sm.OLS(y_log, x_with_const).fit()
        print("\nSummary after logarithmic transformation:")
        print(model_log.summary())
    else:
        print("\nConclusion: No heteroskedasticity detected.")
