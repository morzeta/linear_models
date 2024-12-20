def regression_report(result_bp, mse, r2, fig):
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
        <div>{fig}</div>
    </body>
    </html>
    """

    # Save to HTML file
    with open("regression report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Report generated : regression report.html")