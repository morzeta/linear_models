import random

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import seaborn as sns

from preparation import data_columns

color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(len(data_columns))]


def correlation_subplots(data):
    fig, axes = plt.subplots(ncols=4, nrows=4)
    fig.subplots_adjust(hspace=.5, wspace=.5)

    for col, ax, clr in zip(data_columns, axes.flat, color):
        sns.regplot(x=col,
                    y='Time',
                    data=data,
                    ax=ax,
                    marker=".",
                    scatter_kws={"color": clr},
                    line_kws={"color": "black"})


def correlation_heatmap(data):
    sns.heatmap(data.corr(), annot=True, linewidths=.4)
    plt.title('Heatmap of co-relation between variables', fontsize=16)


# Plot the residuals against one or more explanatory variables
def residual_subplots(x_test, residuals):
    fig, axes = plt.subplots(ncols=4, nrows=4)
    fig.subplots_adjust(hspace=.5, wspace=.5)

    for col, ax, clr in zip(data_columns, axes.flat, color):
        ax.scatter(x_test[col], residuals, alpha=0.5, color=clr)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel(col)

    fig.supylabel('residuals')
