# -*- coding: utf-8 -*-
"""function for plot."""
import matplotlib.pyplot as plt
import numpy as np
from grid_search import get_best_parameters
import seaborn as sns
import pandas as pd

# Visualize filtered data with correlation to target
def visualize_filtered_data(headers, data, y):
    """
    Visualize the dataset after dropping columns.

    Args:
        headers (list): List of feature names.
        data (np.array): The filtered dataset (x_train_filtered).
        y (np.array): The target variable.
    """
    # Convert data to a DataFrame for easier manipulation and visualization
    df = pd.DataFrame(data, columns=headers)

    # Add target variable to the beginning of the DataFrame
    df.insert(0, 'Target', y)

    # Step 1: Check for missing values in percentage for each column
    missing_values_percentage = df.isna().mean() * 100
    print("Missing Values Percentage per Feature:")
    print(missing_values_percentage)

    # Step 2: Plot histograms for the first 15 features
    num_features = min(15, len(headers))  
    df.iloc[:, :num_features].hist(bins=30, figsize=(15, 10), edgecolor='black')
    plt.suptitle("Histograms of the First 15 Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Step 3: Correlation heatmap to visualize relationships between the first 15 features and target
    plt.figure(figsize=(12, 8))
    corr_matrix = df.iloc[:, :num_features + 1].corr()  # Include target variable in correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', linewidths=0.5)
    plt.title("Correlation Heatmap of the First 15 Features and Target", fontsize=16)
    plt.show()

def prediction(w0, w1, mean_x, std_x):
    """Get the regression line from the model."""
    x = np.arange(1.2, 2, 0.01)
    x_normalized = (x - mean_x) / std_x
    return x, w0 + w1 * x_normalized


def base_visualization(grid_losses, w0_list, w1_list, mean_x, std_x, height, weight):
    """Base Visualization for both models."""
    w0, w1 = np.meshgrid(w0_list, w1_list)

    fig = plt.figure()

    # plot contourf
    ax1 = fig.add_subplot(1, 2, 1)
    cp = ax1.contourf(w0, w1, grid_losses.T, cmap=plt.cm.jet)
    fig.colorbar(cp, ax=ax1)
    ax1.set_xlabel(r"$w_0$")
    ax1.set_ylabel(r"$w_1$")
    # put a marker at the minimum
    loss_star, w0_star, w1_star = get_best_parameters(w0_list, w1_list, grid_losses)
    ax1.plot(w0_star, w1_star, marker="*", color="r", markersize=20)

    # plot f(x)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(height, weight, marker=".", color="b", s=5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid()

    return fig


def grid_visualization(grid_losses, w0_list, w1_list, mean_x, std_x, height, weight):
    """Visualize how the trained model looks like under the grid search."""
    fig = base_visualization(
        grid_losses, w0_list, w1_list, mean_x, std_x, height, weight
    )

    loss_star, w0_star, w1_star = get_best_parameters(w0_list, w1_list, grid_losses)
    # plot prediciton
    x, f = prediction(w0_star, w1_star, mean_x, std_x)
    ax2 = fig.get_axes()[2]
    ax2.plot(x, f, "r")

    return fig


def gradient_descent_visualization(
    gradient_losses,
    gradient_ws,
    grid_losses,
    grid_w0,
    grid_w1,
    mean_x,
    std_x,
    height,
    weight,
    n_iter=None,
):
    """Visualize how the loss value changes until n_iter."""
    fig = base_visualization(
        grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight
    )

    ws_to_be_plotted = np.stack(gradient_ws)
    if n_iter is not None:
        ws_to_be_plotted = ws_to_be_plotted[:n_iter]

    ax1, ax2 = fig.get_axes()[0], fig.get_axes()[2]
    ax1.plot(
        ws_to_be_plotted[:, 0],
        ws_to_be_plotted[:, 1],
        marker="o",
        color="w",
        markersize=10,
    )
    pred_x, pred_y = prediction(
        ws_to_be_plotted[-1, 0], ws_to_be_plotted[-1, 1], mean_x, std_x
    )
    ax2.plot(pred_x, pred_y, "r")

    return fig
