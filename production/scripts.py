"""Module for listing down additional custom functions required for housing price prediction."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_housing_data(housing_df, output_path=None):
    """Create visualization plots for the housing data."""
    # Geographical scatter plot
    plt.figure(figsize=(10, 7))
    plt.scatter(
        housing_df["longitude"],
        housing_df["latitude"],
        alpha=0.4,
        s=housing_df["population"] / 100,
        c=housing_df["median_house_value"],
        cmap=plt.get_cmap("jet"),
    )
    plt.colorbar(label="Median House Value")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Housing Prices in California")

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    return plt


def binned_income(df):
    """Bin the median_income column using quantiles."""
    return pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
