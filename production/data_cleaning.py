"""Processors for the data cleaning step of the workflow.

The processors in this step apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
import os
import tarfile
import urllib.request
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning
)


def binned_income(df):
    """Bin the median_income column for stratified sampling."""
    return pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )


@register_processor("data-cleaning", "housing")
def clean_housing_table(context, params):
    """Clean the ``housing`` data table.

    The table contains information on housing data in California.
    This function downloads the data if necessary and cleans it.
    """
    # Define paths
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join("./data/", "raw", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    # Create directories if they don't exist
    os.makedirs(HOUSING_PATH, exist_ok=True)

    # Download and extract data
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    if not os.path.exists(os.path.join(HOUSING_PATH, "housing.csv")):
        urllib.request.urlretrieve(HOUSING_URL, tgz_path)
        with tarfile.open(tgz_path) as housing_tgz:
            housing_tgz.extractall(path=HOUSING_PATH)

    # Now load and clean the dataset
    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    # load dataset (if it exists in the catalog)
    try:
        housing_df = load_dataset(context, input_dataset)
    except Exception:
        # If the dataset isn't found in the catalog, read directly from file
        csv_path = os.path.join(HOUSING_PATH, "housing.csv")
        housing_df = pd.read_csv(csv_path)
        # Save it to the raw location
        save_dataset(context, housing_df, input_dataset)
        # Load it again to ensure consistent processing
        housing_df = load_dataset(context, input_dataset)

    # Handling missing values
    housing_df_clean = (
        housing_df
        # Fill missing values with median for numeric columns
        .fillna(housing_df.select_dtypes(include=[np.number]).median())
        # Add engineered features
        .assign(
            rooms_per_household=housing_df["total_rooms"] / housing_df["households"],
            bedrooms_per_room=housing_df["total_bedrooms"] / housing_df["total_rooms"],
            population_per_household=housing_df["population"] / housing_df["households"]
        )
        # Clean column names (comment out this line while cleaning data above)
        .clean_names(case_type="snake")
    )

    # save the dataset
    save_dataset(context, housing_df_clean, output_dataset)

    return housing_df_clean


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``housing`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df = load_dataset(context, input_dataset)

    # Add income_cat for stratified sampling
    housing_df["income_cat"] = pd.cut(
        housing_df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # split the data using stratified sampling
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    train_df, test_df = custom_train_test_split(
        housing_df, splitter, by="income_cat"
    )

    # Remove the income_cat column as it's no longer needed
    train_df = train_df.drop("income_cat", axis=1)
    test_df = test_df.drop("income_cat", axis=1)

    # split train dataset into features and target
    target_col = params["target"]
    train_X, train_y = (
        train_df
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        test_df
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)