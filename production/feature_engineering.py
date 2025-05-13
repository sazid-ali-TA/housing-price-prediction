"""Processors for the feature engineering step of the workflow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""

import logging
import os.path as op

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)

from ta_lib.data_processing.api import Outlier
from custom_transformer import LogTransformer

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # Identify column types
    cat_columns = train_X.select_dtypes("object").columns.tolist()
    num_columns = train_X.select_dtypes("number").columns.tolist()

    # Treating Outliers
    outlier_transformer = Outlier(method=params["outliers"]["method"])
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"]
    )

    # Apply log transformation if enabled
    log_transform_params = params.get("log_transform", {})
    if log_transform_params.get("enabled", False):
        logger.info("Applying log transformation")
        log_features = log_transform_params.get("features", None)
        log_epsilon = log_transform_params.get("epsilon", 1e-6)
        log_base = log_transform_params.get("log_base", 10)

        log_transformer = LogTransformer(
            features=log_features, epsilon=log_epsilon, log_base=log_base
        )

        # Fit and transform
        train_X = log_transformer.fit_transform(train_X)

        # Save the log transformer
        save_pipeline(
            log_transformer,
            op.abspath(op.join(artifacts_folder, "log_transformer.joblib")),
        )

    # Create feature engineering pipeline
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse=False, drop="first")),
        ]
    )

    # Combine transformers in a ColumnTransformer
    features_transformer = ColumnTransformer(
        [
            ("num", num_pipeline, num_columns),
            ("cat", cat_pipeline, cat_columns),
        ]
    )

    # Check if the data should be sampled
    sample_frac = params.get("sampling_fraction", None)
    if sample_frac is not None and sample_frac < 1.0:
        logger.warn(f"The data has been sampled by fraction: {sample_frac}")
        sample_X = train_X.sample(
            frac=sample_frac, random_state=context.random_seed
        )
        sample_y = train_y.loc[sample_X.index]
    else:
        sample_X = train_X
        sample_y = train_y

    # Fit the transformer on the training data
    features_transformer.fit(sample_X, sample_y)

    # Save all feature columns
    curated_columns = get_feature_names_from_column_transformer(
        features_transformer
    )
 
    # Save the list of relevant columns and the pipeline
    save_pipeline(
        curated_columns,
        op.abspath(op.join(artifacts_folder, "curated_columns.joblib")),
    )
    save_pipeline(
        features_transformer,
        op.abspath(op.join(artifacts_folder, "features.joblib")),
    )