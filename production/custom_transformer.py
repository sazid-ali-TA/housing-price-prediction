# production/custom_transformer.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies log transformation to numeric features
    and supports inverse transformation.

    Parameters:
    -----------
    features : list or None, default=None
        List of features to transform. If None, all numeric features will be transformed.
    epsilon : float, default=1e-6
        Small constant added to values before log transform to avoid log(0).
    log_base : float, default=np.e
        Base of the logarithm to use.
    fillna : bool, default=True
        Whether to fill missing values with epsilon before transformation.
    """

    def __init__(self, features=None, epsilon=0.000001, log_base=np.e, fillna=True):
        self.features = features
        self.epsilon = epsilon
        self.log_base = log_base
        self.fillna = fillna
        self._transformed_features = None

    def fit(self, X, y=None):
        """
        Identify the columns to be transformed.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to fit the transformer.
        y : array-like, default=None
            Not used, present for API consistency.

        Returns:
        --------
        self : object
            Returns self.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        # If features not specified, use all numeric columns
        if self.features is None:
            self._transformed_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()
        else:
            # Verify all requested features exist and are numeric
            numeric_cols = X.select_dtypes(include=np.number).columns
            missing_cols = set(self.features) - set(numeric_cols)
            if missing_cols:
                raise ValueError(
                    f"The following features are either missing or not numeric: {missing_cols}"
                )
            self._transformed_features = self.features

        print(f"Fitted features: {self._transformed_features}")
        return self

    def transform(self, X):
        """
        Apply log transformation to selected features.

        Parameters:
        -----------
        X : pandas DataFrame
            Data to transform.

        Returns:
        --------
        X_transformed : pandas DataFrame
            Transformed data.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        X_transformed = X.copy()

        for col in self._transformed_features:

            # Ensure all values are numeric
            X_transformed[col] = pd.to_numeric(X_transformed[col], errors="coerce")

            # Debug: Check for unexpected string values
            if X_transformed[col].dtype != np.float64:
                print(
                    f"Warning: Column {col} is not fully numeric after conversion! Type: {X_transformed[col].dtype}"
                )
                print(
                    "Unique types in column:",
                    X_transformed[col].apply(type).unique(),
                )

            # Handle NaNs before transformation
            if self.fillna:
                X_transformed[col] = X_transformed[col].fillna(0.000001)

            # Apply log transformation
            X_transformed[col] = np.log(X_transformed[col] + 0.000001) / np.log(
                self.log_base
            )
        print(f"Transformed columns: {self._transformed_features}")
        return X_transformed

    def inverse_transform(self, X):
        """
        Apply inverse of log transformation to transformed features.

        Parameters:
        -----------
        X : pandas DataFrame
            Data to inverse transform.

        Returns:
        --------
        X_inverted : pandas DataFrame
            Inverse transformed data.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        X_inverted = X.copy()

        for col in self._transformed_features:
            # Ensure column is numeric before inverse transformation
            X_inverted[col] = pd.to_numeric(X_inverted[col], errors="coerce")

            # Apply inverse transformation
            X_inverted[col] = (self.log_base ** X_inverted[col]) - self.epsilon

        print(f"Inverse transformed columns: {self._transformed_features}")
        return X_inverted

    def save(self, filepath):
        """Save transformer to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Load transformer from disk."""
        return joblib.load(filepath)
