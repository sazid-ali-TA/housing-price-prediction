import unittest
import numpy as np
import pandas as pd
from ..custom_transformer import LogTransformer


class TestLogTransformer(unittest.TestCase):
    """Unit tests for the LogTransformer class."""

    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame(
            {
                "numeric1": [1, 10, 100, 1000],
                "numeric2": [0, 5, 15, 25],
                "categorical": ["a", "b", "c", "d"],
            }
        )

    def test_init_default(self):
        """Test default initialization."""
        transformer = LogTransformer()
        self.assertIsNone(transformer.features)
        self.assertEqual(transformer.epsilon, 0.000001)
        self.assertEqual(transformer.log_base, np.e)

    def test_init_custom(self):
        """Test custom initialization."""
        transformer = LogTransformer(features=["numeric1"], epsilon=0.1, log_base=10)
        self.assertEqual(transformer.features, ["numeric1"])
        self.assertEqual(transformer.epsilon, 0.1)
        self.assertEqual(transformer.log_base, 10)

    def test_fit_all_numeric(self):
        """Test fit with default parameters (all numeric columns)."""
        transformer = LogTransformer()
        transformer.fit(self.data)
        self.assertEqual(
            set(transformer._transformed_features), {"numeric1", "numeric2"}
        )

    def test_fit_specific_columns(self):
        """Test fit with specific columns."""
        transformer = LogTransformer(features=["numeric1"])
        transformer.fit(self.data)
        self.assertEqual(transformer._transformed_features, ["numeric1"])

    def test_fit_invalid_column(self):
        """Test fit with invalid column."""
        transformer = LogTransformer(features=["nonexistent"])
        with self.assertRaises(ValueError):
            transformer.fit(self.data)

    def test_transform(self):
        """Test transform method."""
        transformer = LogTransformer(features=["numeric1"], log_base=10)
        transformer.fit(self.data)
        transformed = transformer.transform(self.data)

        # Calculate expected values manually
        expected_col = np.log10(self.data["numeric1"] + 0.000001)

        # Check that original data is not modified
        self.assertTrue((self.data["numeric1"] == [1, 10, 100, 1000]).all())

        # Check that numeric1 column is transformed
        np.testing.assert_array_almost_equal(
            transformed["numeric1"].values, expected_col.values
        )

        # Check that numeric2 column is not transformed
        np.testing.assert_array_equal(
            transformed["numeric2"].values, self.data["numeric2"].values
        )

    def test_inverse_transform(self):
        """Test inverse_transform method."""
        transformer = LogTransformer(features=["numeric1", "numeric2"], log_base=10)
        transformer.fit(self.data)
        transformed = transformer.transform(self.data)
        inverted = transformer.inverse_transform(transformed)

        # Check that the inverse transform approximately recovers the original data
        np.testing.assert_array_almost_equal(
            inverted["numeric1"].values,
            self.data["numeric1"].values,
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            inverted["numeric2"].values,
            self.data["numeric2"].values,
            decimal=5,
        )

    def test_transform_and_inverse_transform_workflow(self):
        """Test the full workflow from original to transformed to back."""
        transformer = LogTransformer()
        transformer.fit(self.data)
        transformed = transformer.transform(self.data)
        inverted = transformer.inverse_transform(transformed)

        # Verify all numeric columns were processed correctly
        for col in ["numeric1", "numeric2"]:
            np.testing.assert_array_almost_equal(
                inverted[col].values, self.data[col].values, decimal=5
            )

        # Verify categorical column remains unchanged
        self.assertTrue((inverted["categorical"] == self.data["categorical"]).all())

    def test_save_and_load(self):
        """Test saving and loading the transformer."""
        import tempfile
        import os

        transformer = LogTransformer(features=["numeric1"])
        transformer.fit(self.data)
        transformed = transformer.transform(self.data)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            filepath = f.name

        try:
            # Save the transformer
            transformer.save(filepath)

            # Load the transformer
            loaded_transformer = LogTransformer.load(filepath)

            # Check that the loaded transformer works the same
            loaded_transformed = loaded_transformer.transform(self.data)
            np.testing.assert_array_equal(
                loaded_transformed["numeric1"].values,
                transformed["numeric1"].values,
            )
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
