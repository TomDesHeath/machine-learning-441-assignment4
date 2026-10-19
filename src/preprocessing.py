"""
Data Preprocessing module for UNSW-NB15 dataset.
Implements custom One-Hot Encoding, Min-Max scaling, and array conversions.
"""

import numpy as np
import pandas as pd


class UNSWPreprocessor:
    """
    Preprocessor for UNSW-NB15 dataset.
    Handles One-Hot Encoding for nominal features and optional Min-Max scaling for numeric features.
    """

    def __init__(self, nominal_cols, numeric_cols, scale_numeric=False):
        """
        Parameters:
        -----------
        nominal_cols : list of str
            List of nominal column names.
        numeric_cols : list of str
            List of numeric column names.
        scale_numeric : bool
            Whether to apply Min-Max scaling to numeric features (required for k-NN).
        """
        self.nominal_cols = nominal_cols
        self.numeric_cols = numeric_cols
        self.scale_numeric = scale_numeric
        self.categories_ = {}
        self.min_vals_ = None
        self.max_vals_ = None
        self.feature_names_ = []

    def fit(self, df):
        """
        Fit one-hot encoder categories and min-max feature ranges on training DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Training dataframe containing nominal and numeric features.
        """
        # Fit categories for nominal features
        self.categories_ = {}
        self.feature_names_ = list(self.numeric_cols)

        for col in self.nominal_cols:
            cats = sorted(df[col].astype(str).unique())
            self.categories_[col] = cats
            for c in cats:
                self.feature_names_.append(f"{col}_{c}")

        # Fit min-max ranges for numeric features
        if self.scale_numeric:
            num_arr = df[self.numeric_cols].values.astype(np.float64)
            self.min_vals_ = np.min(num_arr, axis=0)
            self.max_vals_ = np.max(num_arr, axis=0)
            # Avoid division by zero for constant features
            self.max_vals_[self.max_vals_ == self.min_vals_] += 1.0

        return self

    def transform(self, df):
        """
        Transform DataFrame into 2D float64 NumPy array.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to transform.

        Returns:
        --------
        np.ndarray
            Processed feature matrix of shape (N, D).
        """
        # Transform numeric features
        num_arr = df[self.numeric_cols].values.astype(np.float64)
        if self.scale_numeric:
            num_arr = (num_arr - self.min_vals_) / (self.max_vals_ - self.min_vals_)

        # Transform nominal features via one-hot encoding
        cat_blocks = []
        for col in self.nominal_cols:
            col_vals = df[col].astype(str).values
            cats = self.categories_[col]
            ohe = np.zeros((len(col_vals), len(cats)), dtype=np.float64)
            for idx, c in enumerate(cats):
                ohe[:, idx] = (col_vals == c).astype(np.float64)
            cat_blocks.append(ohe)

        if cat_blocks:
            cat_arr = np.hstack(cat_blocks)
            X = np.hstack([num_arr, cat_arr])
        else:
            X = num_arr

        return X

    def fit_transform(self, df):
        """
        Fit preprocessor and transform DataFrame.
        """
        return self.fit(df).transform(df)
