"""Data preprocessing for CICIDS2017 network flow data."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE


# Columns to drop before training (non-features)
DROP_COLS = ["Label", "attack_category"]


class Preprocessor:
    """Handles cleaning, scaling, and resampling for CICIDS2017 data."""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.target_encoder = LabelEncoder()
        self._is_fitted = False
        self._feature_cols: list[str] = []

    @staticmethod
    def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf values with NaN, then drop rows that are all-NaN in features."""
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "attack_category",
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Fit preprocessor on training data and transform it.

        Returns:
            (X, y, feature_names)
        """
        df = df.copy()

        # Determine feature columns (everything except label columns)
        self._feature_cols = [c for c in df.columns if c not in DROP_COLS]

        # Clean inf/NaN values
        n_before = len(df)
        df = self._clean_numeric(df)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"  [*] Dropped {n_dropped} rows with inf/NaN values")

        # Encode target
        y = self.target_encoder.fit_transform(df[target_col])

        # Select numeric features only
        X = df[self._feature_cols]
        # Ensure all columns are numeric (drop any non-numeric that slipped through)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"  [*] Dropping non-numeric columns: {non_numeric}")
            self._feature_cols = [c for c in self._feature_cols if c not in non_numeric]
            X = df[self._feature_cols]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self._is_fitted = True

        print(f"[+] Preprocessing complete: {X_scaled.shape[1]} features, {len(y)} samples")

        # Apply SMOTE to balance classes
        # Determine safe k_neighbors (must be < smallest class count)
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        k_neighbors = min(5, min_count - 1)
        if k_neighbors < 1:
            print(f"  [!] Smallest class has {min_count} sample(s), skipping SMOTE")
            return X_scaled, y, self._feature_cols

        print(f"[*] Applying SMOTE oversampling (k_neighbors={k_neighbors})...")
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"    Before SMOTE: {len(y)} samples")
        print(f"    After SMOTE:  {len(y_resampled)} samples")
        for u, c in zip(unique, counts):
            print(f"      {self.target_encoder.classes_[u]}: {c}")

        return X_resampled, y_resampled, self._feature_cols

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str = "attack_category",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform test data using fitted preprocessor.

        Returns:
            (X, y)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        df = df.copy()
        df = self._clean_numeric(df)

        # Encode target
        y = self.target_encoder.transform(df[target_col])

        # Select same feature columns used during fit
        X = df[self._feature_cols]
        X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def get_target_names(self) -> list[str]:
        """Return ordered class names."""
        return list(self.target_encoder.classes_)
