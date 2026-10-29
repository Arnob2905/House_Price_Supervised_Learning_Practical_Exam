"""Input preparation helpers for the Ames House Price Prediction model."""

import pandas as pd

QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLUMNS = ["ExterQual", "KitchenQual", "BsmtQual", "FireplaceQu"]


def prepare_house_features(data):
    """Prepare raw Ames house rows for the fitted model without mutating input."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    X = data.copy()
    X = X.drop(columns=["Id", "SalePrice", "SalePrice_log"], errors="ignore")

    required = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "YrSold", "YearBuilt",
                "YearRemodAdd", "GarageArea", "PoolArea"]
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise ValueError(f"Input is missing columns needed for feature engineering: {missing}")

    X["TotalSF"] = X["TotalBsmtSF"].fillna(0) + X["1stFlrSF"].fillna(0) + X["2ndFlrSF"].fillna(0)
    X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
    X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
    X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
    X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)

    for col in QUALITY_COLUMNS:
        if col in X.columns and not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna("None").map(QUALITY_MAP)
    return X


class RawHousePriceModel:
    """Small wrapper so the saved model accepts raw Kaggle predictor rows."""
    def __init__(self, fitted_pipeline):
        self.pipeline = fitted_pipeline

    def predict(self, X):
        return self.pipeline.predict(prepare_house_features(X))

    def predict_prices(self, X):
        import numpy as np
        return np.expm1(self.predict(X))
