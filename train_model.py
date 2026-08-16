"""Train and export the CarValue AI Random Forest model.

Usage:
    python train_model.py
    python train_model.py --n-iter 20 --reference-year 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "Present_Price",
    "Kms_Driven",
    "Owner",
    "Years_Since_Manufacture",
    "Fuel_Type_Diesel",
    "Fuel_Type_Petrol",
    "Seller_Type_Individual",
    "Transmission_Manual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-year",
        type=int,
        default=2024,
        help="Year used to derive vehicle age (default: 2024).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="RandomizedSearchCV configurations to evaluate (default: 10).",
    )
    return parser.parse_args()


def prepare_data(data: pd.DataFrame, reference_year: int) -> tuple[pd.DataFrame, pd.Series]:
    transformed = data.copy()
    transformed["Years_Since_Manufacture"] = reference_year - transformed["Year"]
    transformed = transformed.drop(columns=["Year", "Car_Name"], errors="ignore")
    transformed = pd.get_dummies(transformed, drop_first=True)
    transformed = transformed.reindex(
        columns=["Selling_Price", *FEATURE_COLUMNS],
        fill_value=0,
    )
    return transformed[FEATURE_COLUMNS], transformed["Selling_Price"]


def main() -> None:
    args = parse_args()
    data_path = PROJECT_DIR / "car.csv"
    model_path = PROJECT_DIR / "random_forest_regression_model.pkl"

    data = pd.read_csv(data_path)
    features, target = prepare_data(data, args.reference_year)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
    parameter_distributions = {
        "n_estimators": [200, 300, 400, 500],
        "max_features": [1.0, "sqrt", "log2"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 6],
    }
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=args.n_iter,
        scoring="neg_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print(f"Training on {len(train_features)} rows and validating on {len(test_features)} rows...")
    search.fit(train_features, train_target)
    best_model = search.best_estimator_
    predictions = best_model.predict(test_features)

    with model_path.open("wb") as model_file:
        pickle.dump(best_model, model_file)

    print(f"Saved model to {model_path.name}")
    print(f"Best parameters: {search.best_params_}")
    print(f"RMSE: {np.sqrt(mean_squared_error(test_target, predictions)):.4f} lakh")
    print(f"MAE:  {mean_absolute_error(test_target, predictions):.4f} lakh")
    print(f"R²:   {r2_score(test_target, predictions):.4f}")


if __name__ == "__main__":
    main()

