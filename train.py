"""Train a GPA predictor from SAT scores and save the best model.

Usage:
    python train.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sat_gpa.csv"
MODEL_DIR = ROOT / "model"

FEATURES = ["math_sat", "verbal_sat"]
TARGET = "gpa"


def load_data() -> pd.DataFrame:
    # The CSV has no header row: math SAT, verbal SAT, GPA
    df = pd.read_csv(DATA_PATH, header=None, names=FEATURES + [TARGET])
    return df.dropna()


def candidate_models() -> dict:
    return {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge_poly2": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=4, random_state=42
        ),
    }


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(df.describe().round(2), "\n")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pick the model with the best 5-fold cross-validated R^2 on the training set
    cv_scores = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_scores[name] = scores.mean()
        print(f"{name:15s} CV R^2 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = candidate_models()[best_name].fit(X_train, y_train)

    preds = best_model.predict(X_test)
    metrics = {
        "model": best_name,
        "cv_r2": round(cv_scores[best_name], 4),
        "test_r2": round(r2_score(y_test, preds), 4),
        "test_mae": round(mean_absolute_error(y_test, preds), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(f"\nBest: {best_name} -> test R^2 = {metrics['test_r2']}, "
          f"MAE = {metrics['test_mae']} GPA points")

    # Refit on all data for the deployed model
    final_model = candidate_models()[best_name].fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
