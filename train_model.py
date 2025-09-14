import os
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib


DEFAULT_CSV = (
    "data/Vehicle Valuation and Price Prediction of Used Vehicles in Sri Lankan Market "
    "Using Machine Learning Algorithms.csv"
)
TARGET_COLUMN = "price"
ALTERNATE_PRICE_COLUMNS = [
    "What was the selling or buying price (LKR) ?",
]

def main() -> None:
    csv_path = os.environ.get("EXISTING_CSV", DEFAULT_CSV)
    model_path = Path("models/pipeline.joblib")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        for alt in ALTERNATE_PRICE_COLUMNS:
            if alt in df.columns:
                df = df.rename(columns={alt: TARGET_COLUMN})
                break
        else:
            raise ValueError("CSV must contain a 'price' column as target.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(exclude=["number"]).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])
    model.fit(X, y)

    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    main()
