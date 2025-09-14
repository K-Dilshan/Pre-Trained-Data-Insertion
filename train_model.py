import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

EXISTING_CSV = os.environ.get("EXISTING_CSV", "data/responses.csv")
MODEL_OUT = os.environ.get("MODEL_OUT", "models/model_pipeline.joblib")
RANDOM_STATE = 42

os.makedirs(os.path.dirname(MODEL_OUT) or ".", exist_ok=True)

df = pd.read_csv(EXISTING_CSV)
# If you previously appended predicted rows, filter them out if flagged
if 'IsPredicted' in df.columns:
    df = df[df['IsPredicted'].astype(str).str.lower() != 'true']

possible_targets = [c for c in df.columns if 'price' in c.lower()]
if not possible_targets:
    raise SystemExit("No target column with 'price' found. Set column name containing 'price'.")
target_col = possible_targets[0]
print("Using target:", target_col)

skip_cols = [c for c in df.columns if 'timestamp' in c.lower()]
X = df.drop(columns=[target_col] + skip_cols, errors='ignore')
y = df[target_col].copy()

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer([('num', num_pipeline, numeric_cols), ('cat', cat_pipeline, categorical_cols)], remainder='drop')

pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}")

joblib.dump(pipeline, MODEL_OUT)
print("Saved model pipeline to", MODEL_OUT)
