import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

MODEL_PIPELINE = os.environ.get("MODEL_PIPELINE", "models/model_pipeline.joblib")
SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "service_account.json")
SPREADSHEET_URL = os.environ.get("SPREADSHEET_URL")
EXISTING_CSV = os.environ.get("EXISTING_CSV", "data/responses.csv")
NEW_ENTRIES_CSV = os.environ.get("NEW_ENTRIES_CSV", "data/new_entries.csv")
NUM_SYNTHETIC = int(os.environ.get("NUM_SYNTHETIC", 20))

if not SPREADSHEET_URL:
    raise SystemExit("Set SPREADSHEET_URL environment variable.")

pipeline = joblib.load(MODEL_PIPELINE)

if os.path.exists(NEW_ENTRIES_CSV):
    new_df = pd.read_csv(NEW_ENTRIES_CSV)
else:
    df_existing = pd.read_csv(EXISTING_CSV)
    # filter out previously predicted rows
    if 'IsPredicted' in df_existing.columns:
        df_existing = df_existing[df_existing['IsPredicted'].astype(str).str.lower() != 'true']
    drop_cols = [c for c in df_existing.columns if 'price' in c.lower() or 'timestamp' in c.lower()]
    features_df = df_existing.drop(columns=[c for c in drop_cols if c in df_existing.columns], errors='ignore')
    new_df = features_df.sample(n=NUM_SYNTHETIC, replace=True, random_state=42).reset_index(drop=True)
    # small numeric noise
    for col in new_df.select_dtypes(include=[np.number]).columns:
        std = new_df[col].std() if new_df[col].std() > 0 else 1.0
        new_df[col] = (new_df[col] + np.random.normal(0, 0.02 * std, size=len(new_df))).clip(lower=0)

preds = pipeline.predict(new_df)
new_df["Predicted Price"] = preds
new_df["IsPredicted"] = True
new_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
gc = gspread.authorize(creds)

if SPREADSHEET_URL.startswith("http"):
    sh = gc.open_by_url(SPREADSHEET_URL)
else:
    sh = gc.open_by_key(SPREADSHEET_URL)

ws = sh.sheet1
headers = ws.row_values(1)
if not headers:
    raise SystemExit("Target sheet has no header row. Add headers to the sheet matching your CSV columns.")

# Build output rows matching sheet headers
out_df = pd.DataFrame(columns=headers)
new_cols_norm = {c.lower().strip(): c for c in new_df.columns}

for col in headers:
    key = col.lower().strip()
    if key in new_cols_norm:
        out_df[col] = new_df[new_cols_norm[key]].astype(str)
    elif 'price' in key:
        out_df[col] = new_df.get("Predicted Price", "").astype(str)
    elif 'time' in key or 'timestamp' in key:
        out_df[col] = new_df.get("Timestamp", "").astype(str)
    else:
        out_df[col] = ""

# add any extra columns at the end
for c in new_df.columns:
    if c not in out_df.columns:
        out_df[c] = new_df[c].astype(str)

rows_to_append = out_df.values.tolist()
ws.append_rows(rows_to_append, value_input_option='USER_ENTERED')
print(f"Appended {len(rows_to_append)} rows to the spreadsheet.")
