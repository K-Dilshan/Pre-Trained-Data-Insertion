import os
from pathlib import Path
import pandas as pd
import joblib
import gspread

def main() -> None:
    csv_path = os.environ.get("EXISTING_CSV", "data/responses.csv")
    model_path = Path("models/pipeline.joblib")
    sa_file = os.environ.get("SERVICE_ACCOUNT_FILE", "service_account.json")
    spreadsheet_url = os.environ.get("SPREADSHEET_URL")

    if not model_path.exists():
        print(f"Model file not found at {model_path}. Run train_model.py first.")
        return

    model = joblib.load(model_path)

    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        feature_df = df.drop(columns=["price"], errors="ignore")
        new_row = feature_df.sample(1, random_state=0)
    else:
        print(f"CSV file not found at {csv_path}. Unable to generate data.")
        return

    preds = model.predict(new_row)
    output = new_row.copy()
    output["predicted_price"] = preds

    if spreadsheet_url:
        gc = gspread.service_account(filename=sa_file)
        sh = gc.open_by_url(spreadsheet_url)
        ws = sh.sheet1
        ws.append_rows(output.astype(str).values.tolist())
        print("Appended predictions to Google Sheet.")
    else:
        print("SPREADSHEET_URL not provided. Predictions:")
        print(output)

if __name__ == "__main__":
    main()
