# Pre-Trained-Data-Insertion

This project provides utilities to train a regression model on CSV survey responses and append prediction rows to a Google Sheet.

## Setup

1. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```
2. **Set environment variables**
   - `EXISTING_CSV` (default: `data/responses.csv`)
   - `SERVICE_ACCOUNT_FILE` path to your Google service account JSON.
   - `SPREADSHEET_URL` URL of the Google Sheet to append to.

Never commit your `service_account.json`. Add it to `.gitignore`.

A small sample dataset is provided in `data/responses.csv`. Replace it with your own data as needed. If your trained model becomes large, save it under `models/` and consider using [Git LFS](https://git-lfs.github.com/).

## Usage

Train the model:
```bash
python train_model.py
```

Append predictions to a Google Sheet:
```bash
python append_predictions.py
```

## GitHub Actions

The workflow in `.github/workflows/predict-and-append.yml` demonstrates how to run training and prediction in CI. Store the service account JSON in the `SERVICE_ACCOUNT_JSON` secret and the spreadsheet URL in `SPREADSHEET_URL`.
