# SmartWaste AI — Python + Machine Learning Upgrade

A Flask + SQLite food donation platform upgraded with a trained Scikit-learn model, prediction page, model evaluation metrics and feature-importance view. Existing donor/NGO/admin workflows remain in place.

## Machine learning workflow

1. Download the public restaurant food-wastage dataset.
2. Clean numeric values, remove duplicates and impossible target rows.
3. Split into train/test sets (80/20, random seed 42).
4. Impute missing values and one-hot encode categorical columns inside a Scikit-learn pipeline.
5. Train a Random Forest Regressor to predict `Wastage Food Amount`.
6. Evaluate the held-out test split with MAE, RMSE and R².
7. Save the pipeline, metrics and feature importances under `models/`.
8. Use the trained pipeline on the `/ml-lab` dashboard.

## Run in VS Code (Windows)

Open this `SmartWasteAI` folder in VS Code. Since your computer has Python 3.10 installed at a full path and the `py` launcher may not work, use the Python executable selected in VS Code or this command if needed:

```cmd
"C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv
venv\Scripts\activate
python -m pip install --default-timeout=300 -r requirements.txt
python download_dataset.py
python train_model.py
python app.py
```

If you already have a working `venv` outside this folder, activate that environment instead of creating a second one. From the original layout in the earlier setup, if the venv is in the parent folder, activate it with `..\venv\Scripts\activate`.

Open `http://127.0.0.1:5000` and select **ML Prediction** in the navigation.

## Demo accounts

- Donor: `donor@demo.com` / `1234`
- NGO: `ngo@demo.com` / `1234`
- Admin: `admin@demo.com` / `admin`

## What is real ML here?

The food-waste estimate is produced by a fitted `RandomForestRegressor`, trained from a public CSV and evaluated on a held-out test split. The NGO matching score in the donation workflow is a separate explainable rules-based ranking using distance, expiry urgency, food requirement and capacity; it is **not** an ML classifier.

## Important dataset caveat

The upstream dataset does not clearly document the physical units for the quantity and waste fields. The ML page therefore displays *dataset units*, not kilograms. Metrics are only evidence of performance on this dataset's test split and should not be presented as validated real-world accuracy. Do not use the model to make food-safety decisions. Dataset details: [DATASET_SOURCE.md](DATASET_SOURCE.md).

## GitHub safety

Do not upload your `venv/`, local database, passwords, or secrets. The `.gitignore` file excludes these. The demo login accounts and hard-coded Flask secret are for local demonstration only; replace them and implement password hashing/CSRF protection before public deployment.
