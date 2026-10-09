"""Train and evaluate SmartWaste AI's food-waste regression model."""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "food_wastage_data.csv"
MODEL_DIR = ROOT / "models"
TARGET = "Wastage Food Amount"
NUMERIC = ["Number of Guests", "Quantity of Food"]
CATEGORICAL = ["Type of Food", "Event Type", "Storage Conditions", "Purchase History", "Seasonality", "Preparation Method", "Geographical Location", "Pricing"]
FEATURES = NUMERIC + CATEGORICAL


def clean_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python download_dataset.py")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[FEATURES + [TARGET]].copy()
    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET, "Number of Guests", "Quantity of Food"])
    df = df[(df["Number of Guests"] > 0) & (df["Quantity of Food"] > 0) & (df[TARGET] >= 0)]
    # Wastage cannot logically exceed prepared quantity for this project target.
    df = df[df[TARGET] <= df["Quantity of Food"]]
    for col in CATEGORICAL:
        df[col] = df[col].astype("object").map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
        df.loc[df[col].isin(["", "nan", "None"]), col] = np.nan
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < 50:
        raise ValueError(f"Only {len(df)} usable rows remain. Check the dataset source/format.")
    return df


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_dataset(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    metrics = {
        "dataset_name": "Restaurant Food Wastage Data",
        "dataset_source": "https://github.com/Ashutosh-dutta07/Smart-Food-Waste-Intelligence",
        "dataset_rows_raw": int(pd.read_csv(DATA_PATH, encoding="utf-8-sig").shape[0]),
        "dataset_rows_clean": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "model": "RandomForestRegressor",
        "test_mae": round(float(mean_absolute_error(y_test, pred)), 3),
        "test_rmse": round(rmse, 3),
        "test_r2": round(float(r2_score(y_test, pred)), 4),
        "test_median_absolute_error": round(float(np.median(np.abs(y_test.to_numpy() - pred))), 3),
        "note": "Metrics describe this public dataset's held-out test split only. They do not guarantee accuracy on a particular local restaurant or donation operation.",
    }
    joblib.dump({"pipeline": model, "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL, "target": TARGET}, MODEL_DIR / "food_waste_pipeline.joblib")
    (MODEL_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Export readable importance values from the fitted Random Forest pipeline.
    names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["regressor"].feature_importances_
    importance_df = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
    importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print("\n=== SmartWaste AI ML training complete ===")
    print(f"Clean rows: {len(df)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Model: {metrics['model']}")
    print(f"MAE : {metrics['test_mae']} dataset units")
    print(f"RMSE: {metrics['test_rmse']} dataset units")
    print(f"R²  : {metrics['test_r2']}")
    print(f"Saved model: {MODEL_DIR / 'food_waste_pipeline.joblib'}")
    print(f"Saved metrics: {MODEL_DIR / 'model_metrics.json'}")
    print("Important: these metrics are from the public dataset, not a local real-world validation study.")


if __name__ == "__main__":
    main()
