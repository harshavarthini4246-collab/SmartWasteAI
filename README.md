# SmartWaste AI

An AI-powered food waste prediction and donation management system built with Python, Flask, and Machine Learning.

## Features

* Food waste prediction using Random Forest Regression
* Public restaurant food-wastage dataset
* Model evaluation using MAE, RMSE, and R²
* ML metrics and feature-importance outputs
* Food donation management interface
* NGO and donor workflows

## Machine Learning

The model is trained using a public dataset and evaluated on a held-out test set.

* Algorithm: RandomForestRegressor
* Clean dataset rows: 1,618
* Training rows: 1,294
* Test rows: 324
* MAE: 1.858 dataset units
* RMSE: 2.938 dataset units
* R²: 0.9199

These metrics reflect evaluation on the public dataset and are not a real-world validation study.

## Setup

1. Install Python 3.10.

2. Create and activate a virtual environment.

3. Install dependencies:

   `pip install -r requirements.txt`

4. Download the dataset:

   `python download_dataset.py`

5. Train the model:

   `python train_model.py`

6. Run the application:

   `python app.py`

7. Open `http://127.0.0.1:5000` in your browser.

## Project Structure

* `app.py` — Flask application
* `ml_engine.py` — Machine learning integration
* `train_model.py` — Model training and evaluation
* `download_dataset.py` — Dataset downloader
* `data/` — Dataset files
* `models/` — Saved model metrics and artifacts
* `templates/` — HTML templates
* `static/` — CSS and JavaScript

## Dataset

Refer to `DATASET_SOURCE.md` for the dataset source and related details.

## Disclaimer

Predictions are estimates based on the training dataset. Verify predictions against real-world data before operational use.
