# main.py

import pandas as pd
import numpy as np
import time
import joblib
from data_collection import get_all_tickers, get_historical_data
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering, label_data
from model_training import prepare_training_data, train_models, save_models, load_models
from model_evaluation import evaluate_models
from imblearn.over_sampling import SMOTE

# In main.py or a new function
def get_new_data(tickers, start_date, end_date):
    new_data = []
    for ticker in tickers:
        print(f"Collecting new data for {ticker}...")
        df = get_historical_data(ticker, start_date, end_date)
        if df.empty:
            print(f"No data for {ticker}")
            continue
        df, _ = preprocess_data(df)
        df = feature_engineering(df)
        if df.empty:
            print(f"No data after feature engineering for {ticker}")
            continue
        df['ticker'] = ticker
        new_data.append(df)
    if new_data:
        new_data_df = pd.concat(new_data, ignore_index=True)
        return new_data_df
    else:
        return pd.DataFrame()

def prepare_prediction_data(df):
    # Ensure the features match the training features
    features = df.drop(columns=['date', 'ticker'], errors='ignore')
    # Select only numeric columns
    features = features.select_dtypes(include=[np.number])
    # Replace infinite values with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    features.dropna(inplace=True)
    return features

def run_predictions(models, features):
    predictions = {}
    for name, model in models.items():
        preds = model.predict(features)
        preds_proba = model.predict_proba(features)[:, 1] if hasattr(model, 'predict_proba') else None
        predictions[name] = {
            'predictions': preds,
            'probabilities': preds_proba
        }
    return predictions

def interpret_predictions(df, predictions):
    for name, preds in predictions.items():
        df[f'{name}_prediction'] = preds['predictions']
        if preds['probabilities'] is not None:
            df[f'{name}_probability'] = preds['probabilities']
    return df

def main():
    # Step 1: Fetch all tickers and save to CSV
    # get_all_tickers()
    
    # Step 2: Read tickers from CSV
    tickers_df = pd.read_csv('tickers.csv')
    tickers = tickers_df['ticker'].tolist()
    
    # Limit tickers for testing
    tickers = tickers[:3000]  # Adjust as needed
    
    all_data = []
    start_date = '2010-01-01'
    end_date = '2022-01-01'  # Training data up to a specific date
    
    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        df = get_historical_data(ticker, start_date, end_date)
        if df.empty:
            print(f"No data for {ticker}")
            continue
        df, _ = preprocess_data(df)
        df = feature_engineering(df)
        df = label_data(df)
        if df.empty:
            print(f"No labeled data for {ticker}")
            continue
        df['ticker'] = ticker
        all_data.append(df)
        time.sleep(0.2)  # Be polite with API rate limits
    
    if not all_data:
        print("No data collected for any tickers.")
        return
    
    # Combine data from all tickers
    full_data = pd.concat(all_data, ignore_index=True)
    
    # Prepare training data
    X_train, X_test, y_train, y_test = prepare_training_data(full_data)
    
    if X_train is None:
        print("Insufficient data for training.")
        return
    
    # Convert labels to integer type
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Apply SMOTE oversampling
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train models
    models = train_models(X_train_resampled, y_train_resampled)
    
    # Save trained models
    save_models(models)
    
    # Evaluate models
    evaluate_models(models, X_test, y_test)
    
    # Prepare new data for prediction
    prediction_start_date = '2024-01-02'
    prediction_end_date = '2024-10-01'
    new_data = get_new_data(tickers, prediction_start_date, prediction_end_date)
    if new_data.empty:
        print("No new data available for prediction.")
        return
    
    # Prepare features for prediction
    prediction_features = prepare_prediction_data(new_data)
    if prediction_features.empty:
        print("No features available for prediction.")
        return
    
    # Load models (optional, if not using models in memory)
    # model_names = ['RandomForest', 'GradientBoosting', 'XGBoost']
    # models = load_models(model_names)
    
    # Run predictions
    predictions = run_predictions(models, prediction_features)
    
    # Interpret and save predictions
    prediction_results = interpret_predictions(new_data, predictions)
    prediction_results.to_csv('prediction_results.csv', index=False)
    print("Predictions saved to prediction_results.csv")
    
    # Optional: Display top predictions
    for name in models.keys():
        predicted_positive = prediction_results[prediction_results[f'{name}_prediction'] == 1]
        print(f"\n{name} Model - Predicted Positive Cases:")
        print(predicted_positive[['date', 'ticker', 'close', f'{name}_probability']].sort_values(by=f'{name}_probability', ascending=False).head(10))
    
if __name__ == "__main__":
    main()
