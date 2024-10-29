from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from collections import Counter
import numpy as np
import joblib

def prepare_training_data(df):
    """
    Prepare the feature matrix X and target vector y.
    """
    # Define features and target
    features = df.drop(columns=['date', 'future_max', 'target', 'ticker'])
    target = df['target']

    # Select only numeric columns
    features = features.select_dtypes(include=[np.number])

    # Replace infinite values with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    features.dropna(inplace=True)
    target = target.loc[features.index]  # Align target with features

    if features.empty or target.empty:
        print("No features or target data available for training.")
        return None, None, None, None

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )

    return X_train, X_test, y_train, y_test

def compute_scale_pos_weight(y):
    counter = Counter(y)
    majority = max(counter.values())
    minority = min(counter.values())
    return majority / minority

def train_models(X_train, y_train):
    """
    Train multiple classification models.
    """
    # Compute class weights
    class_weight_rf = 'balanced'
    scale_pos_weight = compute_scale_pos_weight(y_train)

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=class_weight_rf
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=scale_pos_weight
        )
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} model trained.")

    return trained_models

def save_models(models):
    for name, model in models.items():
        filename = f"{name}_model.joblib"
        joblib.dump(model, filename)
        print(f"{name} model saved to {filename}")

def load_models(model_names):
    models = {}
    for name in model_names:
        filename = f"{name}_model.joblib"
        model = joblib.load(filename)
        models[name] = model
        print(f"{name} model loaded from {filename}")
    return models
