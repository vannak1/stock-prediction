from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_models(models, X_test, y_test):
    """
    Evaluate each model and print performance metrics.
    """
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        
        print(f"--- {name} Model Evaluation ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
