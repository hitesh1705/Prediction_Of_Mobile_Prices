# src/evaluate.py

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import load_data, preprocess_data
from train_models import train_models

def evaluate_models(models: dict, X_test, y_test, results_dir="../results"):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n==== {name} ====")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        fname = f"{results_dir}/confusion_matrix_{name.replace(' ', '_')}.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved confusion matrix at {fname}")

def plot_feature_importances(model, feature_names, results_dir="../results"):
    # Only applicable for tree-based models
    import numpy as np

    if hasattr(model, "feature_importances_"):
        import matplotlib.pyplot as plt
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10,6))
        sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices])
        plt.title("Feature Importances")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"{results_dir}/feature_importances.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved feature importances at {fname}")
    else:
        print("Model has no feature_importances_ attribute; skipping feature importances")

if __name__ == "__main__":
    df = load_data("../data/train.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

    # Load best model and plot feature importances
    with open("../models/best_model.pkl", "rb") as f:
        best_model = pickle.load(f)
    # Feature names after preprocessing
    feature_names = list(X_train.columns)
    plot_feature_importances(best_model, feature_names)
