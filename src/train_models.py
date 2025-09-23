import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_data, preprocess_data

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"Trained {name}")
    return trained_models

def tune_knn(X_train, y_train):
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    print("Best KNN params:", grid.best_params_)
    print("Best KNN score:", grid.best_score_)
    return grid.best_estimator_

def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best RF params:", grid.best_params_)
    print("Best RF score:", grid.best_score_)
    return grid.best_estimator_

if __name__ == "__main__":
    df = load_data("../data/train.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = train_models(X_train, y_train)

    # Tune KNN
    best_knn = tune_knn(X_train, y_train)
    models["KNN"] = best_knn

    # Tune Random Forest
    best_rf = tune_random_forest(X_train, y_train)
    models["Random Forest"] = best_rf

    # Choose which one to save, e.g., the best overall
    # Let's assume KNN is best, but you can compare via validation
    best_model = best_knn  

    with open("../models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("Best model saved at models/best_model.pkl")
