# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str):
    """
    Load the mobile price classification dataset CSV.
    Expects a CSV with columns including:
    ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
     'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
     'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
     'touch_screen', 'wifi', 'price_range']
    """
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Preprocess dataset: split into features & target, scale features, train/test split.
    Returns: X_train, X_test, y_train, y_test
    """
    # Optional: drop or handle features if needed (e.g. if you want to drop 'm_dep' or others)
    # For now, use all features except target
    target_col = "price_range"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # It's often good to scale numerical features
    # Identify which columns are numerical vs categorical
    # For simplicity, scale all numeric features
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data("../data/train.csv")  # or wherever train.csv is located
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Class distribution in training set:\n", y_train.value_counts())
