# 📱 Prediction of Mobile Prices

## 📌 Overview

This project builds machine learning models to **predict mobile phone price categories** based on technical specifications such as RAM, storage, battery, and more.
The goal is to classify phones into price ranges using supervised learning, and to compare different ML algorithms on accuracy and efficiency.

---

## 🚀 Models Implemented

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**
* **Naive Bayes**
* **K-Nearest Neighbors (KNN)** → **Best performing model**

---

## 📊 Results

* Achieved **93.5% accuracy with KNN**, outperforming all other classifiers.
* Evaluated with **Accuracy, Precision, Recall, and F1-Score**.
* Applied **hyperparameter tuning** (GridSearchCV) to optimize KNN and Random Forest.

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | \~82%     |
| Decision Tree       | \~86%     |
| Random Forest       | \~90%     |
| Naive Bayes         | \~81%     |
| **KNN**             | **93.5%** |

---

## 📂 Repository Structure

```
pred_of_mobile_prices/
│── data/                 # Dataset or download instructions
│── notebooks/            # Jupyter notebooks for exploration & EDA
│── src/                  
│   ├── data_preprocessing.py   # Load & preprocess data
│   ├── train_models.py         # Train & compare ML models
│   ├── evaluate.py             # Metrics & visualizations
│── results/              # Confusion matrices, feature importance, plots
│── models/               # Saved trained models (pkl files)
│── app.py                # (Optional) Streamlit app for predictions
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/hitesh1705/pred_of_mobile_prices.git
cd pred_of_mobile_prices
pip install -r requirements.txt
```

---

## ▶️ Usage

1. **Preprocess Data**

```bash
python src/data_preprocessing.py
```

2. **Train Models**

```bash
python src/train_models.py
```

3. **Evaluate Performance**

```bash
python src/evaluate.py
```

4. **(Optional) Run App**

```bash
streamlit run app.py
```

---

## 📈 Example Outputs

### Confusion Matrix (KNN)

![Confusion Matrix](results/confusion_matrix_knn.png)

### Feature Correlation

![Feature Correlation](results/correlation_heatmap.png)

---

## 🔮 Future Improvements

* Extend to **regression models** for exact price prediction.
* Deploy best model using **Flask/Streamlit** with an interactive UI.
* Add **explainability** with SHAP/LIME.

---

## 🛠️ Tech Stack

* **Python 3**
* **Scikit-learn, Pandas, NumPy**
* **Matplotlib, Seaborn**
* (Optional) **Streamlit / Flask** for deployment

---

This structure + README will make your repo look like a **case study portfolio project** instead of just a notebook.

---

👉 Do you want me to also **convert your existing notebook into modular scripts** (`data_preprocessing.py`, `train_models.py`, `evaluate.py`) so your repo matches this structure?
