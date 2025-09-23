# app.py - Streamlit demo for Mobile Price Prediction
import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📱 Mobile Price Range Prediction")
st.write("Enter phone specifications to predict price category.")

# Example input features (adjust according to your dataset)
ram = st.slider("RAM (MB)", 256, 8000, 4096)
battery = st.slider("Battery Power (mAh)", 500, 5000, 2500)
internal_memory = st.slider("Internal Memory (GB)", 2, 256, 64)
px_height = st.slider("Pixel Height", 500, 2500, 1200)
px_width = st.slider("Pixel Width", 500, 2500, 1200)

if st.button("Predict Price Range"):
    # Example feature vector (update order to match dataset)
    features = np.array([[ram, battery, internal_memory, px_height, px_width]])
    prediction = model.predict(features)[0]
    st.success(f"💡 Predicted Price Range: {prediction}")
