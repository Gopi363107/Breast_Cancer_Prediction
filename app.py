import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# Load the model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load dataset to get min/max for sliders
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Optionally calculate accuracy if you have test data
# Here, using the same data just for display (not recommended for real use)
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)

# App title
st.title("🩺 Breast Cancer Prediction App")
st.markdown("Predict whether a tumor is Benign or Malignant using ML")
st.markdown("---")

st.subheader("Model Performance")
st.write(f"Accuracy on dataset: {accuracy*100:.2f}%")

# Feature importance chart
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("Feature Importance")
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
plt.figure(figsize=(10,8))
sns.barplot(x=feat_importances, y=feat_importances.index)
st.pyplot(plt)

# Sidebar sliders for patient details
st.sidebar.title("Patient Details")
mean_radius = st.sidebar.slider('Mean Radius', float(X['mean radius'].min()), float(X['mean radius'].max()))
mean_texture = st.sidebar.slider('Mean Texture', float(X['mean texture'].min()), float(X['mean texture'].max()))
mean_perimeter = st.sidebar.slider('Mean Perimeter', float(X['mean perimeter'].min()), float(X['mean perimeter'].max()))
mean_area = st.sidebar.slider('Mean Area', float(X['mean area'].min()), float(X['mean area'].max()))
mean_smoothness = st.sidebar.slider('Mean Smoothness', float(X['mean smoothness'].min()), float(X['mean smoothness'].max()))

# Prepare input for prediction
input_data = np.array([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]).reshape(1, -1)
# Append zeros for remaining features
input_data_scaled = scaler.transform(np.concatenate([input_data, np.zeros((1, X.shape[1]-5))], axis=1))

# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 0:
        st.error("Prediction: Malignant")
        st.metric(label="Prediction", value="Malignant", delta="High Risk")
    else:
        st.success("Prediction: Benign")
        st.metric(label="Prediction", value="Benign", delta="Low Risk")  