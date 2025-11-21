import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open("model.pkl", "rb"))

st.title("Anomaly / Iris Prediction App")

# Input fields (4 features)
f1 = st.number_input("Feature 1", value=5.1)
f2 = st.number_input("Feature 2", value=3.5)
f3 = st.number_input("Feature 3", value=1.4)
f4 = st.number_input("Feature 4", value=0.2)

if st.button("Predict"):
    data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(data)
    st.success(f"Prediction: {prediction[0]}")
