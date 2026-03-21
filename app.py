import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚗 Car Price Prediction")

year = st.number_input("Year", 2000, 2025)
km_driven = st.number_input("KM Driven")
engine = st.number_input("Engine")
max_power = st.number_input("Max Power")

if st.button("Predict"):
    input_data = pd.DataFrame([[year, km_driven, engine, max_power]],
                              columns=["year", "km_driven", "engine", "max_power"])

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Price: ₹ {prediction:,.0f}")
