import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("car_data.csv")

# Simple preprocessing (adjust if needed)
data = data.dropna()

# Example features (modify based on your dataset)
X = data.drop("price", axis=1)
y = data["price"]

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)


# Page config
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Prediction")
st.markdown("### Enter car details below")

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", 2000, 2025)
    km_driven = st.number_input("Kilometers Driven")

with col2:
    engine = st.number_input("Engine (cc)")
    max_power = st.number_input("Max Power (bhp)")


st.markdown("---")

if st.button("🚀 Predict Price"):
    input_data = pd.DataFrame([[year, km_driven, engine, max_power]],
                              columns=X.columns)

    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")

    input_df = pd.DataFrame([input_dict])

    categorical_values = {
        "fuel_type": fuel_type,
        "body_type": body_type,
        "transmission": transmission,
        "brand": brand,
        "city": city
    }

    for col, val in categorical_values.items():
        input_df[f"{col}_{val}"] = 1

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    predicted_log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(predicted_log_price)

    st.success(f"💰 Estimated Car Price: ₹ {predicted_price:,.0f}")