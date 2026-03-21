import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Prediction")
st.write("Enter car details to estimate the car price.")

# Load cleaned dataset
data = pd.read_csv("final_cleaned_data.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Keep only required columns
required_cols = ["model_year", "mileage", "engine", "max_power", "price"]
data = data[required_cols].dropna()

# Convert numeric columns safely
for col in required_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

X = data[["model_year", "mileage", "engine", "max_power"]]
y = data["price"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Input form
col1, col2 = st.columns(2)

with col1:
    model_year = st.number_input("Model Year", min_value=2000, max_value=2025, value=2018)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=18.0)

with col2:
    engine = st.number_input("Engine (cc)", min_value=500, max_value=5000, value=1200)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=500.0, value=80.0)

if st.button("Predict Price"):
    input_data = pd.DataFrame(
        [[model_year, mileage, engine, max_power]],
        columns=["model_year", "mileage", "engine", "max_power"]
    )

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ₹ {prediction:,.0f}")
