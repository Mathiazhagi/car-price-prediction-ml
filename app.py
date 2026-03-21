import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("data/car_data.csv")  # make sure this file exists

# Clean column names
data.columns = data.columns.str.strip()

# Convert to numeric
data["engine"] = data["engine"].astype(str).str.extract('(\d+)').astype(float)
data["max_power"] = data["max_power"].astype(str).str.extract('(\d+)').astype(float)

# Drop missing values
data = data.dropna()

# Select features
X = data[["year", "km_driven", "engine", "max_power"]]
y = data["selling_price"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# UI
st.title("🚗 Car Price Prediction")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", 2000, 2025, 2018)
    km_driven = st.number_input("Kilometers Driven", 0, 200000, 50000)

with col2:
    engine = st.number_input("Engine (cc)", 500, 5000, 1200)
    max_power = st.number_input("Max Power (bhp)", 20.0, 500.0, 80.0)

if st.button("Predict Price"):
    input_data = pd.DataFrame(
        [[year, km_driven, engine, max_power]],
        columns=["year", "km_driven", "engine", "max_power"]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Price: ₹ {prediction:,.0f}")
