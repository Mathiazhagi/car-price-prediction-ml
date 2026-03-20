import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load and combine datasets from data folder
files = [
    "data/bangalore_cars.xlsx",
    "data/chennai_cars.xlsx",
    "data/delhi_cars.xlsx",
    "data/hyderabad_cars.xlsx",
    "data/jaipur_cars.xlsx",
    "data/kolkata_cars.xlsx",
]

df_list = [pd.read_excel(file) for file in files]
data = pd.concat(df_list, ignore_index=True)

# Keep only needed columns
data = data[["model_year", "km_driven", "engine", "max_power", "price"]].dropna()

X = data[["model_year", "km_driven", "engine", "max_power"]]
y = data["price"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Page config
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Prediction")
st.markdown("### Enter car details below")

col1, col2 = st.columns(2)

with col1:
    model_year = st.number_input("Model Year", 2000, 2025, 2018)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)

with col2:
    engine = st.number_input("Engine (cc)", min_value=500, value=1200)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, value=80.0)

st.markdown("---")

if st.button("🚀 Predict Price"):
    input_data = pd.DataFrame(
        [[model_year, km_driven, engine, max_power]],
        columns=X.columns
    )

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ₹ {prediction:,.0f}")