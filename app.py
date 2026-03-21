import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load all uploaded Excel files
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

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Auto-detect useful columns
year_col = [col for col in data.columns if "year" in col][0]
km_col = [col for col in data.columns if "km" in col][0]
engine_col = [col for col in data.columns if "engine" in col][0]
power_col = [col for col in data.columns if "power" in col][0]
price_col = [col for col in data.columns if "price" in col][0]

# Convert text columns like "1197 CC" and "81 bhp" to numbers
data[engine_col] = data[engine_col].astype(str).str.extract(r"(\d+)").astype(float)
data[power_col] = data[power_col].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)

# Keep only needed columns
data = data[[year_col, km_col, engine_col, power_col, price_col]].dropna()

X = data[[year_col, km_col, engine_col, power_col]]
y = data[price_col]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# App UI
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Prediction")
st.write("Enter car details to estimate the selling price.")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2018)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=50000)

with col2:
    engine = st.number_input("Engine (cc)", min_value=500, max_value=5000, value=1200)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=500.0, value=80.0)

if st.button("Predict Price"):
    input_data = pd.DataFrame(
        [[year, km_driven, engine, max_power]],
        columns=[year_col, km_col, engine_col, power_col]
    )

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ₹ {prediction:,.0f}")
