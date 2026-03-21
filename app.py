import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load datasets
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
st.write(data.columns)

# Select columns
data = data[["year", "km_driven", "engine", "max_power", "selling_price"]].dropna()

X = data[["year", "km_driven", "engine", "max_power"]]
y = data["selling_price"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# UI
st.set_page_config(page_title="Car Price Predictor")

st.title("🚗 Used Car Price Prediction")

col1, col2 = st.columns(2)

with col1:
    model_year = st.number_input("Model Year", 2000, 2025, 2018)
    km_driven = st.number_input("Kilometers Driven", 0, 200000, 50000)

with col2:
    engine = st.number_input("Engine (cc)", 500, 5000, 1200)
    max_power = st.number_input("Max Power (bhp)", 20.0, 500.0, 80.0)

if st.button("Predict Price"):
    input_data = pd.DataFrame(
        [[model_year, km_driven, engine, max_power]],
        columns=X.columns
    )

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Car Price: ₹ {prediction:,.0f}")
