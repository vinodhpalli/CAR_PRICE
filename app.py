import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car MSRP Predictor", page_icon="🚗")

st.title("🚗 Car MSRP Prediction App")

# Load model
model = joblib.load(r"mlruns/5/models/m-fa915a9df145470796d5d04c157453d5/artifacts/model.pkl")
# Replace with your best model file

st.sidebar.header("Input Features")

year = st.sidebar.number_input("Year", 1990, 2035, 2020)
engine_hp = st.sidebar.number_input("Engine HP", 50, 2000, 300)
engine_cylinders = st.sidebar.number_input("Engine Cylinders", 2, 16, 4)
highway_mpg = st.sidebar.number_input("Highway MPG", 5, 100, 30)
city_mpg = st.sidebar.number_input("City MPG", 5, 100, 22)
popularity = st.sidebar.number_input("Popularity", 0, 10000, 1000)

make = st.sidebar.text_input("Make")
fuel_type = st.sidebar.text_input("Engine Fuel Type")
transmission = st.sidebar.text_input("Transmission Type")
driven_wheels = st.sidebar.text_input("Driven Wheels")
vehicle_size = st.sidebar.text_input("Vehicle Size")
vehicle_style = st.sidebar.text_input("Vehicle Style")
market_category = st.sidebar.text_input("Market Category")

input_df = pd.DataFrame({
    "Year": [year],
    "Engine HP": [engine_hp],
    "Engine Cylinders": [engine_cylinders],
    "highway MPG": [highway_mpg],
    "city mpg": [city_mpg],
    "Popularity": [popularity],
    "Make": [make],
    "Engine Fuel Type": [fuel_type],
    "Transmission Type": [transmission],
    "Driven_Wheels": [driven_wheels],
    "Vehicle Size": [vehicle_size],
    "Vehicle Style": [vehicle_style],
    "Market Category": [market_category]
})

if st.button("Predict MSRP"):
    try:
        prediction = model.predict(input_df)[0]

        st.success(
            f"Predicted MSRP: ${prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")