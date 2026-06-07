import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Car MSRP Prediction",
    page_icon="🚗",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Title
st.title("🚗 Car MSRP Prediction App")
st.markdown("Predict the Manufacturer's Suggested Retail Price (MSRP) of a car.")

# Sidebar Inputs
st.sidebar.header("Enter Vehicle Details")

year = st.sidebar.number_input("Year", min_value=1990, max_value=2035, value=2020)
engine_hp = st.sidebar.number_input("Engine HP", min_value=50, max_value=2000, value=300)
engine_cylinders = st.sidebar.number_input("Engine Cylinders", min_value=2, max_value=16, value=4)
highway_mpg = st.sidebar.number_input("Highway MPG", min_value=5, max_value=100, value=30)
city_mpg = st.sidebar.number_input("City MPG", min_value=5, max_value=100, value=22)
popularity = st.sidebar.number_input("Popularity", min_value=0, max_value=10000, value=1000)

make = st.sidebar.text_input("Make", "Toyota")
fuel_type = st.sidebar.text_input("Engine Fuel Type", "Regular Unleaded")
transmission = st.sidebar.text_input("Transmission Type", "AUTOMATIC")
driven_wheels = st.sidebar.text_input("Driven Wheels", "front wheel drive")
vehicle_size = st.sidebar.text_input("Vehicle Size", "Midsize")
vehicle_style = st.sidebar.text_input("Vehicle Style", "Sedan")
market_category = st.sidebar.text_input("Market Category", "Crossover")

# Input Data
input_data = pd.DataFrame({
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

# Prediction
if st.button("Predict MSRP", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]

        st.success(
            f"💰 Predicted MSRP: ${prediction:,.2f}"
        )

        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
