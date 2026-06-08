import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Car MSRP Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🚗 Car MSRP Prediction App")
st.write("Predict the Manufacturer's Suggested Retail Price (MSRP)")

# Sidebar Inputs
st.sidebar.header("Enter Vehicle Details")

year = st.sidebar.number_input("Year", 1990, 2035, 2020)
engine_hp = st.sidebar.number_input("Engine HP", 50, 2000, 300)
engine_cylinders = st.sidebar.number_input("Engine Cylinders", 2, 16, 4)
highway_mpg = st.sidebar.number_input("Highway MPG", 5, 100, 30)
city_mpg = st.sidebar.number_input("City MPG", 5, 100, 22)
popularity = st.sidebar.number_input("Popularity", 0, 10000, 1000)

make = st.sidebar.text_input("Make", "Toyota")
fuel_type = st.sidebar.text_input("Engine Fuel Type", "regular unleaded")
transmission = st.sidebar.text_input("Transmission Type", "AUTOMATIC")
driven_wheels = st.sidebar.text_input("Driven Wheels", "front wheel drive")
vehicle_size = st.sidebar.text_input("Vehicle Size", "Midsize")
vehicle_style = st.sidebar.text_input("Vehicle Style", "Sedan")
market_category = st.sidebar.text_input("Market Category", "Crossover")

# Create Input DataFrame
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

if st.button("Predict MSRP"):

    try:
        # Load original dataset used during training
        train_df = pd.read_csv("car_MSRP.csv")

        target_col = "MSRP (Manufacturer's suggested retail Price)"

        # Same preprocessing as training
        train_df = train_df.drop_duplicates()
        train_df = train_df.dropna(subset=[target_col])

        train_df = train_df.drop(columns=["Model"], errors="ignore")

        num_cols = train_df.select_dtypes(include=["number"]).columns.drop(target_col)
        train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())

        cat_cols = train_df.select_dtypes(include=["object"]).columns
        train_df[cat_cols] = train_df[cat_cols].fillna("Unknown")

        train_df = pd.get_dummies(
            train_df,
            columns=cat_cols,
            drop_first=True
        )

        # Training columns
        training_columns = train_df.drop(columns=[target_col]).columns

        # Encode user input
        input_encoded = pd.get_dummies(input_data)

        # Match training columns exactly
        input_encoded = input_encoded.reindex(
            columns=training_columns,
            fill_value=0
        )

        # Prediction
        prediction = model.predict(input_encoded)[0]

        st.success(
            f"💰 Predicted MSRP: {prediction:,.2f}"
        )

        st.subheader("Input Summary")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
