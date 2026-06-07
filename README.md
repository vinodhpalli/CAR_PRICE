# 🚗 Car MSRP Prediction

A Machine Learning web application that predicts the Manufacturer's Suggested Retail Price (MSRP) of a car based on vehicle specifications. The project uses regression algorithms, hyperparameter tuning with Optuna, experiment tracking with MLflow, and deployment using Streamlit.

## 📌 Features

- Predict car MSRP instantly
- Interactive Streamlit web interface
- Multiple regression models evaluated
- Hyperparameter optimization using Optuna
- Experiment tracking with MLflow
- Model deployment with Streamlit Cloud

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Optuna
- MLflow
- Joblib
- Streamlit

## 📊 Dataset

The dataset contains vehicle specifications such as:

- Make
- Year
- Engine HP
- Engine Cylinders
- Engine Fuel Type
- Transmission Type
- Driven Wheels
- Vehicle Size
- Vehicle Style
- Highway MPG
- City MPG
- Popularity

Target Variable:

- MSRP (Manufacturer's Suggested Retail Price)

## 🤖 Models Evaluated

- Ridge Regression
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)

## 🚀 Streamlit Deployment

Live App:

https://carprice-k6awt7pyy5kh7de6nujlgd.streamlit.app/

## 📂 Project Structure

```text
CAR_PRICE/
│
├── app.py
├── model.pkl
├── car_MSRP.csv
├── requirements.txt
├── README.md
└── .gitignore
