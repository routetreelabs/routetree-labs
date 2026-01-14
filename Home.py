
import streamlit as st

st.set_page_config(page_title="RouteTree Labs Predictive Models", layout="wide")

st.title("RouteTree Labs Prediction Models")
st.write("""
Welcome to RouteTree Labs. Choose a model from the left sidebar:

- **2026 NBA Points and Spread Predictor** (Ridge Regression)
- **2025 NFL Moneyline Predictions** (Logistic Regression)
- **2025 NFL Over/Under Predictions** (K-Nearest Neighbors)
- **2025 NFL Points Predictor** (Linear Regression)
- **2025 NFL Against the Spread Predictor** (Logistic Regression)
""")
