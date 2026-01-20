
import streamlit as st

st.set_page_config(page_title="RouteTree Labs Predictive Models", layout="wide")

st.title("RouteTree Labs Sports Prediction Models")
st.caption("RouteTree Labs delivers NFL and NBA predictions powered by machine learning, helping sharp bettors and sports fans make smarter moves backed by real data.")
st.write("""Welcome to the Lab. Choose a model from the left sidebar:

- **2026 NBA Points Predictor** (Ridge Regression)
- **2026 NBA Spreads Predictor** (Logistic Regression)
- **2025 NFL Playoffs Predictions** (Logistic Regression, Linear Regression)
- **2025 NFL Moneyline Predictions** (Logistic Regression)
- **2025 NFL Points Predictor** (Linear Regression)
- **2025 NFL Against the Spread Predictor** (Logistic Regression)
- **2025 NFL Over/Under Predictions** (K-Nearest Neighbors)
""")
