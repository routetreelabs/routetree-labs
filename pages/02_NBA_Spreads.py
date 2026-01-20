import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date

# Page config
st.set_page_config(
    page_title="NBA Spread Model",
    layout="wide"
)

st.title("NBA Spread Prediction Model")
st.caption("Logistic Regression")

# Paths
PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "nba_spread_predictions_latest.csv"


# Load predictions
@st.cache_data
def load_predictions():
    if not CSV_PATH.exists():
        return None
    return pd.read_csv(CSV_PATH)


df = load_predictions()


# ---------------- UI ---------------- #

if df is None:
    st.error("No prediction file found.")
    st.info("Run the notebook to generate today's predictions.")
    st.code(str(CSV_PATH))
    st.stop()


# Header info
today = date.today().isoformat()
st.subheader(f"Predictions for {today}")

st.write("Model output:")
st.dataframe(
    df,
    use_container_width=True
)


# Optional filters
with st.expander("🔍 Filters"):
    teams = sorted(set(df["Home"]).union(df["Away"]))
    selected_team = st.selectbox(
        "Filter by team (optional)",
        ["All"] + teams
    )

    if selected_team != "All":
        df = df[(df["Home"] == selected_team) | (df["Away"] == selected_team)]

# Display filtered
st.subheader("Filtered Results")
st.dataframe(df, use_container_width=True)


# Download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    csv,
    "nba_spread_predictions.csv",
    "text/csv"
)


# Footer
st.caption("Model: Logistic Regression | Target: Home team cover")
