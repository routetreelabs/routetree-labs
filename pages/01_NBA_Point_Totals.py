import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")

st.title("NBA Points Prediction Model (2026)")
st.caption("Ridge regression model using rolling PPP + pace features")


# Load data

MODEL_PATH_PPP = "nba/models/ppp_ridge_pipeline.joblib"
MODEL_PATH_PACE = "nba/models/pace_ridge_pipeline.joblib"

GAME_PREDS_PATH = "nba/data/nba_todays_predictions.csv"
TEAM_PREDS_PATH = "nba/data/nba_team_predictions.csv"


@st.cache_data
def load_predictions():
    game_df = pd.read_csv(GAME_PREDS_PATH)
    team_df = pd.read_csv(TEAM_PREDS_PATH)
    return game_df, team_df


@st.cache_resource
def load_models():
    ppp_model = joblib.load(MODEL_PATH_PPP)
    pace_model = joblib.load(MODEL_PATH_PACE)
    return ppp_model, pace_model


game_df, team_df = load_predictions()
ppp_model, pace_model = load_models()


# Display

st.subheader("Game-Level Predictions")

display_cols = [
    "Game_Date",
    "Away_Team", "Home_Team",
    "Away_Pred_Pts", "Home_Pred_Pts", "Home_Spread",
    "Total_Pred_Pts", "Sportsbook_Total"
]

st.dataframe(
    game_df[display_cols],
    use_container_width=True
)

st.divider()

st.subheader("Team-Level Predictions")

team_filter = st.selectbox(
    "Filter by team (optional)",
    ["All"] + sorted(team_df["Team"].unique())
)

if team_filter != "All":
    team_df = team_df[team_df["Team"] == team_filter]

st.dataframe(
    team_df.sort_values("Pred_Pts", ascending=False),
    use_container_width=True
)

# Model info

with st.expander("Model Details"):
    st.markdown("""
**Models Used**
- PPP Model: Ridge regression
- Pace Model: Ridge regression

**Features**
- Rolling offensive PPP (1,3,5,10)
- Rolling opponent defense PPP
- Rolling possessions
- Rest days & back-to-back
- One-hot team encoding

**Prediction Logic**
Predicted Points = Predicted PPP × Expected Possessions

Models are retrained daily in Jupyter and persisted with joblib.
""")

st.success("Models and predictions loaded successfully.")
