import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Header
st.title("NFL Points Predictor - Regular Season (2025)")
st.caption("Linear Regression model for team and game point totals using FanDuel and DraftKings lines")

debug = st.checkbox("Debug mode (print intermediate variables)")

# Data
DATA_DIR = os.path.join(os.getcwd(), "nfl", "data")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_Points_week17_copy.csv")

if not os.path.exists(csv_path):
    st.error(f"Dataset not found: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)
df = df.sort_values(by=["Team", "Season", "Week"])
df["Total_Points_Scored"] = df["Tm_Pts"] + df["Opp_Pts"]

# Helper Functions
def rolling_stat(df, group_col, target_col, window):
    return (
        df.groupby(group_col)[target_col]
        .shift(1)
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

# Feature Engineering
df["Tm_Pts_Last1"] = df.groupby("Team")["Tm_Pts"].shift(1)
df["Tm_Pts_Roll3"] = rolling_stat(df, "Team", "Tm_Pts", 3)
df["Tm_Pts_Roll5"] = rolling_stat(df, "Team", "Tm_Pts", 5)

df[["Tm_Pts_Last1", "Tm_Pts_Roll3", "Tm_Pts_Roll5"]] = df[
    ["Tm_Pts_Last1", "Tm_Pts_Roll3", "Tm_Pts_Roll5"]
].fillna(0)

# Model Features
features = [
    "Season", "Week", "Home", "Team", "Opp",
    "Spread", "Total",
    "Tm_Pts_Last1", "Tm_Pts_Roll3", "Tm_Pts_Roll5"
]

target = "Tm_Pts"

categorical_features = ["Team", "Opp"]
numerical_features = [c for c in features if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

X = df[features]
y = df[target]

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write("### Model Performance (Rolling Averages Model)")
st.write(f"**R²:** {r2:.2f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# Prediction Functions
def predict_team_points(season, week, home, team, opponent, spread, total, last1, roll3, roll5):
    input_df = pd.DataFrame([{
        "Season": season,
        "Week": week,
        "Home": home,
        "Team": team,
        "Opp": opponent,
        "Spread": spread,
        "Total": total,
        "Tm_Pts_Last1": last1,
        "Tm_Pts_Roll3": roll3,
        "Tm_Pts_Roll5": roll5,
    }])

    input_processed = preprocessor.transform(input_df)
    return model.predict(input_processed)[0]

def predict_matchups(matchups):
    team_results = []
    game_results = []

    for g in matchups:
        team_pred = predict_team_points(
            g["Season"], g["Week"], g["Home"],
            g["Team"], g["Opp"], g["Spread"],
            g["Total"], g["Last1"], g["Roll3"], g["Roll5"]
        )

        opp_pred = predict_team_points(
            g["Season"], g["Week"], 1 - g["Home"],
            g["Opp"], g["Team"], -g["Spread"],
            g["Total"], g.get("Opp_Last1", 0),
            g.get("Opp_Roll3", 0), g.get("Opp_Roll5", 0)
        )

        team_results.extend([
            {
                "Team": g["Team"],
                "Pred_Pts": round(team_pred, 2),
                "Opp": g["Opp"],
                "Home": g["Home"],
                "Spread": g["Spread"],
            },
            {
                "Team": g["Opp"],
                "Pred_Pts": round(opp_pred, 2),
                "Opp": g["Team"],
                "Home": 1 - g["Home"],
                "Spread": -g["Spread"],
            },
        ])

        game_results.append({
            "Matchup": f"{g['Team']} vs {g['Opp']}",
            "Pred_Total": round(team_pred + opp_pred, 2),
            "Vegas_Total": g["Total"],
            "Diff": round(team_pred + opp_pred - g["Total"], 2),
        })

    return pd.DataFrame(team_results), pd.DataFrame(game_results)
