import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

columns = [
    "Player Name", "Role", "Country", "Matches Played",
    "Total Runs", "Total Wickets", "Batting Average",
    "Bowling Average", "Strike Rate"
]

df = pd.read_csv(
    r"C:\Users\SOHAM\OneDrive\Documents\internship_AIML\data.csv",
    names=columns
)

def predict_role(row):
    if row["Total Runs"] > 2000 and row["Total Wickets"] > 100:
        return "All-Rounder"
    elif row["Total Runs"] > row["Total Wickets"] * 20:
        return "Batsman"
    else:
        return "Bowler"

def predict_selection(row):
    if row["Batting Average"] > 35 or row["Bowling Average"] < 30:
        return "Yes"
    return "No"

def predict_next_match(row):
    runs = row["Batting Average"] * np.random.uniform(0.8, 1.2)
    wickets = row["Total Wickets"] / (row["Matches Played"] + 1) * np.random.uniform(0.8, 1.2)
    return runs, wickets

st.title("Cricket Player Performance Predictor")

player_name = st.selectbox("Select a Player", df["Player Name"].unique())

player_data = df[df["Player Name"] == player_name].iloc[0]

pred_role = predict_role(player_data)
pred_selection = predict_selection(player_data)
pred_runs, pred_wickets = predict_next_match(player_data)

st.subheader("Prediction Results")
st.write(f"**Role:** {pred_role}")
st.write(f"**Selected in Team:** {pred_selection}")
st.write(f"**Predicted Runs Next Match:** {pred_runs:.2f}")
st.write(f"**Predicted Wickets Next Match:** {pred_wickets:.2f}")

st.subheader("Performance Trends")

sns.set_palette("dark:pastel")
fig, ax = plt.subplots(figsize=(8, 5))
df_plot = df[df["Player Name"] == player_name][["Matches Played", "Total Runs", "Total Wickets"]]
df_plot = df_plot.melt("Matches Played", var_name="Metric", value_name="Value")

sns.lineplot(data=df_plot, x="Matches Played", y="Value", hue="Metric", ax=ax)
ax.set_title(f"{player_name} - Performance Trends")
st.pyplot(fig)
