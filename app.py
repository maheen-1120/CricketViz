import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Cricketdata.csv")

def predict_role(runs, wickets):
    if runs > 2000 and wickets > 100:
        return "All-Rounder"
    elif runs > wickets * 20:
        return "Batsman"
    else:
        return "Bowler"

def predict_selection(bat_avg, bowl_avg):
    if bat_avg > 35 or bowl_avg < 30:
        return "Yes"
    return "No"

def predict_next_match(runs, wickets, matches, bat_avg):
    pred_runs = bat_avg * np.random.uniform(0.8, 1.2)
    pred_wickets = (wickets / (matches + 1)) * np.random.uniform(0.8, 1.2)
    return pred_runs, pred_wickets

st.set_page_config(page_title="CricketViz", page_icon="🏏")
st.title("CricketViz")

player_name = st.selectbox("Select a Player", df["Player Name"].unique())

if not df[df["Player Name"] == player_name].empty:
    player_data = df[df["Player Name"] == player_name].iloc[0]

    pred_role = predict_role(player_data["Total Runs"], player_data["Total Wickets"])
    pred_selection = predict_selection(player_data["Batting Average"], player_data["Bowling Average"])
    pred_runs, pred_wickets = predict_next_match(
        player_data["Total Runs"], player_data["Total Wickets"],
        player_data["Matches Played"], player_data["Batting Average"]
    )

    st.subheader("Prediction Results")
    st.write(f"**Role:** {pred_role}")
    st.write(f"**Selected in Team:** {pred_selection}")
    st.write(f"**Predicted Runs Next Match:** {pred_runs:.2f}")
    st.write(f"**Predicted Wickets Next Match:** {pred_wickets:.2f}")

    st.subheader("Performance Trends")
    sns.set_palette("dark:pastel")
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = {
        "Batting Average": player_data["Batting Average"],
        "Bowling Average": player_data["Bowling Average"],
        "Strike Rate": player_data["Strike Rate"]
    }
    df_plot = pd.DataFrame(metrics, index=[0]).melt(var_name="Metric", value_name="Value")
    sns.barplot(data=df_plot, x="Metric", y="Value", ax=ax, errorbar=None)
    ax.set_title(f"{player_name} - Current Stats")
    st.pyplot(fig)
else:
    st.error("Player not found in dataset.")

