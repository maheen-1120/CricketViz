import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"Cricketdata.csv")

def predict_role(row):
    if row['Total Wickets'] >= 150 and row['Batting Average'] < 25:
        return "Bowler"
    elif row['Total Runs'] >= 4000 and row['Batting Average'] >= 35:
        return "Batsman"
    elif row['Total Runs'] >= 2000 and row['Total Wickets'] >= 50:
        return "All-Rounder"
    else:
        return "Batsman" if row['Batting Average'] > row['Bowling Average'] else "Bowler"

df['Predicted Role'] = df.apply(predict_role, axis=1)

def is_selected(row):
    role = row['Predicted Role']
    bat_avg = row['Batting Average']
    bowl_avg = row['Bowling Average']
    sr = row['Strike Rate']
    wickets = row['Total Wickets']
    if role in ['Batsman', 'All-Rounder']:
        if bat_avg >= 35 and sr >= 120:
            return "Yes"
    if role in ['Bowler', 'All-Rounder']:
        if bowl_avg <= 30 and wickets >= 50:
            return "Yes"
    weighted_score = (bat_avg/100) + (sr/200) + (wickets/100) - (bowl_avg/100)
    if weighted_score >= df['Batting Average'].median()/100:
        return "Yes"
    return "No"

df['Selected in Team'] = df.apply(is_selected, axis=1)

st.title("🏏 Cricket Player Role Prediction & Selection")

player_name = st.selectbox("Select a Player", df['Player Name'].unique())
player = df[df['Player Name'] == player_name].iloc[0]

st.subheader(f"📊 Player Details: {player_name}")
st.write(f"**Country:** {player['Country']}")
st.write(f"**Matches Played:** {player['Matches Played']}")
st.write(f"**Runs:** {player['Total Runs']}")
st.write(f"**Wickets:** {player['Total Wickets']}")
st.write(f"**Batting Avg:** {player['Batting Average']}")
st.write(f"**Bowling Avg:** {player['Bowling Average']}")
st.write(f"**Strike Rate:** {player['Strike Rate']}")
st.write(f"### 🏷️ Predicted Role: {player['Predicted Role']}")
st.write(f"### ✅ Selected in Team: {player['Selected in Team']}")

st.write("### 📈 Performance Trends by Role")
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="Matches Played", y="Total Runs", hue="Predicted Role", palette="dark:pastel")
plt.title("Performance Trends (Dark Pastel Palette)")
st.pyplot(plt)

st.write("### 📊 Selection Distribution")
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Selected in Team", palette="Set2")
plt.title("Selected vs Not Selected Players")
st.pyplot(plt)

st.write("### 📋 Dataset Preview")
st.dataframe(df.head())
