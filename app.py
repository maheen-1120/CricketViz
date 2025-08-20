import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

df = pd.read_csv("Cricketdata.csv")
numeric_cols = ['Matches','Runs','Wickets','Batting Average','Bowling Average','Strike Rate']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols)
df = df[df['Matches'] > 0]

def assign_role(row):
    if row['Runs'] > 2000 and row['Batting Average'] > 30:
        if row['Wickets'] > 50 and row['Bowling Average'] < 35:
            return "All-Rounder"
        return "Batsman"
    elif row['Wickets'] > 50:
        return "Bowler"
    else:
        return "Batsman"
df['Role'] = df.apply(assign_role, axis=1)

formation = {'Batsman': 5, 'Bowler': 4, 'All-Rounder': 2}
df['Selected'] = 0
for role, count in formation.items():
    players = df[df['Role'] == role].sort_values(by=['Runs','Matches'], ascending=False)
    df.loc[players.head(count).index, 'Selected'] = 1

features = ['Matches','Runs','Wickets','Batting Average','Bowling Average','Strike Rate']
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
reg_runs = RandomForestRegressor(random_state=42).fit(X, df['Runs']/df['Matches'])
reg_wkts = RandomForestRegressor(random_state=42).fit(X, df['Wickets']/df['Matches'])

st.title("CricketViz")
st.write("Predicts player role, selection, and performance in next match.")

name = st.text_input("Player Name")
country = st.text_input("Country")
matches = st.number_input("Matches Played", min_value=0)
runs = st.number_input("Total Runs", min_value=0)
wickets = st.number_input("Total Wickets", min_value=0)
bat_avg = st.number_input("Batting Average", min_value=0.0, format="%.2f")
bowl_avg = st.number_input("Bowling Average", min_value=0.0, format="%.2f")
strike_rate = st.number_input("Strike Rate", min_value=0.0, format="%.2f")

if st.button("Predict"):
    player = pd.DataFrame([{
        'Matches': matches,'Runs': runs,'Wickets': wickets,
        'Batting Average': bat_avg,'Bowling Average': bowl_avg,'Strike Rate': strike_rate
    }])
    role = assign_role(player.iloc[0])
    player_scaled = scaler.transform(player[features])
    pred_runs = min(120, reg_runs.predict(player_scaled)[0])
    pred_wkts = min(6, reg_wkts.predict(player_scaled)[0])
    top_players = df[df['Role']==role].sort_values(by=['Runs','Matches'], ascending=False).head(formation[role])
    selected = "Yes" if (runs >= top_players['Runs'].min() and matches >= top_players['Matches'].min()) else "No"

    st.subheader("Prediction Results")
    st.write(f"**Role:** {role}")
    st.write(f"**Selected in Team:** {selected}")
    st.write(f"**Predicted Runs Next Match:** {pred_runs:.2f}")
    st.write(f"**Predicted Wickets Next Match:** {pred_wkts:.2f}")

    plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots()
    ax.bar(['Predicted Runs','Predicted Wickets'], [pred_runs,pred_wkts], color=['#a1c9f4','#ffb482'])
    ax.set_ylabel("Performance")
    ax.set_title(f"Next Match Performance: {name}")
    st.pyplot(fig)

    plt.style.use('seaborn-dark-pastel')
    fig2, ax2 = plt.subplots()
    for role_val in df['Role'].unique():
        subset = df[df['Role']==role_val].sort_values('Matches')
        ax2.plot(subset['Matches'], subset['Runs'], label=role_val)
    ax2.set_xlabel("Matches")
    ax2.set_ylabel("Runs")
    ax2.set_title("Role-wise Performance Trends")
    ax2.legend()
    st.pyplot(fig2)
