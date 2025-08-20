import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

df = pd.read_csv("Cricketdata.csv")

numeric_cols = ['Strike Rate', 'Bowling Average', 'Matches', 'Runs', 'Wickets', 'Batting Average']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols)
df = df[df['Matches'] > 0]

features = ['Matches', 'Runs', 'Wickets', 'Batting Average', 'Bowling Average', 'Strike Rate']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

X_role = X_scaled
y_role = df['Role']
role_clf = RandomForestClassifier(random_state=42)
role_clf.fit(X_role, y_role)

df['Selected'] = 0
formation = {'Batsman': 5, 'Bowler': 4, 'All-Rounder': 2}
for role, count in formation.items():
    role_players = df[df['Role'] == role]
    role_players = role_players.sort_values(by='Matches', ascending=False)
    df.loc[role_players.head(count).index, 'Selected'] = 1

X = X_scaled
y = df['Selected']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

reg_runs = RandomForestRegressor(random_state=42)
reg_runs.fit(X, df['Runs'] / df['Matches'])
reg_wickets = RandomForestRegressor(random_state=42)
reg_wickets.fit(X, df['Wickets'] / df['Matches'])

st.title("CricketViz: Cricket Player Prediction App")
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
    player_df = pd.DataFrame([{
        'Name': name,
        'Country': country,
        'Matches': matches,
        'Runs': runs,
        'Wickets': wickets,
        'Batting Average': bat_avg,
        'Bowling Average': bowl_avg,
        'Strike Rate': strike_rate
    }])

    player_scaled = scaler.transform(player_df[features])
    role = role_clf.predict(player_scaled)[0]
    selected = clf.predict(player_scaled)[0]
    predicted_runs = reg_runs.predict(player_scaled)[0]
    predicted_wickets = reg_wickets.predict(player_scaled)[0]

    st.subheader("Prediction Results")
    st.write(f"**Role:** {role}")
    st.write(f"**Selected in Team:** {'Yes' if selected==1 else 'No'}")
    st.write(f"**Predicted Runs Next Match:** {predicted_runs:.2f}")
    st.write(f"**Predicted Wickets Next Match:** {predicted_wickets:.2f}")

    fig, ax = plt.subplots()
    ax.bar(['Predicted Runs', 'Predicted Wickets'], [predicted_runs, predicted_wickets], color=['#FFB3BA','#BAE1FF'])
    ax.set_ylabel("Performance")
    ax.set_title(f"Next Match Performance: {name}")
    st.pyplot(fig)

st.subheader("Average Player Stats by Role")
role_stats = df.groupby('Role')[['Runs','Wickets','Strike Rate']].mean().reset_index()
fig2, ax2 = plt.subplots()
colors = {'Batsman':'#FFB3BA', 'Bowler':'#BAE1FF', 'All-Rounder':'#BAFFC9'}

for col in ['Runs','Wickets','Strike Rate']:
    ax2.plot(role_stats['Role'], role_stats[col], marker='o', label=col)

ax2.set_xlabel("Role")
ax2.set_ylabel("Average Value")
ax2.set_title("Average Stats by Role")
ax2.legend()
st.pyplot(fig2)
