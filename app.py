import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st

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

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_role_map = {}
for c in df['Cluster'].unique():
    cluster_role_map[c] = df[df['Cluster'] == c]['Role'].mode()[0]

df['Role_from_Cluster'] = df['Cluster'].map(cluster_role_map)
df['Selected'] = 0

formation = {'Batsman': 5, 'Bowler': 4, 'All-Rounder': 2}
for role, count in formation.items():
    role_players = df[df['Role_from_Cluster'] == role]
    role_players = role_players.sort_values(by=['Runs', 'Matches'], ascending=[False, False])
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
    cluster = kmeans.predict(player_scaled)
    role = cluster_role_map[cluster[0]]
    selected = clf.predict(player_scaled)[0]

    base_runs = reg_runs.predict(player_scaled)[0]
    base_wickets = reg_wickets.predict(player_scaled)[0]

    if role == "Batsman":
        predicted_runs = base_runs * 1.4
        predicted_wickets = base_wickets * 0.4
    elif role == "Bowler":
        predicted_runs = base_runs * 0.4
        predicted_wickets = base_wickets * 1.4
    else:
        predicted_runs = base_runs * 1.1
        predicted_wickets = base_wickets * 1.1

    st.subheader("Prediction Results")
    st.write(f"**Role:** {role}")
    st.write(f"**Selected in Team:** {'Yes' if selected == 1 else 'No'}")
    st.write(f"**Predicted Runs Next Match:** {predicted_runs:.2f}")
    st.write(f"**Predicted Wickets Next Match:** {predicted_wickets:.2f}")

    fig, ax = plt.subplots()
    ax.bar(['Predicted Runs', 'Predicted Wickets'],
           [predicted_runs, predicted_wickets],
           color=['#8A2BE2', '#32CD32'])
    ax.set_ylabel("Performance")
    ax.set_title(f"Prediction for {name}")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot([0, matches], [0, runs], marker='o', linewidth=2)
    ax2.set_xlabel("Matches")
    ax2.set_ylabel("Runs")
    ax2.set_title(f"Career Trend: {name}")
    st.pyplot(fig2)
