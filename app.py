import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import streamlit as st

df = pd.read_csv("Cricketdata.csv")

numeric_cols = ['Strike Rate', 'Bowling Average', 'Matches', 'Runs', 'Wickets', 'Batting Average']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols)
df = df[df['Matches'] > 0]

x = df[['Strike Rate', 'Bowling Average']]
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df['Cluster'] = kmeans.fit_predict(x)

cluster_role_map = {0: 'Bowler', 1: 'Allrounder', 2: 'Batsman'}
df['Role_from_Cluster'] = df['Cluster'].map(cluster_role_map)

df['Selected'] = 0
formation = {'Batsman': 5, 'Bowler': 4, 'Allrounder': 2}
for role, count in formation.items():
    role_players = df[df['Role_from_Cluster'] == role]
    role_players = role_players.sort_values(by='Cluster', ascending=False)
    df.loc[role_players.head(count).index, 'Selected'] = 1

features = ['Matches', 'Runs', 'Wickets', 'Batting Average', 'Bowling Average', 'Strike Rate']
X = df[features]
y = df['Selected']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

reg_runs = RandomForestRegressor(random_state=42)
reg_runs.fit(X, df['Runs'] / df['Matches'])
reg_wickets = RandomForestRegressor(random_state=42)
reg_wickets.fit(X, df['Wickets'] / df['Matches'])

st.title("CricketViz: Cricket Player Prediction App")
st.write("Predicts player role, selection, and performance in next match based on stats.")

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
    cluster = kmeans.predict(player_df[['Strike Rate', 'Bowling Average']])
    player_df['Role_from_Cluster'] = cluster_role_map[cluster[0]]
    player_df['Predicted_Selected'] = clf.predict(player_df[features])
    player_df['Predicted_Runs_Next_Match'] = reg_runs.predict(player_df[features])
    player_df['Predicted_Wickets_Next_Match'] = reg_wickets.predict(player_df[features])
    
    st.subheader("Prediction Results")
    st.write(f"**Role:** {player_df['Role_from_Cluster'][0]}")
    st.write(f"**Selected in Team:** {'Yes' if player_df['Predicted_Selected'][0]==1 else 'No'}")
    st.write(f"**Predicted Runs Next Match:** {player_df['Predicted_Runs_Next_Match'][0]:.2f}")
    st.write(f"**Predicted Wickets Next Match:** {player_df['Predicted_Wickets_Next_Match'][0]:.2f}")
