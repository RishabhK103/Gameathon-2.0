import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pulp

# Step 1: Load and Preprocess Historical Data
df = pd.read_csv("dream11_dataset_with_rolling_updated.csv")

# Add long-term and distributional features
df["career_avg_points"] = df.groupby("player")["points"].transform("mean")
df["std_runs"] = df.groupby("player")["runs"].transform("std").fillna(0)
df["max_wickets"] = df.groupby("player")["wickets"].transform("max")
df["freq_50_plus_runs"] = df.groupby("player")["runs"].transform(lambda x: (x >= 50).mean())
df["avg_points_vs_opposition"] = df.groupby(["player", "opposition"])["points"].transform("mean").fillna(df["career_avg_points"])
df["avg_points_at_venue"] = df.groupby(["player", "venue"])["points"].transform("mean").fillna(df["career_avg_points"])
df["avg_wickets_at_venue"] = df.groupby(["player", "venue"])["wickets"].transform("mean").fillna(df.groupby("player")["wickets"].transform("mean"))

# Weighted rolling average
def weighted_rolling_avg(group, col):
    weights = np.array([1, 1.5, 2, 2.5, 3])
    return group[col].rolling(5, min_periods=1).apply(lambda x: np.average(x[-5:], weights=weights[:len(x)]), raw=True)

for col in ["points", "runs", "wickets"]:
    df[f"wavg_{col}_last_5"] = df.groupby("player").apply(lambda x: weighted_rolling_avg(x, col)).reset_index(level=0, drop=True).fillna(0)

# Moderate batting feature boost
df["wavg_runs_last_5"] *= 2.0
df["freq_50_plus_runs"] *= 2.0

# Preprocess historical data
numerical_cols = ["wavg_points_last_5", "wavg_runs_last_5", "wavg_wickets_last_5", 
                  "career_avg_points", "std_runs", "max_wickets", "freq_50_plus_runs",
                  "avg_points_vs_opposition", "avg_points_at_venue", "avg_wickets_at_venue"]
categorical_cols = ["venue", "opposition", "team", "role"]

print("Checking for NaN/Inf in input data...")
print("NaN in numerical cols:", df[numerical_cols].isna().sum().sum())
print("Inf in numerical cols:", np.isinf(df[numerical_cols]).sum().sum())
print("NaN in categorical cols:", df[categorical_cols].isna().sum().sum())

# Fill NaN values and clip extreme values
df[numerical_cols] = df[numerical_cols].fillna(0)
df["points"] = df["points"].fillna(0).clip(0, 1000)

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = encoder.fit_transform(df[categorical_cols])
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols))

X = pd.concat([df[numerical_cols], cat_encoded_df], axis=1)
y = np.log1p(df["points"])  # Log-transform target

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("NaN in X_train:", X_train.isna().sum().sum())
print("NaN in y_train:", y_train.isna().sum())
print("Inf in X_train:", np.isinf(X_train).sum().sum())
print("Inf in y_train:", np.isinf(y_train).sum())

# Build and train NN
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1)  # No relu
])
model.compile(optimizer=Adam(learning_rate=0.00005), loss="mae")
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=1, callbacks=[lr_scheduler])

# Debug prediction range
y_pred = np.expm1(model.predict(X_test).flatten())
print("Predicted range:", y_pred.min(), y_pred.max())
print("Actual range:", df["points"].min(), df["points"].max())

# Step 2: Function to Predict and Optimize Team
def predict_best_xi(squad_df, venue, team1, team2):
    squad_df = squad_df[squad_df["IsPlaying"] == "PLAYING"].copy()
    squad_df = squad_df.rename(columns={"Player Name": "Player", "Player Type": "role", "Team": "team"})
    squad_df["venue"] = venue
    squad_df["opposition"] = squad_df["team"].apply(lambda x: team2 if x == team1 else team1)

    # Load alias mapping
    alias_df = pd.read_csv("/home/parth/iplgameathon/fantasy-sports-machine-learning/squad.csv")
    alias_map = dict(zip(alias_df["Player Name"].str.lower(), alias_df["ESPN player name"].str.lower()))
    squad_df["espn_player_name"] = squad_df["Player"].str.lower().map(lambda x: alias_map.get(x, x))

    # Get latest historical data
    latest_historical = df.groupby("player").tail(1)[["player"] + numerical_cols + ["team", "role"]].reset_index(drop=True)
    latest_historical["player_lower"] = latest_historical["player"].str.lower()

    # Merge with latest historical data
    squad_df = squad_df.merge(latest_historical, 
                              left_on="espn_player_name", right_on="player_lower", 
                              how="left", suffixes=("_squad", "_hist"))

    # Compute team/role-based averages for fallback
    team_role_means = df.groupby(["team", "role"])[numerical_cols].mean().reset_index()
    for col in numerical_cols:
        squad_df[col] = squad_df[col].fillna(
            squad_df.merge(team_role_means, left_on=["team_squad", "role_squad"], right_on=["team", "role"], how="left")[col + "_y"]
        ).fillna(df[col].mean())

    # Debugging unmatched players
    unmatched = squad_df[squad_df["espn_player_name"].isna() | squad_df[numerical_cols[0]].isna()]["Player"].tolist()
    if unmatched:
        print(f"Warning: No historical data found for players: {unmatched}")

    # Use raw squad roles (assume historical data uses BAT, BOWL, etc.)
    squad_df["role"] = squad_df["role_squad"]

    # Map unseen categorical values
    for col in ["venue", "opposition", "team_squad", "role"]:
        unseen = set(squad_df[col]) - set(encoder.categories_[categorical_cols.index(col if col != "team_squad" else "team")])
        if unseen:
            print(f"Unseen {col} values: {unseen}. Mapping to most frequent category.")
            most_frequent = df[col if col != "team_squad" else "team"].mode()[0]
            squad_df[col] = squad_df[col].replace(list(unseen), most_frequent)

    # Rename team_squad to team
    squad_df["team"] = squad_df["team_squad"]

    # Preprocess squad data
    squad_numerical = scaler.transform(squad_df[numerical_cols])
    squad_cat_encoded = encoder.transform(squad_df[["venue", "opposition", "team", "role"]])
    squad_features = np.hstack([squad_numerical, squad_cat_encoded])

    # Predict points with balanced scaling
    raw_predictions = np.expm1(model.predict(squad_features).flatten())
    squad_df["Raw_Score"] = raw_predictions  # Store raw predictions
    squad_df["Score"] = squad_df.apply(
        lambda row: row["Raw_Score"] * 10.0 if row["role_squad"] in ["BAT", "WK"] else row["Raw_Score"] * 8.0, axis=1
    )

    # Optimization with WK constraint
    total_players = 11
    batters = squad_df[squad_df["role_squad"].str.strip().str.upper() == "BAT"]
    bowlers = squad_df[squad_df["role_squad"].str.strip().str.upper() == "BOWL"]
    allrounders = squad_df[squad_df["role_squad"].str.strip().str.upper() == "ALL"]
    keepers = squad_df[squad_df["role_squad"].str.strip().str.upper() == "WK"]

    print(f"Available: Batters={len(batters)}, Bowlers={len(bowlers)}, Keepers={len(keepers)}, Allrounders={len(allrounders)}")
    if len(batters) < 4 or len(bowlers) < 3 or len(keepers) < 1:
        print("Not enough players for constraints.")
        return None

    prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
    players = squad_df.index.tolist()
    x = pulp.LpVariable.dicts("player", players, cat="Binary")
    prob += pulp.lpSum([x[i] * squad_df.loc[i, "Score"] for i in players])
    prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
    prob += pulp.lpSum([x[i] for i in batters.index]) >= 4, "Min_Batters"
    prob += pulp.lpSum([x[i] for i in bowlers.index]) >= 3, "Min_Bowlers"
    prob += pulp.lpSum([x[i] for i in keepers.index]) == 1, "Exactly_One_Keeper"
    prob += pulp.lpSum([x[i] for i in squad_df[squad_df["team_squad"] == team1].index]) >= 1, f"Min_from_{team1}"
    prob += pulp.lpSum([x[i] for i in squad_df[squad_df["team_squad"] == team2].index]) >= 1, f"Min_from_{team2}"

    prob.solve()
    if pulp.LpStatus[prob.status] != "Optimal":
        print("No solution found!")
        return None

    selected = [i for i in players if pulp.value(x[i]) == 1]
    selected_df = squad_df.loc[selected].copy()
    selected_df.sort_values("Score", ascending=False, inplace=True)

    # Custom captain/vice-captain: Prioritize BAT/ALL, allow top performers
    selected_df["Role_In_Team"] = "Player"
    bat_all_indices = selected_df[selected_df["role_squad"].isin(["BAT", "ALL"])].index
    if len(bat_all_indices) > 0:
        captain_idx = selected_df.loc[bat_all_indices, "Score"].idxmax()
        selected_df.loc[captain_idx, "Role_In_Team"] = "Captain"
        if len(bat_all_indices) > 1:
            vice_captain_idx = selected_df.loc[bat_all_indices.drop(captain_idx), "Score"].idxmax()
            selected_df.loc[vice_captain_idx, "Role_In_Team"] = "Vice Captain"
    else:
        selected_df.iloc[0, selected_df.columns.get_loc("Role_In_Team")] = "Captain"
        selected_df.iloc[1, selected_df.columns.get_loc("Role_In_Team")] = "Vice Captain"

    return selected_df[["Player", "team_squad", "role_squad", "Score", "Role_In_Team"]].rename(columns={"team_squad": "team", "role_squad": "role"})

# Step 3: Example Usage
squad_df = pd.read_csv("/home/parth/Downloads/LSG.csv")
venue = "Eden Gardens, Kolkata"
team1 = "GT"
team2 = "PBKS"
result = predict_best_xi(squad_df, venue, team1, team2)

if result is not None:
    print("Best Playing XI:")
    print(result)
else:
    print("Failed to generate a team.")