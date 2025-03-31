import pandas as pd

# Load 2025 deliveries
df = pd.read_csv("../data/ipl/ipl_2025_deliveries.csv")
df["match_id"] = pd.to_datetime(df["date"], format="%b %d, %Y").dt.strftime("%Y-%m-%d")

# Aggregate stats (unchanged)
batting = df.groupby(["match_id", "striker", "batting_team", "bowling_team", "venue", "season"]).agg(
    runs=pd.NamedAgg("runs_of_bat", "sum"),
    balls_faced=pd.NamedAgg("over", lambda x: len(x) - df.loc[x.index, ["wide", "noballs"]].sum().sum()),
    fours=pd.NamedAgg("runs_of_bat", lambda x: (x == 4).sum()),
    sixes=pd.NamedAgg("runs_of_bat", lambda x: (x == 6).sum())
).reset_index()

bowling = df.groupby(["match_id", "bowler", "bowling_team", "batting_team", "venue", "season"]).agg(
    wickets=pd.NamedAgg("wicket_type", lambda x: x.notna().sum()),
    runs_conceded=pd.NamedAgg("runs_of_bat", lambda x: x.sum() + df.loc[x.index, "extras"].sum()),
    balls_bowled=pd.NamedAgg("over", "count"),
    dot_balls=pd.NamedAgg("runs_of_bat", lambda x: ((x + df.loc[x.index, "extras"]) == 0).sum())
).reset_index()

fielding = df[df["wicket_type"].notna()].groupby(["match_id", "fielder"]).agg(
    catches=pd.NamedAgg("wicket_type", lambda x: (x == "caught").sum()),
    stumpings=pd.NamedAgg("wicket_type", lambda x: (x == "stumped").sum()),
    run_outs=pd.NamedAgg("wicket_type", lambda x: (x == "runout").sum())
).reset_index()

players = pd.concat([
    batting.rename(columns={"striker": "player", "batting_team": "team", "bowling_team": "opposition"}),
    bowling.rename(columns={"bowler": "player", "bowling_team": "team", "batting_team": "opposition"})
]).groupby(["match_id", "player", "team", "opposition", "venue", "season"]).sum().reset_index()

players = players.merge(fielding.rename(columns={"fielder": "player"}), on=["match_id", "player"], how="left").fillna(0)

# Corrected points calculation
def calculate_points(row):
    points = (row["runs"] * 1 + row["fours"] * 4 + row["sixes"] * 6 + 
              row["wickets"] * 25 + row["dot_balls"] * 1 + 
              row["catches"] * 8 + row["stumpings"] * 12 + row["run_outs"] * 12)
    if row["runs"] >= 100:
        points += 16
    elif row["runs"] >= 75:
        points += 12
    elif row["runs"] >= 50:
        points += 8
    elif row["runs"] >= 25:
        points += 4
    if row["balls_faced"] >= 10:
        sr = (row["runs"] / row["balls_faced"]) * 100
        if sr > 170:
            points += 6
        elif sr > 150:
            points += 4
        elif sr >= 130:
            points += 2
    if row["balls_bowled"] >= 12:
        economy = row["runs_conceded"] / (row["balls_bowled"] / 6)
        if economy < 5:
            points += 6
        elif 5 <= economy <= 5.99:
            points += 4
        elif 6 <= economy <= 7:
            points += 2
    if row["wickets"] >= 5:
        points += 12
    elif row["wickets"] == 4:
        points += 8
    elif row["wickets"] == 3:
        points += 4
    return points + 4

players["points"] = players.apply(calculate_points, axis=1)
players["role"] = "Unknown"

# Load existing and combine
existing = pd.read_csv("dream11_dataset_with_rolling.csv")
combined = pd.concat([existing, players]).sort_values("match_id")
combined["points"] = combined.apply(calculate_points, axis=1)  # Recompute all points

# Rolling averages
for col in ["points", "runs", "wickets"]:
    combined[f"avg_{col}_last_5"] = combined.groupby("player")[col].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )

combined.fillna(0, inplace=True)
combined.to_csv("dream11_dataset_with_rolling_updated.csv", index=False)

# Debug
print("Top High Scorers:")
print(combined[combined["points"] > 200][["match_id", "player", "runs", "wickets", "points"]].head(10))
print("\nQ de Kock 2022-05-18:")
print(combined[(combined["player"] == "Q de Kock") & (combined["match_id"] == "2022-05-18")])