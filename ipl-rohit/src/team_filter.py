import pandas as pd

def normalize_columns(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    numeric_cols = ["mat", "inns", "no", "runs", "bf", "100", "50", "0", "4s", "6s", "hs",
                    "overs", "mdns", "wkts", "4", "5", "dis", "ct", "st", "ct_wk", "ct_fi", "md", "credits"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def filter_match_players(squad_file, team1, team2, output_file):
    squad_df = pd.read_csv(squad_file)
    squad_df = normalize_columns(squad_df)
    
    match_df = squad_df[(squad_df["team"].isin([team1, team2])) & (squad_df["isplaying"] == "PLAYING")]
    
    required_cols = ["player_name", "team", "player_type", "credits", "isplaying"]
    for col in required_cols:
        if col not in match_df.columns:
            raise ValueError(f"Column {col} missing in squad data")
    
    try:
        stats_df = pd.read_csv("../data/merged_data.csv")
        match_df = match_df.merge(stats_df[["player_name", "ave_bat", "ave_bowl", "dis", "mat_bat", "mat_bowl", "mat", "runs_bat", "wkts"]], 
                                 on="player_name", how="left")
    except FileNotFoundError:
        print("Warning: merged_data.csv not found; proceeding with squad data only")
    
    match_df["mat_bat"] = match_df["mat_bat"].fillna(20)
    match_df["mat_bowl"] = match_df["mat_bowl"].fillna(20)
    match_df["mat"] = match_df["mat"].fillna(20)
    match_df["runs_bat"] = match_df["runs_bat"].fillna(match_df["player_type"].map({
        "BAT": 600,
        "ALL": 500,
        "WK": 500,
        "BOWL": 100
    }))
    match_df["wkts"] = match_df["wkts"].fillna(match_df["player_type"].map({
        "BOWL": 30,
        "ALL": 20,
        "BAT": 0,
        "WK": 0
    }))
    match_df["dis"] = match_df["dis"].fillna(match_df["player_type"].map({
        "WK": 15,
        "BAT": 5,
        "ALL": 5,
        "BOWL": 5
    }))
    
    match_df["runs_per_mat"] = match_df["runs_bat"] / match_df["mat_bat"].replace(0, 1)
    match_df["wkts_per_mat"] = match_df["wkts"] / match_df["mat_bowl"].replace(0, 1)
    match_df["dis_per_mat"] = match_df["dis"] / match_df["mat"].replace(0, 1)
    
    # Relaxed filter: only exclude very low-impact players
    match_df = match_df[
        (match_df["runs_per_mat"] >= 5) | (match_df["wkts_per_mat"] >= 0.2) | (match_df["dis_per_mat"] >= 0.1)
    ]
    
    # Debug: Print player counts by role and team
    print("Player counts by role:")
    print(match_df.groupby("player_type").size())
    print("\nPlayer counts by team:")
    print(match_df.groupby("team").size())
    
    match_df.to_csv(output_file, index=False)
    return match_df

if __name__ == "__main__":
    team1, team2 = "PBKS", "GT"
    filter_match_players("../data/squad.csv", team1, team2, "../data/match_players.csv")