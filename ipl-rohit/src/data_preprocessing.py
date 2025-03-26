import pandas as pd
from fuzzywuzzy import fuzz

def normalize_columns(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    numeric_cols = ["mat", "inns", "no", "runs", "bf", "100", "50", "0", "4s", "6s", "hs",
                    "overs", "mdns", "wkts", "4", "5", "dis", "ct", "st", "ct_wk", "ct_fi", "md"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def match_player_names(squad_names, historical_names):
    name_mapping = {}
    for squad_name in squad_names:
        best_match = None
        best_score = 0
        for hist_name in historical_names:
            score = fuzz.token_sort_ratio(squad_name, hist_name)
            if score > best_score and score > 70:  # Lowered threshold from 80 to 70
                best_score = score
                best_match = hist_name
        if best_match:
            name_mapping[squad_name] = best_match
    return name_mapping

def parse_bbi(bbi):
    if pd.isna(bbi) or bbi == "":
        return 0
    try:
        wickets, _ = bbi.split("/")
        return int(wickets)
    except (ValueError, AttributeError):
        return 0

def aggregate_batting(df):
    df = normalize_columns(df)
    df.fillna({"mat": 0, "inns": 0, "no": 0, "runs": 0, "bf": 0, "100": 0, "50": 0, "0": 0, "4s": 0, "6s": 0, "hs": 0}, inplace=True)
    batting_agg = df.groupby("player").agg({
        "mat": "sum", "inns": "sum", "no": "sum", "runs": "sum", "bf": "sum",
        "100": "sum", "50": "sum", "0": "sum", "4s": "sum", "6s": "sum", "hs": "max"
    }).reset_index()
    batting_agg["ave"] = batting_agg.apply(
        lambda row: row["runs"] / (row["inns"] - row["no"]) if row["inns"] > row["no"] else 0, axis=1
    )
    batting_agg["sr"] = batting_agg.apply(
        lambda row: (row["runs"] / row["bf"]) * 100 if row["bf"] > 0 else 0, axis=1
    )
    return batting_agg

def aggregate_bowling(df):
    df = normalize_columns(df)
    df.fillna({"mat": 0, "inns": 0, "overs": 0, "mdns": 0, "runs": 0, "wkts": 0, "4": 0, "5": 0, "bbi": "0/0"}, inplace=True)
    df["bbi_wickets"] = df["bbi"].apply(parse_bbi)
    bowling_agg = df.groupby("player").agg({
        "mat": "sum", "inns": "sum", "overs": "sum", "mdns": "sum", "runs": "sum",
        "wkts": "sum", "4": "sum", "5": "sum", "bbi_wickets": "max", "bbi": "first"
    }).reset_index()
    bowling_agg.rename(columns={"bbi_wickets": "best_wickets"}, inplace=True)
    bowling_agg["ave"] = bowling_agg.apply(
        lambda row: row["runs"] / row["wkts"] if row["wkts"] > 0 else float("inf"), axis=1
    )
    bowling_agg["econ"] = bowling_agg.apply(
        lambda row: (row["runs"] / row["overs"]) * 6 if row["overs"] > 0 else 0, axis=1
    )
    bowling_agg["sr"] = bowling_agg.apply(
        lambda row: (row["overs"] / row["wkts"]) * 6 if row["wkts"] > 0 else float("inf"), axis=1
    )
    return bowling_agg

def aggregate_fielding(df):
    df = normalize_columns(df)
    df.fillna({"mat": 0, "inns": 0, "dis": 0, "ct": 0, "st": 0, "ct_wk": 0, "ct_fi": 0, "md": 0}, inplace=True)
    fielding_agg = df.groupby("player").agg({
        "mat": "sum", "inns": "sum", "dis": "sum", "ct": "sum", "st": "sum",
        "ct_wk": "sum", "ct_fi": "sum", "md": "max"
    }).reset_index()
    fielding_agg["d/i"] = fielding_agg.apply(
        lambda row: row["dis"] / row["inns"] if row["inns"] > 0 else 0, axis=1
    )
    return fielding_agg

def preprocess_data(batting_file, bowling_file, fielding_file, squad_file, output_file):
    batting = pd.read_csv(batting_file)
    bowling = pd.read_csv(bowling_file)
    fielding = pd.read_csv(fielding_file)
    squad = pd.read_csv(squad_file)

    squad = normalize_columns(squad)

    batting_agg = aggregate_batting(batting)
    bowling_agg = aggregate_bowling(bowling)
    fielding_agg = aggregate_fielding(fielding)

    merged_df = batting_agg.merge(bowling_agg, on="player", how="outer", suffixes=("_bat", "_bowl")) \
                          .merge(fielding_agg, on="player", how="outer")

    squad_names = squad["player_name"].unique()
    historical_names = merged_df["player"].unique()
    name_mapping = match_player_names(squad_names, historical_names)

    squad["historical_name"] = squad["player_name"].map(name_mapping)
    
    full_df = squad.merge(merged_df, left_on="historical_name", right_on="player", how="left")
    full_df.drop(columns=["player", "historical_name"], inplace=True)

    full_df.fillna({
        "mat_bat": 0, "inns_bat": 0, "no": 0, "runs_bat": 0, "bf": 0,
        "100": 0, "50": 0, "0": 0, "4s": 0, "6s": 0, "hs": 0,
        "mat_bowl": 0, "inns_bowl": 0, "overs": 0, "mdns": 0, "runs_bowl": 0,
        "wkts": 0, "4": 0, "5": 0, "best_wickets": 0,
        "mat": 0, "inns": 0, "dis": 0, "ct": 0, "st": 0, "ct_wk": 0, "ct_fi": 0, "md": 0
    }, inplace=True)
    
    full_df["ave_bat"] = full_df.apply(
        lambda row: row["runs_bat"] / (row["inns_bat"] - row["no"]) if row["inns_bat"] > row["no"] else 0, axis=1
    )
    full_df["sr"] = full_df.apply(
        lambda row: (row["runs_bat"] / row["bf"]) * 100 if row["bf"] > 0 else 0, axis=1
    )
    full_df["ave_bowl"] = full_df.apply(
        lambda row: row["runs_bowl"] / row["wkts"] if row["wkts"] > 0 else float("inf"), axis=1
    )
    full_df["econ"] = full_df.apply(
        lambda row: (row["runs_bowl"] / row["overs"]) * 6 if row["overs"] > 0 else 0, axis=1
    )
    full_df["sr_bowl"] = full_df.apply(
        lambda row: (row["overs"] / row["wkts"]) * 6 if row["wkts"] > 0 else float("inf"), axis=1
    )
    full_df["d/i"] = full_df.apply(
        lambda row: row["dis"] / row["inns"] if row["inns"] > 0 else 0, axis=1
    )
    full_df["bbi"] = full_df["bbi"].fillna("0/0")

    full_df.to_csv(output_file, index=False)
    return full_df

if __name__ == "__main__":
    preprocess_data("../data/batting_averages.csv", "../data/bowling_averages.csv", "../data/fielding_averages.csv", "../data/squad.csv", "../data/merged_data.csv")