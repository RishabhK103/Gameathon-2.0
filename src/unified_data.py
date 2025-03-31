import pandas as pd
import numpy as np

def standardize_columns(df, col_mapping=None):
    """
    Standardize DataFrame columns by stripping spaces, converting to lowercase,
    and optionally renaming columns using a provided mapping.
    """
    df.columns = df.columns.str.strip().str.lower()
    if col_mapping:
        df.rename(columns=col_mapping, inplace=True)
    return df

# ----------------- Step 1: Load & Standardize Match-by-Match Data (2025 IPL) -----------------

# --- Batting Data ---
# Header: Team,Player,Mat,Inns,NO,Runs,HS,Ave,BF,SR,100,50,0,4s,6s,Match_ID,Date
bat_df = pd.read_csv("../data/ipl/march_2025/batting_all_matches.csv")
bat_df = standardize_columns(bat_df)
# Rename key columns and some stats to more descriptive names
bat_df.rename(columns={
    "bf": "balls_faced",
    "sr": "strike_rate",
    "100": "centuries",
    "50": "fifties",
    "0": "ducks",
    "match_id": "match_id"  # this converts 'Match_ID' to 'match_id' due to lower-casing
}, inplace=True)
# Extract season from date (assuming date format is YYYY-MM-DD)
bat_df['season'] = pd.to_datetime(bat_df['date']).dt.year

# --- Bowling Data ---
# Adjust these column names as per your bowling CSV's actual structure.
bowl_df = pd.read_csv("../data/ipl/march_2025/bowling_all_matches.csv")
bowl_df = standardize_columns(bowl_df)
bowl_df.rename(columns={
    "wkts": "wickets",
    "econ": "economy_rate"
}, inplace=True)
if 'date' in bowl_df.columns:
    bowl_df['season'] = pd.to_datetime(bowl_df['date']).dt.year
else:
    bowl_df['season'] = 2025  # default if date is missing

# --- Fielding Data ---
# Expected columns might include: Team,Player,Mat,Inns,Catches,Stumpings,RunOuts,Match_ID,Date
field_df = pd.read_csv("../data/ipl/march_2025/fielding_all_matches.csv")
field_df = standardize_columns(field_df)
field_df.rename(columns={
    "runouts": "run_outs"
}, inplace=True)
if 'date' in field_df.columns:
    field_df['season'] = pd.to_datetime(field_df['date']).dt.year
else:
    field_df['season'] = 2025

# ----------------- Step 2: Merge Match-by-Match Datasets (2025 IPL) -----------------
# Merge batting and bowling on common keys: match_id, player, team, and season.
match_merge = pd.merge(bat_df, bowl_df, on=["match_id", "player", "team", "season"], how="outer", suffixes=("_bat", "_bowl"))
# Merge the result with fielding data.
match_merge = pd.merge(match_merge, field_df, on=["match_id", "player", "team", "season"], how="outer", suffixes=("", "_field"))
# match_merge now represents the unified match-level dataset for 2025 IPL.

# ----------------- Step 3: Load & Process Ball-by-Ball Data (2022-2024) -----------------
ball_df = pd.read_csv("dream11_dataset_with_rolling.csv")
ball_df = standardize_columns(ball_df)
# Assuming ball_df already contains rich features (rolling averages, etc.)
# Optionally, compute batting strike rate if not present:
if 'balls_faced' in ball_df.columns and 'runs' in ball_df.columns:
    ball_df['strike_rate'] = ball_df.apply(lambda r: (r['runs'] / r['balls_faced'] * 100) if r['balls_faced'] > 0 else 0, axis=1)

# ----------------- Step 4: Unify Columns Across Datasets -----------------
# We want both historical ball-by-ball and 2025 match-level data to have the same columns.
# Compute the union of all columns.
final_cols = list(set(ball_df.columns).union(set(match_merge.columns)))
final_cols.sort()  # Optional: sort for consistency

# Ensure both datasets have all final columns by adding missing ones as NaN.
for col in final_cols:
    if col not in ball_df.columns:
        ball_df[col] = np.nan
    if col not in match_merge.columns:
        match_merge[col] = np.nan

# ----------------- Step 5: Concatenate the Datasets -----------------
# Vertically stack the historical data with the 2025 match-level data.
final_df = pd.concat([ball_df[final_cols], match_merge[final_cols]], ignore_index=True)
final_df.sort_values(by=["season", "match_id"], inplace=True)

# Save the final unified dataset.
final_df.to_csv("unified_dataset.csv", index=False)
print("Unified dataset saved to 'unified_dataset.csv'")
