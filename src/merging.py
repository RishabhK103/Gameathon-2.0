import pandas as pd
import numpy as np

# ---------------- Step 1: Load the Data ----------------

# Load unified dataset and squad mapping
unified_df = pd.read_csv("unified_dataset.csv")
squad_df = pd.read_csv("../squad.csv")

# ---------------- Step 2: Standardize Column Names ----------------

# Standardize column names for consistency: lower-case and strip spaces
unified_df.columns = unified_df.columns.str.strip().str.lower()
squad_df.columns = squad_df.columns.str.strip().str.lower()

# Expected columns in squad.csv:
#   espn_player_name, player, player_type
# In the unified dataset, we assume the "player" column currently holds the ESPN player name.

# ---------------- Step 3: Merge Squad Mapping with Unified Dataset ----------------

# Merge the squad mapping to get the actual player names and their types.
# Left join on unified_df['player'] and squad_df['espn_player_name']
merged_df = pd.merge(unified_df,
                     squad_df[['espn player name', 'player name', 'player type']],
                     left_on='player name',
                     right_on='espn player name',
                     how='left')

# Update the "player" column: if an actual name is available from squad.csv, use it.
merged_df['player'] = merged_df.apply(
    lambda row: row['player_y'] if pd.notnull(row['player_y']) else row['player'],
    axis=1
)

# Update the "role" column using "player_type" from squad mapping, if available.
merged_df['role'] = merged_df.apply(
    lambda row: row['player_type'] if pd.notnull(row['player_type']) else row['role'],
    axis=1
)

# ---------------- Step 4: Cleanup and Save Updated Dataset ----------------

# Drop extra columns added from the merge (the original squad mapping columns)
merged_df.drop(columns=['espn_player_name', 'player_y', 'player_type'], inplace=True)

# Save the updated unified dataset to a new CSV file
merged_df.to_csv("unified_dataset_updated.csv", index=False)
print("Updated unified dataset saved as 'unified_dataset_updated.csv'")
