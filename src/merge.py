import pandas as pd

# Load the CSV files
csv1 = pd.read_csv("data/player_form_last3.csv")
csv2 = pd.read_csv("data/form_2025.csv")

weight_prev=0.6
weight_recent=0.4
# Merge on 'Player' while keeping all columns
merged = csv1.merge(csv2, on=['Player', 'Player Type', 'Team', 'Credits'], suffixes=('_csv1', '_csv2'))

# Compute the weighted averages
merged['Batting Form'] = weight_prev * merged['Batting Form_csv1'] + weight_recent * merged['Batting Form_csv2']
merged['Bowling Form'] = weight_prev * merged['Bowling Form_csv1'] + weight_recent * merged['Bowling Form_csv2']
merged['Fielding Form'] = weight_prev * merged['Fielding Form_csv1'] + weight_recent * merged['Fielding Form_csv2']

# Select required columns
final_df = merged[['Player', 'Batting Form', 'Bowling Form', 'Fielding Form', 'Credits', 'Player Type', 'Team', 'Bowler Type']]

# Save the new CSVimport pandas as pd
import yaml

# Load configuration from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the CSV files for previous and recent form data
csv1 = pd.read_csv("data/player_form_last3.csv")
csv2 = pd.read_csv("data/form_2025.csv")

weight_prev = 0.6
weight_recent = 0.4

# Merge on common columns; ensure "Bowler Type" is included
merged = csv1.merge(csv2, on=['Player', 'Player Type', 'Team', 'Credits'], suffixes=('_csv1', '_csv2'))

# Compute the weighted averages for each form metric
merged['Batting Form'] = weight_prev * merged['Batting Form_csv1'] + weight_recent * merged['Batting Form_csv2']
merged['Bowling Form'] = weight_prev * merged['Bowling Form_csv1'] + weight_recent * merged['Bowling Form_csv2']
merged['Fielding Form'] = weight_prev * merged['Fielding Form_csv1'] + weight_recent * merged['Fielding Form_csv2']

# Select required columns – note that we now include "Bowler Type"
final_df = merged[['Player', 'Batting Form', 'Bowling Form', 'Fielding Form', 'Credits', 'Player Type', 'Team', 'Bowler Type']]

# --- Update player names using the squad file from config ---
squad = pd.read_csv(config["data"]["squad_file"])
# Assumes squad.csv has columns: "ESPN player name", "Player Name", and "Bowler Type" (if present)
final_df = final_df.merge(
    squad[["ESPN player name", "Player Name"]],
    left_on="Player",
    right_on="ESPN player name",
    how="left"
)
# Replace Player names if available
final_df["Player"] = final_df["Player Name"].combine_first(final_df["Player"])
final_df.drop(["ESPN player name", "Player Name"], axis=1, inplace=True)

# Save the merged output to the file defined in config (e.g. data/recent_player_form.csv)
final_df.to_csv(config["data"]["player_form"], index=False)
print(f"Merged output saved to {config['data']['player_form']}")

final_df.to_csv("data/merged_output.csv", index=False)
