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
final_df = merged[['Player', 'Batting Form', 'Bowling Form', 'Fielding Form', 'Credits', 'Player Type', 'Team']]

# Save the new CSV
final_df.to_csv("data/merged_output.csv", index=False)
