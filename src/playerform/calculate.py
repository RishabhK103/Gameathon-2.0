import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import os
import sys
from datetime import datetime # Ensure datetime is imported if not already implicitly via pandas

class PlayerForm:
    def __init__(self):
        # Removed fielding_file
        self.bowling_file = "data/recent_averages/bowling_data.csv" # Updated filename if needed
        self.batting_file = "data/recent_averages/batting_data.csv" # Updated filename if needed

        self.output_file = "data/recent_averages/player_form_scores_final.csv"
        self.squad_file = "data/squad.csv"

        self.previous_months = 36
        self.decay_rate = 0.1
        self.key_cols = ["Player", "Team", "Span", "Mat"]

    def load_data(self):
        try:
            # Load only batting and bowling
            bowling = pd.read_csv(self.bowling_file)
            batting = pd.read_csv(self.batting_file)
            print(f"Loaded bowling data: {bowling.shape}")
            print(f"Loaded batting data: {batting.shape}")
        except FileNotFoundError as e:
             print(f"Error: Input file not found - {e}. Please ensure the scraping script ran successfully.")
             sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV files: {e}")
            sys.exit(1)

        # Drop columns that are entirely NaN
        bowling = bowling.dropna(axis=1, how='all')
        batting = batting.dropna(axis=1, how='all')

        # Process Span column if it exists
        for df in [bowling, batting]: # Removed fielding
            if "Span" in df.columns:
                try:
                    # Handle potential errors during split or conversion
                    span_split = df["Span"].astype(str).str.split("-", expand=True)
                    # Assume format YYYY-YYYY, create start/end dates
                    df["Start Date"] = pd.to_datetime(span_split[0] + '-01-01', format='%Y-%m-%d', errors='coerce')
                    df["End Date"] = pd.to_datetime(span_split[1] + '-12-31', format='%Y-%m-%d', errors='coerce')
                    # Drop rows where date conversion failed
                    df.dropna(subset=['Start Date', 'End Date'], inplace=True)
                except Exception as e:
                    print(f"Warning: Could not process 'Span' column correctly. Error: {e}")
                    # Decide how to handle: maybe drop Span or skip date creation
                    if "Start Date" not in df.columns: df["Start Date"] = pd.NaT
                    if "End Date" not in df.columns: df["End Date"] = pd.NaT


        # Rename columns, prefixing with 'bowl ' or 'bat '
        bowling_renamed = bowling.rename(
            columns=lambda x: (
                f"bowl {x}".lower()
                if x not in self.key_cols + ["Start Date", "End Date", "Span"] # Keep Span if needed elsewhere
                else x
            )
        )
        batting_renamed = batting.rename(
            columns=lambda x: (
                f"bat {x}".lower()
                if x not in self.key_cols + ["Start Date", "End Date", "Span"]
                else x
            )
        )
        # Removed fielding_renamed

        # Merge bowling and batting data
        # Use specific merge keys to avoid issues if columns differ slightly
        merge_keys = list(set(self.key_cols + ["Start Date", "End Date", "Span"]) & set(bowling_renamed.columns) & set(batting_renamed.columns))
        print(f"Merging bowling and batting data on keys: {merge_keys}")

        df = bowling_renamed.merge(
            batting_renamed, on=merge_keys, how="outer"
        )
        # Removed merge with fielding_renamed

        print(f"Shape after merging bowling and batting: {df.shape}")
        return df

    def include_all_squad_players(self, df):
        try:
            squad_df = pd.read_csv(self.squad_file)
            squad_df["ESPN player name"] = squad_df["ESPN player name"].str.strip()
            print(f"Loaded squad data: {squad_df.shape}")
        except FileNotFoundError as e:
             print(f"Error: Squad file not found - {e}. Please ensure '{self.squad_file}' exists.")
             sys.exit(1)
        except Exception as e:
            print(f"Error reading squad CSV file: {e}")
            sys.exit(1)

        # Ensure required columns exist in squad_df
        required_squad_cols = ["Credits", "Player Type", "Player Name", "Team", "ESPN player name"]
        if not all(col in squad_df.columns for col in required_squad_cols):
            print(f"Error: Squad file '{self.squad_file}' is missing one or more required columns: {required_squad_cols}")
            sys.exit(1)

        valid_players = squad_df["ESPN player name"].dropna().unique().tolist()
        print(f"Total unique players in squad.csv: {len(valid_players)}")

        # Check if 'Player' column exists in the scraped data df
        if 'Player' not in df.columns:
            print("Error: 'Player' column not found in the merged scraped data. Cannot merge with squad.")
            # Attempt to find a similar column or exit
            print("Available columns in scraped data:", df.columns.tolist())
            sys.exit(1)


        # Merge squad data with scraped data, keeping all squad players (left join from squad perspective)
        combined_df = squad_df[required_squad_cols].merge(
            df,
            left_on="ESPN player name",
            right_on="Player", # Assumes 'Player' is the name column in scraped data
            how="left",
            suffixes=("_squad", "_scraped"), # Suffixes help identify origin if columns clash
        )
        print(f"Shape after merging with squad data: {combined_df.shape}")


        # --- Data Harmonization after Merge ---
        # Use Player Name from squad file as the definitive 'Player'
        combined_df['Player'] = combined_df['Player Name']
        # Use Team from squad file as the definitive 'Team'
        combined_df['Team'] = combined_df['Team_squad']

        # Identify and handle potential duplicate columns from the merge if necessary
        # For example, if 'Mat_squad' and 'Mat_scraped' exist, decide which one to keep or how to combine.
        # For now, we primarily rely on the scraped data for stats.

        # Drop redundant columns after harmonization
        cols_to_drop = ['ESPN player name', 'Player Name', 'Team_squad']
        # Add potentially conflicting columns from scraped data if they exist
        if 'Team_scraped' in combined_df.columns: cols_to_drop.append('Team_scraped')
        if 'Player_scraped' in combined_df.columns: cols_to_drop.append('Player_scraped') # If 'Player' was the merge key

        # Check which columns actually exist before trying to drop
        existing_cols_to_drop = [col for col in cols_to_drop if col in combined_df.columns]
        combined_df.drop(columns=existing_cols_to_drop, inplace=True)

        print(f"Shape after dropping redundant columns: {combined_df.shape}")


        # --- Reporting Coverage ---
        data_players = df["Player"].dropna().unique().tolist()
        squad_players_in_final = combined_df["Player"].dropna().unique().tolist() # Use the final 'Player' column

        # Players in scraped data but not listed in squad.csv (using ESPN name for comparison)
        missing_in_squad = set(data_players) - set(valid_players)

        # Players listed in squad.csv but having no matching scraped data (check if key stats are NaN)
        # A simple check: count players from squad where 'Mat' (or another key stat) is NaN after the merge
        missing_in_data_count = combined_df[combined_df['Mat'].isna() & combined_df['Player'].isin(valid_players)].shape[0]
        # Get the names of those missing data
        missing_in_data_names = combined_df[combined_df['Mat'].isna() & combined_df['Player'].isin(valid_players)]['Player'].tolist()


        print(f"Players in scraped data but potentially missing/mismatched in squad.csv: {len(missing_in_squad)}")
        # print(f"   Examples: {list(missing_in_squad)[:10]}") # Uncomment to see examples
        print(f"Players in squad.csv with no scraped data found: {missing_in_data_count}")
        # print(f"   Examples: {missing_in_data_names[:10]}") # Uncomment to see examples
        print(f"Total unique players in final dataset: {len(squad_players_in_final)}")

        return combined_df

    def calculate_form(self, player_df):
        # Ensure 'End Date' is present and in datetime format
        if 'End Date' not in player_df.columns:
            print("Error: 'End Date' column missing, cannot calculate time-based form.")
            # As a fallback, maybe use a default high form score or skip form calculation
            # For now, let's add placeholder columns and return
            player_df['Batting Form'] = np.nan
            player_df['Bowling Form'] = np.nan
            return player_df[['Player', 'Credits', 'Player Type', 'Team', 'Batting Form', 'Bowling Form']].drop_duplicates('Player')

        player_df["End Date"] = pd.to_datetime(player_df["End Date"], errors="coerce")

        # Define cutoff date for recent data
        # Use current date for calculation consistency
        current_date = pd.to_datetime(datetime.now().date())
        cutoff_date = current_date - pd.DateOffset(months=self.previous_months)
        print(f"Calculating form using data since: {cutoff_date.strftime('%Y-%m-%d')}")

        # Filter for recent data, handling potential NaT in End Date
        recent_data = player_df[player_df["End Date"].notna() & (player_df["End Date"] >= cutoff_date)].copy()
        print(f"Number of records within the last {self.previous_months} months: {recent_data.shape[0]}")

        if recent_data.empty:
            print("Warning: No recent data found within the specified period. Form scores will be NaN.")
            # Return base player info with NaN form scores
            form_df = player_df[['Player', 'Credits', 'Player Type', 'Team']].drop_duplicates('Player').copy()
            form_df['Batting Form'] = np.nan
            form_df['Bowling Form'] = np.nan
            return form_df

        # Calculate weights based on recency
        recent_data.sort_values(by=["Player", "End Date"], ascending=[True, False], inplace=True)
        recent_data["match_index"] = recent_data.groupby("Player").cumcount()
        recent_data["weight"] = np.exp(-self.decay_rate * recent_data["match_index"])

        # Helper function for weighted average
        def compute_ewma(g, col):
            # Check if column exists and has non-NA values before calculating
            if col not in g.columns or g[col].isna().all():
                return np.nan # Return NaN if column missing or all NaN
            return np.average(g[col].fillna(0), weights=g["weight"])

        # Helper function for percentile normalization
        def normalize_series(series):
            # Handle series with all NaNs or single values
            if series.isna().all() or series.nunique() <= 1:
                 return pd.Series(50, index=series.index) # Assign median percentile
            # Use dropna() within percentileofscore
            return series.apply(lambda x: percentileofscore(series.dropna(), x) if pd.notna(x) else np.nan)


        # Define weights for T20 format (adjust as needed)
        format_weights = {
            "T20": {
                "batting": {
                    "bat runs": 0.35, "bat ave": 0.05, "bat sr": 0.35,
                    "bat 4s": 0.10, "bat 6s": 0.15,
                },
                "bowling": {"bowl wkts": 0.55, "bowl ave": 0.15, "bowl econ": 0.30},
            },
        }
        batting_weights = format_weights["T20"]["batting"]
        bowling_weights = format_weights["T20"]["bowling"]

        # --- Batting Form ---
        print("Calculating Batting Form...")
        batting_cols = ["bat runs", "bat bf", "bat sr", "bat ave", "bat 4s", "bat 6s"]
        # Check which batting columns are actually present in recent_data
        available_batting_cols = [col for col in batting_cols if col in recent_data.columns]

        batting_metrics = {}
        if available_batting_cols:
             batting_metrics = {
                 metric: recent_data.groupby("Player", group_keys=False).apply(
                     lambda g: compute_ewma(g, metric), include_groups=False
                 )
                 for metric in available_batting_cols
             }
        batting_df = pd.DataFrame(batting_metrics).reset_index()

        # Normalize available metrics needed for form calculation
        batting_norm = {}
        norm_cols_bat = list(batting_weights.keys()) # Columns needed for weighted sum
        for col in norm_cols_bat:
            if col in batting_df.columns:
                batting_norm[col] = normalize_series(batting_df[col])
            else:
                # Handle missing columns - assign NaN or a default (e.g., 50th percentile)
                batting_norm[col] = pd.Series(np.nan, index=batting_df.index)
                print(f"Warning: Batting metric '{col}' not found for normalization.")

        # Calculate weighted batting form score
        batting_df["Batting Form"] = sum(
            batting_weights[col] * batting_norm[col].fillna(50) # Fill NaN with 50th percentile
            for col in batting_weights if col in batting_norm # Only use available normalized cols
        )


        # --- Bowling Form ---
        print("Calculating Bowling Form...")
        bowling_cols = ["bowl wkts", "bowl runs", "bowl econ", "bowl overs", "bowl ave"]
        available_bowling_cols = [col for col in bowling_cols if col in recent_data.columns]

        bowling_metrics = {}
        if available_bowling_cols:
            bowling_metrics = {
                metric: recent_data.groupby("Player", group_keys=False).apply(
                    lambda g: compute_ewma(g, metric), include_groups=False
                )
                for metric in available_bowling_cols
            }
        bowling_df = pd.DataFrame(bowling_metrics).reset_index()

        # Determine if player has bowled recently
        if 'bowl overs' in bowling_df.columns:
             bowling_df["Has Bowled"] = bowling_df["bowl overs"] > 0
        else:
             bowling_df["Has Bowled"] = False # Assume false if overs data is missing
             print("Warning: 'bowl overs' column missing, cannot accurately determine bowlers.")


        # Normalize available metrics needed for form calculation
        bowling_norm = {}
        norm_cols_bowl = list(bowling_weights.keys())
        for col in norm_cols_bowl:
             if col in bowling_df.columns:
                 # Handle inverse relationship for Ave and Econ (lower is better)
                 if col in ["bowl ave", "bowl econ"]:
                     # Replace 0 with infinity for percentile calculation (0 is best)
                     series_to_norm = bowling_df[col].replace(0, np.inf)
                     # Normalize and invert percentile (100 - percentile)
                     bowling_norm[col] = 100 - normalize_series(series_to_norm)
                 else: # Higher is better for Wkts
                     bowling_norm[col] = normalize_series(bowling_df[col])
             else:
                 bowling_norm[col] = pd.Series(np.nan, index=bowling_df.index)
                 print(f"Warning: Bowling metric '{col}' not found for normalization.")

        # Calculate weighted bowling form score
        # Apply only to players who have bowled, otherwise assign a default or NaN
        default_non_bowler_score = 30 # Assign a default score (e.g., 30th percentile)
        bowling_form_calculated = sum(
            bowling_weights[col] * bowling_norm[col].fillna(50) # Fill NaN with 50th percentile
            for col in bowling_weights if col in bowling_norm
        )

        bowling_df["Bowling Form"] = np.where(
            bowling_df["Has Bowled"],
            bowling_form_calculated,
            default_non_bowler_score # Assign default score to non-bowlers
        )


        # --- Fielding Form Removed ---


        # --- Merge forms ---
        print("Merging form scores...")
        # Start with the base player info from the input df (includes all squad players)
        # Ensure we only take unique players to avoid duplicate rows before merging form scores
        base_player_info = player_df[['Player', 'Credits', 'Player Type', 'Team']].drop_duplicates('Player')

        # Merge batting form (left join to keep all players)
        form_df = base_player_info.merge(
            batting_df[['Player', 'Batting Form']], on='Player', how='left'
        )

        # Merge bowling form (left join to keep all players)
        form_df = form_df.merge(
            bowling_df[['Player', 'Bowling Form']], on='Player', how='left'
        )

        # Removed merge for Fielding Form

        # Fill NaN form scores if necessary (e.g., for players with no recent data)
        form_df['Batting Form'].fillna(30, inplace=True) # Example: fill with 30th percentile
        form_df['Bowling Form'].fillna(30, inplace=True) # Example: fill with 30th percentile

        print(f"Final form scores shape: {form_df.shape}")

        # --- Optional: Print data coverage summary ---
        # Check if recent_data and relevant columns exist before calculating coverage
        if not recent_data.empty and 'End Date' in recent_data.columns:
            try:
                print("\n--- Data Coverage Summary (Players with Recent Data) ---")
                player_months = (
                    recent_data.groupby(["Player", "Team"])["End Date"]
                    .agg(
                         count='size', # Number of entries in recent data
                         latest_date='max',
                         oldest_date='min'
                     )
                    .reset_index()
                )
                # Calculate months span more accurately
                player_months['Months of Data'] = ((player_months['latest_date'] - player_months['oldest_date']).dt.days / 30.44).round().astype(int)

                player_months = player_months.sort_values(by="Months of Data", ascending=True)

                print("Months\tPeriod\t\tPlayer (Team)")
                print("------\t------\t\t-------------")
                for _, row in player_months.head(15).iterrows(): # Print top 15 with least data
                     period_str = f"{row['oldest_date'].strftime('%b %y')} - {row['latest_date'].strftime('%b %y')}"
                     print(
                         f"{row['Months of Data']:<6}\t"
                         f"{period_str:<15}\t"
                         f"{row['Player']} ({row['Team']})"
                     )
            except Exception as e:
                print(f"Could not generate data coverage summary: {e}")
        else:
            print("\nSkipping data coverage summary as no recent data was found.")


        return form_df

    def run(self):
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        if not os.path.exists(output_dir):
             try:
                 os.makedirs(output_dir)
                 print(f"Created output directory: {output_dir}")
             except Exception as e:
                 print(f"Error creating output directory '{output_dir}': {e}")
                 sys.exit(1)


        print("Starting Player Form calculation...")
        df = self.load_data()
        if df.empty:
            print("Loaded data is empty after merging batting/bowling. Exiting.")
            return

        combined_df = self.include_all_squad_players(df)
        if combined_df.empty:
            print("Data is empty after merging with squad file. Exiting.")
            return

        form_scores = self.calculate_form(combined_df)

        if form_scores.empty:
            print("No form scores generated. Exiting.")
            return

        print("\nIPL form scores calculated successfully.")
        try:
            form_scores.to_csv(self.output_file, index=False)
            print(f"Form scores saved to: {self.output_file}")
        except Exception as e:
            print(f"Error saving form scores to CSV '{self.output_file}': {e}")


if __name__=="__main__":
    preprocessor = PlayerForm()
    preprocessor.run()