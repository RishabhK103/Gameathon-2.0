import sys
import os
import pandas as pd
import numpy as np
import yaml
from scipy.stats import percentileofscore
from colorama import Fore, init

init(autoreset=True)


class PlayerForm:
    def __init__(self):
        """
        Initializes the PlayerForm with file paths and parameters tailored for IPL data.
        """
        try:
            with open("../config.yaml", "r") as stream:
                config = yaml.safe_load(stream)
        except Exception as e:
            print(Fore.RED + f"Error reading YAML config file: {e}")
            sys.exit(1)

        self.config = config

        # File paths matching ipl20scrapper.py output
        self.bowling_file = "../data/ipl/bowling_averages.csv"
        self.batting_file = "../data/ipl/batting_averages.csv"
        self.fielding_file = "../data/ipl/fielding_averages.csv"
        self.output_file = "../data/ipl/player_form_scores.csv"
        self.squad_file = config["data"].get("squad_file", "../data/ipl/squad.csv")
        self.previous_months = config["data"].get("previous_months", 24)
        self.decay_rate = config["data"].get("decay_rate", 0.1)
        self.key_cols = ["Player", "Team", "Span", "Mat"]  # Adjusted to use Span instead of Start/End Date

    def load_data(self):
        """
        Loads and merges the bowling, batting, and fielding CSV data.
        Splits the 'Span' column into approximate 'Start Date' and 'End Date'.
        """
        try:
            bowling = pd.read_csv(self.bowling_file)
            batting = pd.read_csv(self.batting_file)
            fielding = pd.read_csv(self.fielding_file)
        except Exception as e:
            print(Fore.RED + f"Error reading CSV files: {e}")
            sys.exit(1)

        # Drop columns with all missing values
        bowling = bowling.dropna(axis=1, how="all")
        batting = batting.dropna(axis=1, how="all")
        fielding = fielding.dropna(axis=1, how="all")

        # Process 'Span' column into approximate 'Start Date' and 'End Date'
        for df in [bowling, batting, fielding]:
            if "Span" in df.columns:
                # Split Span into Start and End years (e.g., "2023-2024")
                df[["Start Date", "End Date"]] = df["Span"].str.split("-", expand=True)
                # Assume January 1st for Start Date and December 31st for End Date
                df["Start Date"] = pd.to_datetime(df["Start Date"] + "-01-01", format="%Y-%m-%d")
                df["End Date"] = pd.to_datetime(df["End Date"] + "-12-31", format="%Y-%m-%d")

        # Rename columns for each dataset (except for key columns)
        bowling_renamed = bowling.rename(
            columns=lambda x: f"bowl {x}".lower() if x not in self.key_cols + ["Start Date", "End Date"] else x
        )
        batting_renamed = batting.rename(
            columns=lambda x: f"bat {x}".lower() if x not in self.key_cols + ["Start Date", "End Date"] else x
        )
        fielding_renamed = fielding.rename(
            columns=lambda x: f"field {x}".lower() if x not in self.key_cols + ["Start Date", "End Date"] else x
        )

        # Merge DataFrames on key columns using outer joins
        df = bowling_renamed.merge(batting_renamed, on=self.key_cols + ["Start Date", "End Date"], how="outer")
        df = df.merge(fielding_renamed, on=self.key_cols + ["Start Date", "End Date"], how="outer")

        # Save updated files with Start Date and End Date
        try:
            batting.to_csv(self.batting_file, index=False)
            bowling.to_csv(self.bowling_file, index=False)
            fielding.to_csv(self.fielding_file, index=False)
            print("Updated player files with derived Start Date and End Date")
        except Exception as e:
            print(Fore.RED + f"Error saving updated CSV files: {e}")
            sys.exit(1)

        return df

    def filter_players_by_squad(self, df):
        """
        Filters the DataFrame to retain only rows for players present in the squad CSV file.
        Updates the 'Team' column with the latest team from squad.csv.
        """
        try:
            squad_df = pd.read_csv(self.squad_file)
        except Exception as e:
            print(Fore.RED + f"Error reading squad CSV file: {e}")
            sys.exit(1)

        valid_players = squad_df["ESPN player name"].dropna().tolist()
        player_to_team = squad_df.set_index("ESPN player name")["Team"].to_dict()
        player_abbrev_to_full = squad_df.set_index("ESPN player name")["Player Name"].to_dict()

        filtered_df = df[df["Player"].isin(valid_players)].copy()

        squad_players_set = set(valid_players)
        df_players_set = set(filtered_df["Player"])
        missing_players = squad_players_set - df_players_set

        if missing_players:
            print(Fore.YELLOW + "Missing players from data:")
            for player in sorted(missing_players):
                team = player_to_team.get(player, "Unknown Team")
                print(f"- {player} ({team})")
        else:
            print(Fore.GREEN + "All players from the squad CSV file are present in the DataFrame.")

        print(f"\nExtracted players: {len(df_players_set)} / {len(squad_players_set)}")
        print(f"Missing players: {len(missing_players)}\n")

        # Merge squad data and overwrite Team with squad.csv's Team
        filtered_df = filtered_df.merge(
            squad_df[["Credits", "Player Type", "Player Name", "Team", "ESPN player name"]],
            left_on="Player",
            right_on="ESPN player name",
            how="left",
            suffixes=("_scraped", "_squad")  # Differentiate original and squad Team columns
        )

        # Replace the original Team column with the squad Team
        filtered_df["Team"] = filtered_df["Team_squad"]
        filtered_df["Player"] = filtered_df["Player Name"]

        # Drop unnecessary columns
        filtered_df.drop(["ESPN player name", "Player Name", "Team_squad", "Team_scraped"], axis=1, inplace=True)

        return filtered_df

    def calculate_form(self, player_df):
        """
        Calculates recent form scores for batting, bowling, and fielding using T20 weights.
        Uses derived 'End Date' for time-based filtering.
        """
        player_df["End Date"] = pd.to_datetime(player_df["End Date"])
        cutoff_date = pd.to_datetime("today") - pd.DateOffset(months=self.previous_months)
        recent_data = player_df[player_df["End Date"] >= cutoff_date].copy()
        recent_data.sort_values(by=["Player", "End Date"], ascending=[True, False], inplace=True)
        recent_data["match_index"] = recent_data.groupby("Player").cumcount()
        recent_data["weight"] = np.exp(-self.decay_rate * recent_data["match_index"])

        def compute_ewma(g, col):
            return np.average(g[col].fillna(0), weights=g["weight"])

        def normalize_series(series):
            return series.apply(lambda x: percentileofscore(series.dropna(), x))

        format_weights = {
            "T20": {
                "batting": {
                    "bat runs": 0.35,  # Boosted for run volume (1 pt/run + milestones)
                    "bat ave": 0.05,   # Lowered; consistency matters less in T20
                    "bat sr": 0.35,    # High weight for SR bonuses/penalties
                    "bat 4s": 0.10,   # Kept moderate (2 pt bonus)
                    "bat 6s": 0.15,   # Increased for higher bonus (4 pts)
                },
                "bowling": {
                    "bowl wkts": 0.55,  # Increased; wickets are king (25 pts each)
                    "bowl ave": 0.15,   # Reduced; indirect impact
                    "bowl econ": 0.30,  # Kept strong for econ bonuses/penalties
                },
            },
        }
        format_type = "T20"  # IPL data is T20
        batting_weights = format_weights[format_type]["batting"]
        bowling_weights = format_weights[format_type]["bowling"]

        # Batting Form
        batting_metrics = {}
        for metric in ["bat runs", "bat bf", "bat sr", "bat ave", "bat 4s", "bat 6s"]:
            batting_metrics[metric] = recent_data.groupby("Player", group_keys=False).apply(
                lambda g: compute_ewma(g, metric), include_groups=False
            )
        batting_df = pd.DataFrame(batting_metrics).reset_index()

        batting_norm = {}
        for col in ["bat runs", "bat ave", "bat sr", "bat 4s", "bat 6s"]:
            batting_norm[col] = normalize_series(batting_df[col])

        batting_df["Batting Form"] = (
            batting_weights["bat runs"] * batting_norm["bat runs"]
            + batting_weights["bat ave"] * batting_norm["bat ave"]
            + batting_weights["bat sr"] * batting_norm["bat sr"]
            + batting_weights["bat 4s"] * batting_norm["bat 4s"]
            + batting_weights["bat 6s"] * batting_norm["bat 6s"]
        )

        # Bowling Form
        bowling_metrics = {}
        for metric in ["bowl wkts", "bowl runs", "bowl econ", "bowl overs", "bowl ave"]:
            bowling_metrics[metric] = recent_data.groupby("Player", group_keys=False).apply(
                lambda g: compute_ewma(g, metric), include_groups=False
            )
        bowling_df = pd.DataFrame(bowling_metrics).reset_index()

        bowling_norm = {}
        bowling_norm["bowl wkts"] = normalize_series(bowling_df["bowl wkts"])
        bowling_norm["bowl ave"] = 100 - normalize_series(bowling_df["bowl ave"])
        bowling_norm["bowl econ"] = 100 - normalize_series(bowling_df["bowl econ"])

        bowling_df["Bowling Form"] = (
            bowling_weights["bowl wkts"] * bowling_norm["bowl wkts"]
            + bowling_weights["bowl ave"] * bowling_norm["bowl ave"]
            + bowling_weights["bowl econ"] * bowling_norm["bowl econ"]
        )

        # Fielding Form
        fielding_metrics = {}
        for metric in ["field ct", "field st", "field ct wk"]:
            fielding_metrics[metric] = recent_data.groupby("Player", group_keys=False).apply(
                lambda g: compute_ewma(g, metric), include_groups=False
            )
        fielding_df = pd.DataFrame(fielding_metrics).reset_index()

        fielding_norm = {}
        for col in ["field ct", "field st", "field ct wk"]:
            fielding_norm[col] = normalize_series(fielding_df[col])

        fielding_df["Fielding Form"] = (
            0.5 * fielding_norm["field ct"]
            + 0.3 * fielding_norm["field st"]
            + 0.2 * fielding_norm["field ct wk"]
        )

        form_df = (
            batting_df[["Player", "Batting Form"]]
            .merge(bowling_df[["Player", "Bowling Form"]], on="Player", how="outer")
            .merge(fielding_df[["Player", "Fielding Form"]], on="Player", how="outer")
        )
        metadata_df = player_df[["Player", "Credits", "Player Type", "Team"]].drop_duplicates("Player")
        form_df = form_df.merge(metadata_df, on="Player", how="left")

        # Optional: Print data coverage
        player_months = (
            recent_data.groupby(["Player", "Team"])["End Date"]
            .agg(lambda x: ((x.max() - x.min()).days // 30, x.max(), x.min()))
            .reset_index()
        )
        player_months.rename(columns={"End Date": "Months of Data"}, inplace=True)
        player_months[["Months of Data", "Latest Date", "Oldest Date"]] = pd.DataFrame(
            player_months["Months of Data"].tolist(), index=player_months.index
        )
        player_months = player_months.sort_values(by="Months of Data", ascending=True)
        for _, row in player_months.iterrows():
            if row["Months of Data"] < 3:
                print(
                    f"{Fore.YELLOW}{row['Months of Data'] + 1}\t"
                    f"{row['Oldest Date'].strftime('%b %y')} - "
                    f"{row['Latest Date'].strftime('%b %y')} \t"
                    f"{row['Player']} ({row['Team']})"
                )
            else:
                print(
                    f"{row['Months of Data']}\t"
                    f"{row['Oldest Date'].strftime('%b %y')} - "
                    f"{row['Latest Date'].strftime('%b %y')} \t"
                    f"{row['Player']} ({row['Team']})"
                )

        return form_df

    def run(self):
        """
        Executes the full data preprocessing workflow for IPL data.
        """
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))

        print(Fore.CYAN + "Starting IPL data preprocessing...")
        df = self.load_data()
        filtered_df = self.filter_players_by_squad(df)
        form_scores = self.calculate_form(filtered_df)
        print(Fore.GREEN + "\n\nIPL form scores calculated successfully")
        form_scores.to_csv(self.output_file, index=False)


def UpdatePlayerForm():
    try:
        preprocessor = PlayerForm()
        preprocessor.run()
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        UpdatePlayerForm()
    except KeyboardInterrupt:
        sys.exit(1)