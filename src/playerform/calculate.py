import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import os
import sys


class PlayerForm:
    def __init__(self):
        self.bowling_file = "data/recent_averages/bowling.csv"
        self.batting_file = "data/recent_averages/batting.csv"
        self.fielding_file = "data/recent_averages/fielding.csv"

        self.output_file = "data/recent_averages/player_form_scores.csv"
        self.squad_file = "data/squad.csv"

        self.previous_months = 36
        self.decay_rate = 0.1
        self.key_cols = ["Player", "Team", "Span", "Mat"]

    def load_data(self):
        try:
            bowling = pd.read_csv(self.bowling_file)
            batting = pd.read_csv(self.batting_file)
            fielding = pd.read_csv(self.fielding_file)
        except Exception as e:
            print(f"Error reading CSV files: {e}")
            sys.exit(1)

        bowling = bowling.dropna(axis=1, how="all")
        batting = batting.dropna(axis=1, how="all")
        fielding = fielding.dropna(axis=1, how="all")

        for df in [bowling, batting, fielding]:
            if "Span" in df.columns:
                df[["Start Date", "End Date"]] = df["Span"].str.split("-", expand=True)
                df["Start Date"] = pd.to_datetime(
                    df["Start Date"] + "-01-01", format="%Y-%m-%d"
                )
                df["End Date"] = pd.to_datetime(
                    df["End Date"] + "-12-31", format="%Y-%m-%d"
                )

        bowling_renamed = bowling.rename(
            columns=lambda x: (
                f"bowl {x}".lower()
                if x not in self.key_cols + ["Start Date", "End Date"]
                else x
            )
        )
        batting_renamed = batting.rename(
            columns=lambda x: (
                f"bat {x}".lower()
                if x not in self.key_cols + ["Start Date", "End Date"]
                else x
            )
        )
        fielding_renamed = fielding.rename(
            columns=lambda x: (
                f"field {x}".lower()
                if x not in self.key_cols + ["Start Date", "End Date"]
                else x
            )
        )

        df = bowling_renamed.merge(
            batting_renamed, on=self.key_cols + ["Start Date", "End Date"], how="outer"
        )
        df = df.merge(
            fielding_renamed, on=self.key_cols + ["Start Date", "End Date"], how="outer"
        )
        return df

    def include_all_squad_players(self, df):
        try:
            squad_df = pd.read_csv(self.squad_file)
            squad_df["ESPN player name"] = squad_df["ESPN player name"].str.strip()
        except Exception as e:
            print(f"Error reading squad CSV file: {e}")
            sys.exit(1)

        valid_players = squad_df["ESPN player name"].dropna().tolist()
        print(f"Total players in squad.csv: {len(valid_players)}")

        # Merge squad data with scraped data, keeping all squad players (right join)
        combined_df = squad_df[
            ["Credits", "Player Type", "Player Name", "Team", "ESPN player name"]
        ].merge(
            df,
            left_on="ESPN player name",
            right_on="Player",
            how="left",
            suffixes=("_squad", "_scraped"),
        )

        # Set Player and Team from squad.csv
        combined_df["Player"] = combined_df["Player Name"]
        combined_df["Team"] = combined_df["Team_squad"]

        # Drop redundant columns
        combined_df.drop(
            ["ESPN player name", "Player Name", "Team_squad", "Team_scraped"],
            axis=1,
            inplace=True,
        )

        # Report coverage
        data_players = df["Player"].unique().tolist()
        missing_in_squad = set(data_players) - set(valid_players)
        missing_in_data = set(valid_players) - set(
            combined_df["Player"].dropna().unique()
        )
        print(f"Players in scraped data but missing in squad.csv: {missing_in_squad}")
        print(f"Players in squad.csv with no scraped data: {missing_in_data}")
        print(f"Players in final dataset: {len(combined_df['Player'].unique())}")

        return combined_df

    def calculate_form(self, player_df):
        player_df["End Date"] = pd.to_datetime(player_df["End Date"], errors="coerce")
        cutoff_date = pd.to_datetime("today") - pd.DateOffset(
            months=self.previous_months
        )
        recent_data = player_df[player_df["End Date"] >= cutoff_date].copy()
        recent_data.sort_values(
            by=["Player", "End Date"], ascending=[True, False], inplace=True
        )
        recent_data["match_index"] = recent_data.groupby("Player").cumcount()
        recent_data["weight"] = np.exp(-self.decay_rate * recent_data["match_index"])

        def compute_ewma(g, col):
            return np.average(g[col].fillna(0), weights=g["weight"])

        def normalize_series(series):
            return series.apply(lambda x: percentileofscore(series.dropna(), x))

        format_weights = {
            "T20": {
                "batting": {
                    "bat runs": 0.35,
                    "bat ave": 0.05,
                    "bat sr": 0.35,
                    "bat 4s": 0.10,
                    "bat 6s": 0.15,
                },
                "bowling": {"bowl wkts": 0.55, "bowl ave": 0.15, "bowl econ": 0.30},
            },
        }
        batting_weights = format_weights["T20"]["batting"]
        bowling_weights = format_weights["T20"]["bowling"]

        # Batting Form
        batting_metrics = {
            metric: recent_data.groupby("Player", group_keys=False).apply(
                lambda g: compute_ewma(g, metric), include_groups=False
            )
            for metric in [
                "bat runs",
                "bat bf",
                "bat sr",
                "bat ave",
                "bat 4s",
                "bat 6s",
            ]
        }
        batting_df = pd.DataFrame(batting_metrics).reset_index()
        batting_norm = {
            col: normalize_series(batting_df[col])
            for col in ["bat runs", "bat ave", "bat sr", "bat 4s", "bat 6s"]
        }
        batting_df["Batting Form"] = sum(
            batting_weights[col] * batting_norm[col] for col in batting_weights
        )

        # Bowling Form
        bowling_metrics = {
            metric: recent_data.groupby("Player", group_keys=False).apply(
                lambda g: compute_ewma(g, metric), include_groups=False
            )
            for metric in [
                "bowl wkts",
                "bowl runs",
                "bowl econ",
                "bowl overs",
                "bowl ave",
            ]
        }
        bowling_df = pd.DataFrame(bowling_metrics).reset_index()
        bowling_df["Has Bowled"] = bowling_df["bowl overs"] > 0
        bowling_norm = {
            "bowl wkts": normalize_series(bowling_df["bowl wkts"]),
            "bowl ave": 100
            - normalize_series(bowling_df["bowl ave"].replace(0, np.inf)),
            "bowl econ": 100
            - normalize_series(bowling_df["bowl econ"].replace(0, np.inf)),
        }
        bowling_df["Bowling Form"] = np.where(
            bowling_df["Has Bowled"],
            (
                bowling_weights["bowl wkts"] * bowling_norm["bowl wkts"]
                + bowling_weights["bowl ave"] * bowling_norm["bowl ave"]
                + bowling_weights["bowl econ"] * bowling_norm["bowl econ"]
            ),
            30,  # NaN for non-bowlers or no data
        )

        # Fielding Form
        fielding_metrics = {
            metric: recent_data.groupby("Player", group_keys=False).apply(
                lambda g: compute_ewma(g, metric), include_groups=False
            )
            for metric in ["field ct", "field st", "field ct wk"]
        }
        fielding_df = pd.DataFrame(fielding_metrics).reset_index()
        fielding_norm = {
            col: normalize_series(fielding_df[col])
            for col in ["field ct", "field st", "field ct wk"]
        }
        fielding_df["Fielding Form"] = (
            0.5 * fielding_norm["field ct"]
            + 0.3 * fielding_norm["field st"]
            + 0.2 * fielding_norm["field ct wk"]
        )

        # Merge forms, keeping all players from player_df
        form_df = (
            player_df[["Player", "Credits", "Player Type", "Team"]]
            .drop_duplicates("Player")
            .merge(batting_df[["Player", "Batting Form"]], on="Player", how="left")
            .merge(bowling_df[["Player", "Bowling Form"]], on="Player", how="left")
            .merge(fielding_df[["Player", "Fielding Form"]], on="Player", how="left")
        )

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
                    f"{row['Months of Data'] + 1}\t"
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
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))

        print("Starting IPL data preprocessing...")
        df = self.load_data()
        combined_df = self.include_all_squad_players(df)
        form_scores = self.calculate_form(combined_df)
        print("\n\nIPL form scores calculated successfully")
        form_scores.to_csv(self.output_file, index=False)
