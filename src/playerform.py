# ==============================================================================
# Updating the Player Form
#
# Author: Aditya Godse
#
# This script cleans the player stats data, updates the players from the YAML file,
# filters only players that exist in the squad list, and calculates the recent form
# scores for batting, bowling, and fielding based on the recent matches.
#
# Columns expected:
#   Batting: Player,Mat,Inns,NO,Runs,HS,Ave,BF,SR,100,50,0,4s,6s,Team,Start Date,End Date
#   Bowling: Player,Mat,Inns,Overs,Mdns,Runs,Wkts,BBI,Ave,Econ,SR,4,5,Team,Start Date,End Date
#   Fielding: Player,Mat,Inns,Dis,Ct,St,Ct Wk,Ct Fi,MD,D/I,Team,Start Date,End Date
#
# Usage:
#   python3 -m src.playerform
#
# ==============================================================================

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
        try:
            with open("config.yaml", "r") as stream:
                self.config = yaml.safe_load(stream)
        except Exception as e:
            print(Fore.RED + f"Error reading YAML config file: {e}")
            sys.exit(1)

        # File paths from config
        self.bowling_file = self.config["data"]["bowling_file"]
        self.batting_file = self.config["data"]["batting_file"]
        self.fielding_file = self.config["data"]["fielding_file"]
        self.output_file = self.config["data"]["output_file"]
        self.player_form_file = self.config["data"]["player_form"]
        self.previous_months = self.config["data"]["previous_months"]
        self.decay_rate = self.config["data"]["decay_rate"]
        self.key_cols = ["Player", "Team", "Start Date", "End Date", "Mat"]

    def load_data(self):
        try:
            # Load the merged form scores (which include Bowler Type)
            df = pd.read_csv(self.player_form_file)
        except Exception as e:
            print(Fore.RED + f"Error reading merged CSV file: {e}")
            sys.exit(1)
        return df

    def filter_players_by_squad(self, df):
        try:
            squad_df = pd.read_csv(self.config["data"]["squad_file"])
        except Exception as e:
            print(Fore.RED + f"Error reading squad CSV file: {e}")
            sys.exit(1)

        # Use the full player names from the squad file as valid names.
        valid_players = squad_df["Player Name"].dropna().tolist()
        filtered_df = df[df["Player"].isin(valid_players)].copy()

        squad_players_set = set(valid_players)
        df_players_set = set(filtered_df["Player"])
        missing_players = squad_players_set - df_players_set

        if missing_players:
            print(Fore.YELLOW + "Missing players from data:")
            for player in sorted(missing_players):
                print(f"- {player}")
        else:
            print(Fore.GREEN + "All players from the squad CSV file are present in the DataFrame.")

        print(f"\nExtracted players: {len(df_players_set)} / {len(squad_players_set)}")
        print(f"Missing players: {len(missing_players)}\n")

        # Merge additional columns from squad file including Bowler Type if needed
        filtered_df = filtered_df.merge(
            squad_df[["Credits", "Player Type", "Player Name", "ESPN player name", "Bowler Type"]],
            left_on="Player",
            right_on="ESPN player name",
            how="left",
        )
        filtered_df["Player"] = filtered_df["Player Name"]
        filtered_df.drop(["ESPN player name", "Player Name"], axis=1, inplace=True)

        return filtered_df

    def run(self):
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        print(Fore.CYAN + "Starting data preprocessing...")
        df = self.load_data()
        filtered_df = self.filter_players_by_squad(df)
        # If calculate_form is needed, call it here.
        filtered_df.to_csv(self.output_file, index=False)
        print(Fore.GREEN + "\n\nFiltered player form scores saved successfully")

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
