import os
import argparse
from data_preprocessing import preprocess_data
from team_filter import filter_match_players
from team_optimization import optimize_dream11_team

def main(data_dir="../data"):
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # Construct file paths dynamically
    batting_file = os.path.join(data_dir, "batting_averages.csv")
    bowling_file = os.path.join(data_dir, "bowling_averages.csv")
    fielding_file = os.path.join(data_dir, "fielding_averages.csv")
    squad_file = os.path.join(data_dir, "squad.csv")
    merged_file = os.path.join(data_dir, "merged_data.csv")
    match_players_file = os.path.join(data_dir, "match_players.csv")
    dream11_team_file = os.path.join(data_dir, "dream11_team.csv")

    print("Preprocessing data...")
    preprocess_data(batting_file, bowling_file, fielding_file, squad_file, merged_file)
    
    team1 = input("Enter first team (e.g., PBKS): ").strip().upper()
    team2 = input("Enter second team (e.g., GT): ").strip().upper()
    print(f"Filtering players for {team1} vs {team2}...")
    match_df = filter_match_players(squad_file, team1, team2, match_players_file, data_dir)
    
    print("Optimizing Dream11 team...")
    dream11_team = optimize_dream11_team(match_players_file, dream11_team_file, team1, team2)
    
    print(f"\nYour Dream11 Team for {team1} vs {team2}:")
    print(dream11_team[["player_name", "team", "player_type", "runs_per_mat", "wkts_per_mat", "dis_per_mat", "credits", "role"]].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dream11 Team Optimization")
    parser.add_argument("--data-dir", default="../data", help="Directory containing data files")
    args = parser.parse_args()
    main(args.data_dir)