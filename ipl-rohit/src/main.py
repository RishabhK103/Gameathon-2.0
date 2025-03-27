import argparse
import os
from team_filter import filter_match_players
from team_optimization import optimize_team

def main(data_dir):
    print("Preprocessing data...")
    
    # File paths
    squad_file = os.path.join(data_dir, "squad.csv")
    match_players_file = os.path.join(data_dir, "match_players.csv")
    dream_team_file = os.path.join(data_dir, "dream_team.csv")
    
    # Validate squad file existence
    if not os.path.exists(squad_file):
        raise FileNotFoundError(f"squad.csv not found in {data_dir}")
    
    # Get team names from user input
    team1 = input("Enter first team (e.g., PBKS): ").strip().upper()
    team2 = input("Enter second team (e.g., GT): ").strip().upper()
    
    print(f"Filtering players for {team1} vs {team2}...")
    
    # Filter players for the match
    match_df = filter_match_players(squad_file, team1, team2, match_players_file, data_dir)
    
    # Optimize the Dream11 team
    dream_team_df = optimize_team(match_df, dream_team_file)
    
    return dream_team_df

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dream11 Team Selection")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing data files")
    args = parser.parse_args()
    
    # Run the main function
    main(args.data_dir)