from data_preprocessing import preprocess_data
from team_filter import filter_match_players
from team_optimization import optimize_dream11_team

def main():
    print("Preprocessing data...")
    preprocess_data(
        "../data/batting_averages.csv", 
        "../data/bowling_averages.csv", 
        "../data/fielding_averages.csv", 
        "../data/squad.csv", 
        "../data/merged_data.csv"
    )
    
    team1 = input("Enter first team (e.g., PBKS): ").strip().upper()
    team2 = input("Enter second team (e.g., GT): ").strip().upper()
    print(f"Filtering players for {team1} vs {team2}...")
    match_df = filter_match_players("../data/squad.csv", team1, team2, "../data/match_players.csv")
    
    print("Optimizing Dream11 team...")
    dream11_team = optimize_dream11_team("../data/match_players.csv", "../data/dream11_team.csv")
    
    print(f"\nYour Dream11 Team for {team1} vs {team2}:")
    print(dream11_team.to_string(index=False))

if __name__ == "__main__":
    main()