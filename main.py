from src.team_build import optimize_fantasy_team
from src.update import update_player_data


def main():
    update_player_data(3)

    best_team = optimize_fantasy_team()
    if best_team is not None:
        print("\nOptimal Fantasy Team:")
        print(best_team)
    else:
        print("No valid team could be formed. Check player roles and constraints.")


if __name__ == "__main__":
    main()
