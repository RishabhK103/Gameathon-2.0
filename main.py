from src.optimizer import optimize_fantasy_team
from src.data import preprocess_ipl_data


def main():
    preprocess_ipl_data()

    best_team = optimize_fantasy_team()
    if best_team is not None:
        print("\nOptimal Fantasy Team:")
        print(best_team)
    else:
        print("No valid team could be formed. Check player roles and constraints.")


if __name__ == "__main__":
    main()
