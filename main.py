import click
import pandas as pd

from src.update import update_player_data
from src.team_build import build_team


def summarize_squad_data(file_path):
    df = pd.read_csv(file_path)
    total_players = len(df)
    playing_players = df[df["IsPlaying"] == "PLAYING"].shape[0]
    player_types = df["Player Type"].value_counts()
    print(f"\nTotal number of players: {total_players}")
    print(f"Playing players: {playing_players}")
    print(f"\n{player_types.to_string()}")
    return df["Team"].unique()


@click.command()
@click.option("--build", is_flag=True, help="Build the team using LP.")
@click.option(
    "--genetic", is_flag=True, help="Use genetic algorithm for team selection."
)
@click.option("--updateplayerform", is_flag=True, help="Update player form.")
@click.option("--update", type=int, help="Update player data for the last n months.")
@click.argument("option", required=False, type=int)
def main(build, genetic, updateplayerform, update, option):

    if build:
        update_player_data(3)


if __name__ == "__main__":
    main()
