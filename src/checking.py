import pandas as pd
squad_file="../data/ipl/squad.csv"
squad_df=pd.read_csv(squad_file)
valid_players = squad_df["ESPN player name"].dropna().tolist()
print(valid_players)