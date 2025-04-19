import pandas as pd


def assign_mean_values(csv_path, output_path):
    df = pd.read_csv(csv_path)

    form_columns = ["Batting Form", "Bowling Form", "Fielding Form"]

    role_means = df.groupby("Player Type")[form_columns].mean()

    for col in form_columns:
        df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(), "Player Type"].map(
            role_means[col]
        )

    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to {output_path}")


assign_mean_values(
    "data/recent_averages/player_form_scores.csv", "data/recent_averages/player_form_scores_final.csv"
)
