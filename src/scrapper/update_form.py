import pandas as pd


def assign_mean_values(csv_path, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # List of columns to calculate means for
    form_columns = ["Batting Form", "Bowling Form", "Fielding Form"]

    # Calculate mean values for each Player Type
    role_means = df.groupby("Player Type")[form_columns].mean()

    # Assign mean values based on Player Type only if the value is null
    for col in form_columns:
        df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(), "Player Type"].map(
            role_means[col]
        )

    # Save the updated CSV
    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to {output_path}")


# Example usage
assign_mean_values(
    "../data/ipl/player_form_scores.csv", "../data/ipl/player_form_scores_final.csv"
)
