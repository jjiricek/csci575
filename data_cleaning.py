import pandas as pd

# Load the dataset
df = pd.read_csv('data/lichess_db_puzzle.csv')

# Define the themes of interest (in lowercase for comparison)
themes_of_interest = ['fork', 'skewer', 'pin', 'mate']

def find_first_theme(theme_str):
    if pd.isna(theme_str):
        return None
    tokens = theme_str.lower().split()  # space-separated
    for theme in themes_of_interest:
        if theme in tokens:
            return theme.capitalize()
    return None

df['TacticLabel'] = df['Themes'].apply(find_first_theme)

filtered_df = df[df['TacticLabel'].notna()]

final_df = filtered_df[['FEN', 'Moves', 'Rating', 'TacticLabel']]

final_df.to_csv('data/cleaned_filtered_puzzles.csv', index=False)
