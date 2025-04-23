import pandas as pd

df = pd.read_csv('data/lichess_db_puzzle.csv')

mate_in_one_df = df[df['Themes'].str.contains('mateIn1', case=False, na=False)]

cleaned_df = mate_in_one_df[['FEN', 'Moves']]

cleaned_df.to_csv('data/cleaned_mate_in_one_puzzles.csv', index=False)
