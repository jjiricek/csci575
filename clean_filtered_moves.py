import pandas as pd
import math

# Load the puzzle data
df = pd.read_csv("data/cleaned_filtered_puzzles.csv")

# Ensure "Moves" column is properly parsed
df['NumMoves'] = df['Moves'].apply(lambda x: len(str(x).strip().split()))

# Get the number of full moves (2 ply = 1 move)
df['TacticLength'] = df['NumMoves'] // 2

# Example: Filter and save puzzles of tactic length 1, 2, 3
for length in range(1, 4):  # mate-in-1, mate-in-2, etc.
    filtered = df[df['TacticLength'] == length]
    print(f"Tactic Length {length}: {len(filtered)} puzzles")
    filtered.to_csv(f"data/puzzles_tactic_length_{length}.csv", index=False)
