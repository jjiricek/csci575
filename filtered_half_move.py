import pandas as pd
import chess
import chess.pgn

# Load your previously filtered dataset
df = pd.read_csv("data/cleaned_filtered_puzzles.csv")

cleaned_rows = []

for idx, row in df.iterrows():
    fen = row['FEN']
    moves = str(row['Moves']).strip().split()
    label = row['TacticLabel']

    if len(moves) < 1:
        continue  # Skip if no moves

    try:
        board = chess.Board(fen)
        for move in moves[:-1]:  # Play all but the last move
            board.push_uci(move)

        final_fen = board.fen()
        final_move = moves[-1]

        cleaned_rows.append({
            'FEN': final_fen,
            'Move': final_move,
            'TacticLabel': label
        })

    except Exception as e:
        print(f"Error on row {idx}: {e}")
        continue

# Convert to DataFrame and save
output_df = pd.DataFrame(cleaned_rows)
output_df.to_csv("data/one_move_puzzles.csv", index=False)

print(f"Saved {len(output_df)} puzzles to 'one_move_puzzles.csv'")
