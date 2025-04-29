# starting model

import pandas as pd
import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# APPROACH:
# encode FEN as string of categorical tokens
# train random forest model

# load dataset
df = pd.read_csv('data/cleaned_mate_in_one_puzzles.csv').head(10000)
print("Dataset successfully loaded from CSV.")

# def fxn to turn FEN into simple feature vector
def fen_to_features(fen):
    board = chess.Board(fen)
    piece_map = board.piece_map()
    # initialize 64 board positions with 0
    features = [0] * 64
    for square, piece in piece_map.items():
        features[square] = ord(piece.symbol())
    
    # add turn indicator
    features.append(1 if board.turn == chess.WHITE else 0)
    return features

# prepare dataset
X = df['FEN'].apply(fen_to_features).tolist()
X = np.array(X)

# attempted improvement: use destination square as baseline
df['target_square'] = df['Moves'].apply(lambda m: m[-2:])

# encode moves as labels
le = LabelEncoder()
# y = le.fit_transform(df['Moves'])
y = le.fit_transform(df['target_square'])
print("Data successfully loaded into dataframe & encoded.")

# train test split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train random forest classifier
# print("Training random forest classifier...")
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)
# print("Classifier fitted to training data.")

# now let's try a neural network!
print("Training neural network ...")
model = MLPClassifier(hidden_layer_sizes=(256,128), activation='relu', max_iter=10000)
model.fit(X_train, y_train)
print("Neural network fitted to training data.")

# evaluate
# accuracy = clf.score(X_test, y_test) - for rand forest model
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")