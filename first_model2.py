# simple random forest implementation with the new goal 

# new goal: pattern classification (forks, pins, skewers, and mates)

import pandas as pd
import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical

# import data
df = pd.read_csv('data/puzzles_tactic_length_1.csv')
print('Dataset successfully loaded from CSV.')

# print different tactic types
print(f"Available tactic types: {df['TacticLabel'].unique()}")

################################
# RANDOM FOREST IMPLEMENTATION #
################################

# define fxn to encode FEN to 64-board vector, plus turn indicator
def fen_to_features(fen):
    board = chess.Board(fen)
    piece_map = board.piece_map()
    features = [0] * 64
    for square, piece in piece_map.items():
        features[square] = ord(piece.symbol())
    features.append(1 if board.turn == chess.WHITE else 0)
    return features

# prepare feature matrix & labels
X_rf = df['FEN'].apply(fen_to_features).to_list()
X_rf = np.array(X_rf)

print("Encoding labels...")
le = LabelEncoder()
y_rf = le.fit_transform(df['TacticLabel'])

# train test split data
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# initialize random forest classifier
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_rf, y_train_rf)
rf_preds = rf_model.predict(X_test_rf)

# print results
rf_accuracy = accuracy_score(y_test_rf, rf_preds)
print(f"Random Forest Accuracy: {rf_accuracy}")


    
