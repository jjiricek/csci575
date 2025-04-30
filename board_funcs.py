import chess
import numpy as np
import torch
from torch_geometric.data import Data
import numpy as np

piece_map = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_graph(board: chess.Board) -> Data:
    node_features = []
    node_indices = []
    edge_index = []

    squares = list(board.piece_map().keys())
    square_to_node = {sq: i for i, sq in enumerate(squares)}

    for sq in squares:
        piece = board.piece_at(sq)
        vec = [0] * 12
        vec[piece_map[piece.symbol()]] = 1
        node_features.append(vec)
        node_indices.append(sq)

    # Add edges based on attacks
    for i, sq1 in enumerate(squares):
        for j, sq2 in enumerate(squares):
            if board.is_attacked_by(board.turn, sq2) and sq1 == sq2:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data



def board_to_tensor(board):
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Layers 0-11: Piece positions
    for i, piece in enumerate(pieces):
        piece_type = chess.Piece.from_symbol(piece).piece_type
        color = chess.WHITE if piece.isupper() else chess.BLACK
        for square in board.pieces(piece_type, color):
            row, col = divmod(square, 8)
            tensor[i, row, col] = 1.0

    # Layer 12: Turn (all 1s if white to move, 0s if black)
    tensor[12] = np.ones((8, 8), dtype=np.float32) if board.turn else np.zeros((8, 8), dtype=np.float32)
    
    # Layer 13: Reserved for future metadata
    return tensor
