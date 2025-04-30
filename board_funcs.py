import chess
import numpy as np

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
