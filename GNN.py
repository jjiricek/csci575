from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch
import chess
import pandas as pd
from board_funcs import board_to_graph
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam

label_map = {'Fork': 0, 'Skewer': 1, 'Pin': 2, 'Mate': 3}

class ChessTacticGraphDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, nrows=100000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx]['FEN']
        label = self.data.iloc[idx]['TacticLabel']
        board = chess.Board(fen)

        graph = board_to_graph(board)
        graph.y = torch.tensor([label_map[label]], dtype=torch.long)

        return graph

class ChessTacticGNN(torch.nn.Module):
    def __init__(self, in_channels=12, hidden_channels=64, out_channels=4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Graph-level pooling
        return torch.sigmoid(self.fc(x))

if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from torch.nn import BCELoss
    from torch.optim import Adam
    from sklearn.model_selection import train_test_split

    dataset = ChessTacticGraphDataset('data/one_move_puzzles.csv')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = ChessTacticGNN()
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = BCELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in train_loader:
            out = model(batch)
            target = torch.zeros_like(out)
            target[range(len(batch.y)), batch.y] = 1.0
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'chess_tactic_gnn_weights_half_move.pth')
    print("Model weights saved to 'chess_tactic_gnn_weights_half_move.pth'")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            actual = torch.argmax(labels, dim=1)
            correct += (predicted == actual).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
