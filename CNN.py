import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import chess
from board_funcs import board_to_tensor

label_map = {'Fork': 0, 'Skewer': 1, 'Pin': 2, 'Mate': 3}

class ChessTacticCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)  # Input: 14x8x8
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, 4)  # Output: 4 neurons

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc(x))  # Multi-label: sigmoid (independent probabilities)
        return x

class ChessTacticDataset(Dataset):
    def __init__(self, csv_path):
        import pandas as pd
        self.data = pd.read_csv(csv_path, nrows=100000)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fen = self.data.iloc[idx]['FEN']
        label = self.data.iloc[idx]['TacticLabel']
        board = chess.Board(fen)
        tensor = board_to_tensor(board)
        input_tensor = torch.tensor(tensor, dtype=torch.float32)
        
        label_tensor = torch.zeros(4)
        label_tensor[label_map[label]] = 1.0

        return input_tensor, label_tensor

if __name__ == "__main__":
    dataset = ChessTacticDataset('data/puzzles_tactic_length_1.csv')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)   

    model = ChessTacticCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'chess_tactic_cnn_weights_one_move.pth')
    print("Model weights saved to 'chess_tactic_cnn_weights_one_move.pth'")

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





