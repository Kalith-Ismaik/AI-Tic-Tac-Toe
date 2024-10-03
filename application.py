from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import os
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the CNN model class
class TimeSeriesCNN(nn.Module):
    def __init__(self):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = TimeSeriesCNN()
model.load_state_dict(torch.load('time_series_cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

def preprocess_board(board):
    """Convert board to a suitable input format for the model."""
    current_board = np.zeros((1, 9, 3, 3))  # 1 batch, 9 channels, 3x3 grid
    for i, cell in enumerate(board):
        row, col = divmod(i, 3)
        if cell == 'X':
            current_board[0, 0, row, col] = 1
        elif cell == 'O':
            current_board[0, 0, row, col] = -1
    return current_board

def get_ai_move(board):
    empty_cells = [i for i, cell in enumerate(board) if cell == '']
    if not empty_cells:
        return None  # No moves available

    best_move = None
    best_score = float('-inf')
    
    for move in empty_cells:
        temp_board = board.copy()
        temp_board[move] = 'X'  # Assume AI is 'X'
        
        # Check if AI wins with this move
        if check_winner(temp_board, 'X'):
            return move
        
        # Simulate opponent's move and check if AI prevents opponent from winning
        opponent_best_move = None
        opponent_best_score = float('-inf')
        
        for opponent_move in empty_cells:
            temp_board_opponent = temp_board.copy()
            temp_board_opponent[opponent_move] = 'O'  # Assume opponent is 'O'
            
            # Check if opponent wins with this move
            if check_winner(temp_board_opponent, 'O'):
                opponent_best_move = opponent_move
                break  # Opponent can win, prevent it
        
        # If opponent can't win with any move, proceed with AI move scoring
        if opponent_best_move is None:
            input_board = preprocess_board(temp_board)
            input_tensor = torch.from_numpy(input_board).float()
            
            with torch.no_grad():
                score = model(input_tensor).item()
            
            if score > best_score:
                best_score = score
                best_move = move
    
    return best_move

def check_winner(board, player):
    # Check rows, columns, and diagonals for a win
    board_matrix = np.array(board).reshape((3, 3))
    win_combinations = [
        [board_matrix[0, :], board_matrix[1, :], board_matrix[2, :]],  # rows
        [board_matrix[:, 0], board_matrix[:, 1], board_matrix[:, 2]],  # columns
        [np.diag(board_matrix), np.diag(np.fliplr(board_matrix))]     # diagonals
    ]
    
    for combination in win_combinations:
        for line in combination:
            if np.all(line == player):
                return True
    return False

# Function to update model with new data
def update_model(new_data, target):
    input_data = preprocess_board(new_data)  # Preprocess new data as needed
    input_tensor = torch.from_numpy(input_data).float()
    target_tensor = torch.tensor([target]).float()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Print initial weights for diagnostics
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    model.train()  # Set model to training mode
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_tensor)
    loss = criterion(output, target_tensor)  # Compute loss using new data

    # Backward pass and update weights
    loss.backward()
    optimizer.step()

    # Print updated weights for diagnostics
    updated_weights = {name: param.clone() for name, param in model.named_parameters()}

    # Compare initial and updated weights to ensure they are changing
    for name in initial_weights:
        if not torch.equal(initial_weights[name], updated_weights[name]):
            print(f"Weight updated for {name}")
        else:
            print(f"No change in weight for {name}")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/style.css')
def serve_css():
    return send_from_directory('static', 'style.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('static', 'script.js')

@app.route('/ai-move', methods=['POST'])
def ai_move():
    data = request.get_json()
    board = data['board']
    move = get_ai_move(board)
    return jsonify(move=move)

@app.route('/list-static')
def list_static():
    files = os.listdir('static')
    return jsonify(files=files)

# Route to receive new data for online learning
@app.route('/update-model', methods=['POST'])
def receive_new_data():
    data = request.get_json()
    new_data = data['data']
    target = data['target']  # Target value corresponding to new data
    # Update model with new data
    update_model(new_data, target)
    return jsonify({'message': 'Model updated successfully'})

# Example route to trigger online learning (for demonstration purposes)
@app.route('/trigger-online-learning')
def trigger_online_learning():
    # Example: Trigger online learning with some predefined new data
    example_new_data = ['X', '', '', '', 'O', '', '', '', '']  # Example new data format
    target_value = 0.5  # Example target value
    update_model(example_new_data, target_value)
    return jsonify({'message': 'Online learning triggered successfully'})

if __name__ == '__main__':
    app.run(debug=True)
