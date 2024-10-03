let board = ['', '', '', '', '', '', '', '', ''];
let currentPlayer = 'X';
const cells = document.querySelectorAll('.cell');

function makeMove(index) {
    if (board[index] === '' && currentPlayer === 'X') {
        board[index] = currentPlayer;
        document.getElementById(`cell${index}`).innerText = currentPlayer;
        if (!checkWin()) {
            currentPlayer = 'O';
            fetchMove();
        }
    }
}

function fetchMove() {
    fetch('/ai-move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ board: board })
    })
    .then(response => response.json())
    .then(data => {
        const index = data.move;
        if (board[index] === '') {
            board[index] = 'O';
            document.getElementById(`cell${index}`).innerText = 'O';
            if (!checkWin()) {
                currentPlayer = 'X';
            }
        }
    })
    .catch(error => console.error('Error fetching AI move:', error));
}

function checkWin() {
    const winPatterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // Vertical
        [0, 4, 8], [2, 4, 6]             // Diagonal
    ];
    for (let pattern of winPatterns) {
        const [a, b, c] = pattern;
        if (board[a] && board[a] === board[b] && board[a] === board[c]) {
            alert(`${board[a]} wins!`);
            return true;
        }
    }
    if (!board.includes('')) {
        alert('It\'s a draw!');
        return true;
    }
    return false;
}

function resetGame() {
    board = ['', '', '', '', '', '', '', '', ''];
    cells.forEach((cell, index) => {
        cell.innerText = '';
        cell.onclick = () => makeMove(index);
    });
    currentPlayer = 'X';
}

// Initialize the game by adding event listeners to each cell
cells.forEach((cell, index) => {
    cell.onclick = () => makeMove(index);
});

document.getElementById('update-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const boardInput = document.getElementById('board').value.split(',');
    const target = parseFloat(document.getElementById('target').value);

    const data = {
        data: boardInput,
        target: target
    };

    fetch('/update-model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        console.log(result);
        alert('Model updated successfully');
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to update model');
    });
});
