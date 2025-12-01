// Observe mode JavaScript
const wsClient = new WebSocketClient();
const messagesContainer = document.getElementById('messages-container');
const gameStatusElement = document.getElementById('game-status');
const gameSetup = document.getElementById('game-setup');
const startGameBtn = document.getElementById('start-game-btn');
const numPlayersSelect = document.getElementById('num-players');
const languageSelect = document.getElementById('language');

let messageCount = 0;
let gameStarted = false;

function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

function addMessage(message) {
    messageCount++;
    
    // Clear "waiting" message if this is the first message
    if (messageCount === 1) {
        messagesContainer.innerHTML = '';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    
    // Determine message class based on sender
    if (message.sender === 'Moderator') {
        messageDiv.classList.add('moderator');
    } else if (message.sender.startsWith('Player')) {
        messageDiv.classList.add('agent');
    } else {
        messageDiv.classList.add('user');
    }
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-sender">${escapeHtml(message.sender)}</span>
            <span class="message-time">${formatTime(message.timestamp)}</span>
        </div>
        <div class="message-content">${escapeHtml(message.content)}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateGameState(state) {
    if (state.phase !== null && state.phase !== undefined) {
        const phases = ['Team Selection', 'Team Voting', 'Quest Voting', 'Assassination'];
        const phaseName = phases[state.phase] || 'Unknown';
        let statusText = `Phase: ${phaseName}`;
        
        if (state.mission_id !== null) {
            statusText += ` | Mission: ${state.mission_id}`;
        }
        if (state.round_id !== null) {
            statusText += ` | Round: ${state.round_id}`;
        }
        if (state.leader !== null) {
            statusText += ` | Leader: Player ${state.leader}`;
        }
        
        gameStatusElement.textContent = statusText;
    } else {
        gameStatusElement.textContent = 'Waiting for game to start...';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// WebSocket message handlers
wsClient.onMessage('message', (message) => {
    addMessage(message);
});

wsClient.onMessage('game_state', (state) => {
    updateGameState(state);
    // Show messages container when game starts
    if (state.status === 'running' && !gameStarted) {
        gameSetup.style.display = 'none';
        messagesContainer.style.display = 'block';
        gameStarted = true;
    }
});

wsClient.onMessage('mode_info', (info) => {
    console.log('Mode info:', info);
    if (info.mode !== 'observe') {
        console.warn('Expected observe mode, got:', info.mode);
    }
});

wsClient.onMessage('error', (error) => {
    console.error('Error from server:', error);
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message';
    errorDiv.style.background = '#ffebee';
    errorDiv.style.borderLeftColor = '#f44336';
    errorDiv.innerHTML = `
        <div class="message-header">
            <span class="message-sender" style="color: #f44336;">Error</span>
        </div>
        <div class="message-content">${escapeHtml(error.message || 'Unknown error')}</div>
    `;
    messagesContainer.appendChild(errorDiv);
});

async function startGame() {
    const numPlayers = parseInt(numPlayersSelect.value);
    const language = languageSelect.value;
    
    try {
        startGameBtn.disabled = true;
        startGameBtn.textContent = 'Starting...';
        
        const response = await fetch('/api/start-game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                num_players: numPlayers,
                language: language,
                mode: 'observe',
            }),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Hide setup, show messages
            gameSetup.style.display = 'none';
            messagesContainer.style.display = 'block';
            gameStatusElement.textContent = 'Game starting...';
            gameStarted = true;
        } else {
            alert(`Error: ${result.detail || 'Failed to start game'}`);
            startGameBtn.disabled = false;
            startGameBtn.textContent = 'Start Game';
        }
    } catch (error) {
        console.error('Error starting game:', error);
        alert(`Error: ${error.message}`);
        startGameBtn.disabled = false;
        startGameBtn.textContent = 'Start Game';
    }
}

startGameBtn.addEventListener('click', startGame);

// Connect when page loads
wsClient.onConnect(() => {
    console.log('Connected to game server');
});

wsClient.onDisconnect(() => {
    console.log('Disconnected from game server');
});

// Initialize connection
wsClient.connect();

