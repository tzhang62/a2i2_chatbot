// Get DOM elements
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const autoModeBtn = document.getElementById('auto-mode-btn');
const interactiveModeBtn = document.getElementById('interactive-mode-btn');
const chatInputSection = document.getElementById('chat-input-section');

// Get selected person from session storage
const selectedPerson = sessionStorage.getItem('selectedPerson');
const personaData = JSON.parse(sessionStorage.getItem('personaData'));

// Update chat title
document.getElementById('chat-title').textContent = `Chat with Emergency Operator as ${selectedPerson}`;

let isAutoMode = false;
let useJulie = false; // Flag to determine if Julie should interact
let messages = [];
let lineCounter = 1;

// Add interaction mode toggle
const interactionModeContainer = document.createElement('div');
interactionModeContainer.id = 'interaction-mode-container';
interactionModeContainer.className = 'interaction-mode-container';
interactionModeContainer.innerHTML = `
    <p>Interaction Mode:</p>
    <div class="toggle-buttons">
        <button id="direct-mode-btn" class="active">Direct Interaction</button>
        <button id="julie-mode-btn">Let Julie Handle</button>
    </div>
`;

// Insert interaction mode toggle after chat input section
chatInputSection.parentNode.insertBefore(interactionModeContainer, chatInputSection.nextSibling);

// Get references to new buttons
const directModeBtn = document.getElementById('direct-mode-btn');
const julieModeBtn = document.getElementById('julie-mode-btn');

// Add event listeners for interaction mode buttons
directModeBtn.addEventListener('click', () => {
    useJulie = false;
    directModeBtn.classList.add('active');
    julieModeBtn.classList.remove('active');
});

julieModeBtn.addEventListener('click', () => {
    useJulie = true;
    julieModeBtn.classList.add('active');
    directModeBtn.classList.remove('active');
});

// API Configuration
const API_BASE_URL = 'https://a2i2-chatbot-1.onrender.com';

// Add error handling for backend connection
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/persona/bob`);
        if (!response.ok) {
            throw new Error('Backend connection failed');
        }
    } catch (error) {
        console.error('Backend connection error:', error);
        // addMessage('Warning: Cannot connect to the backend server. The chat functionality may be limited.', 'System');
    }
}

// Check backend connection when page loads
checkBackendConnection();

// Function to add a message to the chat
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender.toLowerCase()}`;
    
    // Add line number
    const lineNumber = document.createElement('span');
    lineNumber.className = 'line-number';
    lineNumber.textContent = lineCounter++;
    messageDiv.appendChild(lineNumber);
    
    // Add message content
    const content = document.createElement('div');
    content.innerHTML = `<strong>${sender}:</strong> ${text}`;
    messageDiv.appendChild(content);
    
    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    messages.push({ sender, text, lineNumber: lineCounter - 1 });
}

// Function to reset line counter
function resetLineCounter() {
    lineCounter = 1;
}

// Function to add message with delay
function addMessageWithDelay(text, sender, delay) {
    return new Promise(resolve => {
        setTimeout(() => {
            addMessage(text, sender);
            resolve();
        }, delay);
    });
}

// Function to handle sending messages in interactive mode
async function sendMessage() {
    const userInput = chatInput.value.trim();
    if (!userInput) return;
    
    // Determine speaker based on interaction mode
    const speaker = useJulie ? "Julie" : "Operator";
    
    // Add user message to chat
    if (useJulie) {
        // If Julie is handling, display Julie's message to the town person
        addMessage(`${userInput}`, "Julie");
    } else {
        // If direct interaction, display operator's message
        addMessage(`${userInput}`, "Operator");
    }
    
    chatInput.value = '';
    
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                townPerson: selectedPerson,
                userInput: userInput,
                mode: 'interactive',
                speaker: speaker  // Pass the speaker (Julie or Operator)
            })
        });
        
        const data = await response.json();
        if (data.response) {
            addMessage(data.response, selectedPerson);
        }
    }

    catch (error) {
        console.error('Error:', error);
        // addMessage('Sorry, there was an error processing your message.', 'System');
    }
}

// Function to handle auto mode chat generation
async function generateAutoChat() {
    chatWindow.innerHTML = '';
    messages = [];
    resetLineCounter();
    
    try {
        // Show loading indicator
        addMessage('Generating conversation...', 'System');
        
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                townPerson: selectedPerson,
                userInput: '',
                mode: 'auto'
            })
        });
        
        const data = await response.json();
        if (data.response) {
            // Clear the loading message
            chatWindow.innerHTML = '';
            resetLineCounter();
            
            // Split the response into lines and filter out empty lines
            const lines = data.response.split('\n').filter(line => line.trim());
            
            // Process each line with delay
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line) {
                    // Determine sender based on line position
                    const sender = i % 2 === 0 ? 'Operator' : selectedPerson;
                    // Add varying delays: longer after operator messages, shorter after user messages
                    const delay = sender === 'Operator' ? 2000 : 1500;
                    await addMessageWithDelay(line, sender, delay);
                }
            }
        }
    } catch (error) {
        console.error('Error:', error);
        // addMessage('Sorry, there was an error generating the conversation.', 'System');
    }
}

// Function to switch to interactive mode
async function enableInteractiveMode() {
    isAutoMode = false;
    chatInputSection.style.display = 'flex';
    interactiveModeBtn.classList.add('active');
    autoModeBtn.classList.remove('active');
    chatWindow.innerHTML = '';
    messages = [];
    resetLineCounter();
    
    // Show loading message
    addMessage('Starting conversation...', 'System');
    
    try {
        // Get initial operator message
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                townPerson: selectedPerson,
                userInput: '',
                mode: 'interactive_start',  // Special mode for initial message
                context: messages.map(m => `${m.sender}: ${m.text}`).join('\n'),  // Include previous messages as context
                instructions: 'Respond creatively and flexibly based on the previous utterance.'  // Add flexibility instructions
            })
        });
        
        const data = await response.json();
        // Clear loading message
        chatWindow.innerHTML = '';
        resetLineCounter();
        
        if (data.response) {
            addMessage(data.response, 'Operator');
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Function to switch to auto mode
function enableAutoMode() {
    isAutoMode = true;
    chatInputSection.style.display = 'none';
    autoModeBtn.classList.add('active');
    interactiveModeBtn.classList.remove('active');
    chatWindow.innerHTML = '';
    messages = [];
    resetLineCounter();
    generateAutoChat();
}

// Event listeners for mode buttons
interactiveModeBtn.addEventListener('click', enableInteractiveMode);
autoModeBtn.addEventListener('click', enableAutoMode);

// Event listeners for sending messages
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Start in interactive mode by default
enableInteractiveMode();

// Clear chat history on page load
window.onload = function() {
    chatWindow.innerHTML = '';
    messages = [];
    resetLineCounter();
};