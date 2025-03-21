// Get DOM elements
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const autoModeBtn = document.getElementById('auto-mode-btn');
const interactiveModeBtn = document.getElementById('interactive-mode-btn');
const chatInputSection = document.getElementById('chat-input-section');
const retrievedContent = document.getElementById('retrieved-content');
const retrievedInfo = document.getElementById('retrieved-info');
const speakerToggleBtn = document.getElementById('speaker-toggle-btn');

// Get selected person from session storage
const selectedPerson = sessionStorage.getItem('selectedPerson');
const personaData = JSON.parse(sessionStorage.getItem('personaData'));

// Track current speaker
let currentSpeaker = 'Operator';

// Update chat title
document.getElementById('chat-title').textContent = `Chat with ${selectedPerson} as Emergency Operator`;

let isAutoMode = false;
let messages = [];
let lineCounter = 1;

// API Configuration
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8001'
    : 'https://your-domain.com:8001';  // Replace with your actual domain

// Function to toggle speaker
function toggleSpeaker() {
    currentSpeaker = currentSpeaker === 'Operator' ? selectedPerson : 'Operator';
    speakerToggleBtn.textContent = `Switch to ${currentSpeaker === 'Operator' ? selectedPerson : 'Operator'}`;
    chatInput.placeholder = `Type ${currentSpeaker}'s message...`;
}

// Function to display retrieved information
function displayRetrievedInfo(info) {
    console.log('Displaying retrieved info:', info);
    if (info) {
        // Format the retrieved information
        let formattedInfo = '';
        if (typeof info === 'object') {
            if (info.context) {
                formattedInfo = info.context;
            } else if (info.examples) {
                formattedInfo = `Category: ${info.category || 'Unknown'}\nSpeaker: ${info.speaker || 'Unknown'}\n\nExample responses:\n` + 
                    info.examples.map(ex => `- ${ex}`).join('\n');
            } else {
                formattedInfo = JSON.stringify(info, null, 2);
            }
        } else {
            formattedInfo = info.toString();
        }
        
        retrievedContent.textContent = formattedInfo;
        retrievedInfo.classList.add('visible');
        retrievedInfo.style.display = 'block';
    } else {
        retrievedContent.textContent = '';
        retrievedInfo.classList.remove('visible');
        retrievedInfo.style.display = 'none';
    }
}

// Function to handle message click
function handleMessageClick(messageDiv, retrievedInfo) {
    console.log('Message clicked, retrieved info:', retrievedInfo);
    // Remove active class from all messages
    document.querySelectorAll('.message').forEach(msg => msg.classList.remove('active'));
    
    // Add active class to clicked message
    messageDiv.classList.add('active');
    
    // Display retrieved information
    displayRetrievedInfo(retrievedInfo);
}

// Function to add a message to the chat
function addMessage(text, sender, retrievedInfo = null) {
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
    
    // Add click handler if there's retrieved info
    if (retrievedInfo) {
        messageDiv.addEventListener('click', () => handleMessageClick(messageDiv, retrievedInfo));
        messageDiv.classList.add('clickable');
    }
    
    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    messages.push({ 
        sender, 
        text, 
        lineNumber: lineCounter - 1,
        retrievedInfo 
    });
}

// Function to reset line counter and clear retrieved info
function resetLineCounter() {
    lineCounter = 1;
    displayRetrievedInfo(null);
}

// Function to add message with delay
function addMessageWithDelay(text, sender, delay, retrievedInfo = null) {
    return new Promise(resolve => {
        setTimeout(() => {
            addMessage(text, sender, retrievedInfo);
            resolve();
        }, delay);
    });
}

// Function to handle sending messages in interactive mode
async function sendMessage() {
    const userInput = chatInput.value.trim();
    if (!userInput) return;
    
    // Add user message to chat
    addMessage(userInput, currentSpeaker);
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
                speaker: currentSpeaker
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received response:', data);
        
        if (data.response) {
            // Add response with retrieved info
            addMessage(data.response, currentSpeaker === 'Operator' ? selectedPerson : 'Operator', data.retrieved_info);
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your message.', 'System');
    }
}

// Function to handle auto mode chat generation
async function generateAutoChat() {
    try {
        const townPerson = selectedPerson;  // Use selectedPerson from session storage
        if (!townPerson) {
            alert('Please select a town person first.');
            return;
        }

        // Clear previous chat
        chatWindow.innerHTML = '';
        retrievedContent.innerHTML = '';
        retrievedInfo.classList.remove('visible');
        retrievedInfo.style.display = 'none';

        // Disable controls during generation
        autoModeBtn.disabled = true;
        interactiveModeBtn.disabled = true;
        chatInput.disabled = true;
        sendBtn.disabled = true;

        try {
            console.log('Generating complete conversation');
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    townPerson: townPerson,
                    mode: 'auto'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received response:', data);
            console.log('Retrieved info:', data.retrieved_info);

            if (data.error) {
                throw new Error(data.error);
            }

            if (!data || !data.transcript) {
                throw new Error('No response received from server');
            }

            // Add the response to the chat window
            const messages = data.transcript.split('\n').filter(msg => msg.trim());
            console.log('Messages:', messages);
            console.log('Retrieved info length:', data.retrieved_info ? data.retrieved_info.length : 0);
            
            for (let i = 0; i < messages.length; i++) {
                const message = messages[i];
                if (message.trim()) {
                    const [speaker, content] = message.split(':').map(s => s.trim());
                    // Pass the corresponding retrieved info for this message
                    const retrievedInfo = data.retrieved_info && data.retrieved_info[i] ? data.retrieved_info[i] : null;
                    console.log(`Message ${i}:`, { speaker, content, retrievedInfo });
                    await addMessageWithDelay(content, speaker, 1000, retrievedInfo);
                }
            }

            // Re-enable controls after generation
            autoModeBtn.disabled = false;
            interactiveModeBtn.disabled = false;
            chatInput.disabled = false;
            sendBtn.disabled = false;

        } catch (error) {
            console.error('Error:', error);
            addMessage('Error generating conversation: ' + error.message, 'System');
            
            // Re-enable controls on error
            autoModeBtn.disabled = false;
            interactiveModeBtn.disabled = false;
            chatInput.disabled = false;
            sendBtn.disabled = false;
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Error: ' + error.message, 'System');
    }
}

// Function to switch to interactive mode
async function enableInteractiveMode() {
    isAutoMode = false;
    chatInputSection.style.display = 'flex';
    speakerToggleBtn.style.display = 'block';
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
                mode: 'interactive_start'  // Special mode for initial message
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
        addMessage('Sorry, there was an error starting the conversation.', 'System');
    }
}

// Function to switch to auto mode
function enableAutoMode() {
    isAutoMode = true;
    chatInputSection.style.display = 'none';
    speakerToggleBtn.style.display = 'none';
    autoModeBtn.classList.add('active');
    interactiveModeBtn.classList.remove('active');
    chatWindow.innerHTML = '';
    messages = [];
    resetLineCounter();
    generateAutoChat();
}

// Event listeners
interactiveModeBtn.addEventListener('click', enableInteractiveMode);
autoModeBtn.addEventListener('click', enableAutoMode);
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
speakerToggleBtn.addEventListener('click', toggleSpeaker);

// Start in interactive mode by default
enableInteractiveMode();