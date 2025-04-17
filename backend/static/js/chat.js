// Get DOM elements
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const autoModeBtn = document.getElementById('auto-mode-btn');
const interactiveModeBtn = document.getElementById('interactive-mode-btn');
const chatInputSection = document.getElementById('chat-input-section');
const speakerToggleBtn = document.getElementById('speaker-toggle-btn');
const retrievedInfo = document.getElementById('retrieved-info');
const retrievedContent = document.getElementById('retrieved-content');

// API Configuration
const API_BASE_URL = 'http://localhost:8001';

// Get selected person from session storage
const selectedPerson = sessionStorage.getItem('selectedPerson');
const personaData = JSON.parse(sessionStorage.getItem('personaData'));

// Update chat title
document.getElementById('chat-title').textContent = `Chat with ${selectedPerson} as Emergency Operator`;

let isAutoMode = false;
let useJulie = false; // Flag to determine if Julie should interact
let messages = [];
let lineCounter = 1;
let currentSpeaker = 'Operator'; // Default speaker

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

// Create Julie mode options container (initially hidden)
const julieOptionsContainer = document.createElement('div');
julieOptionsContainer.id = 'julie-options-container';
julieOptionsContainer.className = 'julie-options-container';
julieOptionsContainer.style.display = 'none';
julieOptionsContainer.innerHTML = `
    <p>Julie Mode:</p>
    <div class="toggle-buttons">
        <button id="auto-julie-btn" class="active">Auto Julie</button>
        <button id="manual-julie-btn">Type as Julie</button>
    </div>
`;

// Insert interaction mode toggle after chat input section
chatInputSection.parentNode.insertBefore(interactionModeContainer, chatInputSection.nextSibling);
// Insert Julie options after interaction mode container
interactionModeContainer.parentNode.insertBefore(julieOptionsContainer, interactionModeContainer.nextSibling);

// Get references to new buttons
const directModeBtn = document.getElementById('direct-mode-btn');
const julieModeBtn = document.getElementById('julie-mode-btn');
const autoJulieBtn = document.getElementById('auto-julie-btn');
const manualJulieBtn = document.getElementById('manual-julie-btn');

// Flag for Julie auto mode
let isAutoJulie = true;

// Add event listeners for interaction mode buttons
directModeBtn.addEventListener('click', () => {
    useJulie = false;
    directModeBtn.classList.add('active');
    julieModeBtn.classList.remove('active');
    julieOptionsContainer.style.display = 'none';
    speakerToggleBtn.style.display = 'block'; // Show the speaker toggle when in direct mode
    chatInput.placeholder = `Type your message as ${currentSpeaker}...`;
    chatInputSection.style.display = 'flex'; // Always show chat input in direct mode
});

julieModeBtn.addEventListener('click', () => {
    useJulie = true;
    julieModeBtn.classList.add('active');
    directModeBtn.classList.remove('active');
    julieOptionsContainer.style.display = 'flex'; // Show Julie options
    speakerToggleBtn.style.display = 'none'; // Hide the speaker toggle when Julie is active
    updateJulieMode(); // Update based on current Julie mode
});

// Add event listeners for Julie mode options
autoJulieBtn.addEventListener('click', () => {
    isAutoJulie = true;
    autoJulieBtn.classList.add('active');
    manualJulieBtn.classList.remove('active');
    updateJulieMode();
});

manualJulieBtn.addEventListener('click', () => {
    isAutoJulie = false;
    manualJulieBtn.classList.add('active');
    autoJulieBtn.classList.remove('active');
    updateJulieMode();
});

// Function to update UI based on Julie mode
function updateJulieMode() {
    if (isAutoJulie) {
        chatInputSection.style.display = 'none'; // Hide input when Auto Julie is active
        if (messages.length > 0) { // Don't auto-generate if no messages yet
            generateJulieResponse(); // Auto-generate Julie's response
        }
    } else {
        chatInputSection.style.display = 'flex'; // Show input for Manual Julie
        chatInput.placeholder = 'Type as Julie...';
    }
}

// Function to generate Julie's response automatically
async function generateJulieResponse() {
    // Don't generate if already generating or auto Julie is off
    if (isGenerating || !isAutoJulie) return;
    
    isGenerating = true;
    
    // Add loading indicator
    const statusElement = document.createElement('div');
    statusElement.id = 'julieStatus';
    statusElement.classList.add('julie-hint');
    statusElement.innerHTML = '<div class="alert alert-info">Auto Julie is thinking...</div>';
    chatWindow.appendChild(statusElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    // Call API to generate Julie's response
    fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            townPerson: selectedPerson,
            userInput: "",
            mode: "interactive",
            speaker: "Julie",
            autoJulie: true
        })
    })
    .then(response => response.json())
    .then(data => {
        isGenerating = false;
        
        // Check if data is null or undefined before accessing properties
        if (!data) {
            throw new Error('Received null or undefined response from server');
        }
        
        // Check for error in response
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Get julieResponse and town person response
        const julieResponse = data.julieResponse;
        const townPersonResponse = data.response;
        
        // Remove loading indicator
        const statusElement = document.getElementById('julieStatus');
        if (statusElement) {
            statusElement.remove();
        }
        
        // Add Julie's message
        if (julieResponse) {
            addMessage("Julie", julieResponse);
        }
        
        // Add town person's response
        if (townPersonResponse) {
            addMessage(selectedPerson, townPersonResponse, data.retrieved_info);
        }
        
        // If in Auto Julie mode, continue after delay
        if (isAutoJulie) {
            setTimeout(() => {
                generateJulieResponse();
            }, 5000); // 5 second delay before next auto response
        }
    })
    .catch(error => {
        console.error('Error:', error);
        isGenerating = false;
        
        // Remove loading indicator
        const statusElement = document.getElementById('julieStatus');
        if (statusElement) {
            statusElement.remove();
        }
        
        // Show error message
        const errorElement = document.createElement('div');
        errorElement.id = 'julieStatus';
        errorElement.classList.add('julie-error');
        errorElement.innerHTML = '<div class="alert alert-danger">Error generating Julie\'s response: ' + error.message + '. Please try again or take over the conversation.</div>';
        
        chatWindow.appendChild(errorElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    });
}

// Function to toggle speaker
function toggleSpeaker() {
    currentSpeaker = currentSpeaker === 'Operator' ? selectedPerson : 'Operator';
    speakerToggleBtn.textContent = `Switch to ${currentSpeaker === 'Operator' ? selectedPerson : 'Operator'}`;
    chatInput.placeholder = `Type ${currentSpeaker}'s message...`;
}

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
function addMessage(text, sender, retrievedInfo) {
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

    // Show retrieved info if available
    if (retrievedInfo) {
        showRetrievedInfo(retrievedInfo);
    }
}

// Function to show retrieved info
function showRetrievedInfo(info) {
    if (!info) return;
    retrievedContent.innerHTML = '';
    
    // Check if this is a Bob response with full prompt
    const hasBobPrompt = info.speaker?.toLowerCase() === 'bob' && info.full_prompt;
    
    // For Bob, only show the full prompt
    if (hasBobPrompt) {
        // Create HTML with only the prompt
        let html = `
            <div class="retrieved-item">
                <div class="full-prompt">
                    <pre>${info.full_prompt.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
                </div>
            </div>`;
        
        retrievedContent.innerHTML = html;
    } else {
        // For other characters, create HTML with category and examples
        let html = `
            <div class="retrieved-item">
                <h4>Category: ${info.category || 'N/A'}</h4>
                <p><strong>Speaker:</strong> ${info.speaker || 'N/A'}</p>
                <div class="examples">
                    <strong>Examples:</strong>
                    <ul>
                        ${info.examples ? info.examples.map(ex => `<li>${ex}</li>`).join('') : '<li>No examples</li>'}
                    </ul>
                </div>
            </div>`;
        
        retrievedContent.innerHTML = html;
    }
    
    retrievedInfo.style.display = 'block';
    retrievedInfo.classList.add('visible');
}

// Function to reset line counter
function resetLineCounter() {
    lineCounter = 1;
}

// Function to add message with delay
function addMessageWithDelay(text, sender, delay, retrievedInfo) {
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
    
    // Determine speaker based on interaction mode
    const speaker = useJulie ? "Julie" : currentSpeaker;
    
    // Add user message to chat
    if (useJulie && !isAutoJulie) {
        // If Julie is handling manually, display Julie's message to the town person
        addMessage(userInput, "Julie");
    } else {
        // If direct interaction, display operator's message
        addMessage(userInput, currentSpeaker);
    }
    
    chatInput.value = '';
    
    // Disable input while waiting for response
    chatInput.disabled = true;
    sendBtn.disabled = true;
    
    let data;
    try {
        console.log(`Sending message as ${speaker}: ${userInput}`);
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
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        data = await response.json();
        console.log('Received response data:', data);
        
        // Check if data is null or undefined before accessing properties
        if (!data) {
            throw new Error('Received null or undefined response from server');
        }
        
        if (data.error) {
            // Handle error response
            console.error('Error from server:', data.error);
            addMessage(`Error: ${data.error}`, 'System');
        } else if (data.julieResponse && data.response) {
            // Handle both Julie and town person responses
            addMessage(data.julieResponse, "Julie", data.retrieved_info);
            addMessage(data.response, selectedPerson, data.retrieved_info);
            
            // If in Auto Julie mode, we need to trigger Julie's next response
            if (useJulie && isAutoJulie) {
                setTimeout(() => generateJulieResponse(), 1500); // Slight delay before Julie responds
            }
            // Toggle speaker back to Operator for next message in direct mode
            else if (currentSpeaker !== "Operator" && !useJulie) {
                toggleSpeaker();
            }
            
            // Check if this is the end of the conversation
            if ((data.category === "progression" || data.category === "final_refusal") && data.retrieved_info?.stage === "final") {
                // Disable input and show appropriate UI feedback
                chatInput.disabled = true;
                sendBtn.disabled = true;
                chatInput.placeholder = "Conversation has ended";
                
                // Add a visual indicator that the conversation has ended
                addMessage("Conversation Ended", "System");
            }
        } else if (data.response) {
            // Handle single response (for non-Julie mode)
            addMessage(data.response, selectedPerson, data.retrieved_info);
            
            // Toggle speaker back to Operator for next message in direct mode
            if (currentSpeaker !== "Operator" && !useJulie) {
                toggleSpeaker();
            }
            
            // Check if this is the end of the conversation
            if ((data.category === "progression" || data.category === "final_refusal") && data.retrieved_info?.stage === "final") {
                // Disable input and show appropriate UI feedback
                chatInput.disabled = true;
                sendBtn.disabled = true;
                chatInput.placeholder = "Conversation has ended";
                
                // Add a visual indicator that the conversation has ended
                addMessage("Conversation Ended", "System");
            }
        } else {
            console.error('Unknown response format:', data);
            addMessage('Received an unknown response format from the server.', 'System');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage(`Error: ${error.message}`, 'System');
    } finally {
        // Re-enable input unless the conversation has explicitly ended
        if (!data?.retrieved_info?.stage || data.retrieved_info.stage !== "final") {
            chatInput.disabled = false;
            sendBtn.disabled = false;
        }
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
        messages = [];
        resetLineCounter();

        // Disable controls during generation
        autoModeBtn.disabled = true;
        interactiveModeBtn.disabled = true;
        chatInput.disabled = true;
        sendBtn.disabled = true;

        // Add loading message
        addMessage("Generating conversation... This may take a moment.", "System");

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

            // Check if data is null or undefined
            if (!data) {
                throw new Error('Received null or undefined response from server');
            }

            if (data.error) {
                throw new Error(data.error);
            }

            if (!data.transcript) {
                throw new Error('No transcript received from server');
            }

            // Clear the loading message
            chatWindow.innerHTML = '';
            resetLineCounter();

            // Add the response to the chat window
            const messages = data.transcript.split('\n').filter(msg => msg.trim());
            console.log('Messages:', messages);
            
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
            // Clear any existing messages
            chatWindow.innerHTML = '';
            resetLineCounter();
            
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
    speakerToggleBtn.style.display = useJulie ? 'none' : 'block'; // Hide speaker toggle in Julie mode
    interactiveModeBtn.classList.add('active');
    autoModeBtn.classList.remove('active');
    chatWindow.innerHTML = '';
    messages = [];
    resetLineCounter();
    
    // Always set speaker to Operator in interactive mode
    currentSpeaker = 'Operator';
    
    // Set initial placeholder text
    chatInput.placeholder = `Type your message as ${useJulie ? 'Julie' : 'Operator'}...`;
    
    // Add initial instructions
    addMessage("You are now in interactive mode. As the Fire Department Operator, your goal is to convince the town person to evacuate. Type your message to begin the conversation.", "System");
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