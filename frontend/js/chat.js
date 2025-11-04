// Get DOM elements
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const autoModeBtn = document.getElementById('auto-mode-btn');
const interactiveModeBtn = document.getElementById('interactive-mode-btn');
const restartBtn = document.getElementById('restart-btn');
const chatInputSection = document.getElementById('chat-input-section');
const speakerToggleBtn = document.getElementById('speaker-toggle-btn');
const retrievedInfo = document.getElementById('retrieved-info');
const retrievedContent = document.getElementById('retrieved-content');

// API Configuration - loaded from config.js
const API_BASE_URL = window.API_CONFIG ? window.API_CONFIG.BASE_URL : 'http://localhost:8001';

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
let isGenerating = false; // Flag to prevent multiple simultaneous Julie responses

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
        console.log("Auto Julie mode activated");
        
        // Add system message for clarity
        addMessage("Auto Julie mode activated. Julie will handle the conversation automatically.", "System");
        
        // Always generate Julie's response
        generateJulieResponse();
    } else {
        chatInputSection.style.display = 'flex'; // Show input for Manual Julie
        chatInput.placeholder = 'Type as Julie...';
    }
}

// Function to generate Julie's response automatically
async function generateJulieResponse() {
    // Don't use isGenerating variable at all - instead use a DOM element to track state
    if (!isAutoJulie || document.getElementById('julieStatus')) {
        console.log("Skipping Julie response - already processing or auto Julie is off");
        return;
    }
    
    console.log("Starting Julie response generation");
    
    // Add loading indicator - this also serves as our "in progress" flag
    const statusElement = document.createElement('div');
    statusElement.id = 'julieStatus';
    statusElement.classList.add('julie-hint');
    statusElement.innerHTML = '<div class="alert alert-info">Auto Julie is thinking...</div>';
    chatWindow.appendChild(statusElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
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
        });

        const data = await response.json();
        console.log("Response from auto Julie API:", data);
        
        // Remove loading indicator
        const loadingElement = document.getElementById('julieStatus');
        if (loadingElement) loadingElement.remove();

        if (data.error) {
            throw new Error(data.error);
        }

        // Add messages to the chat
        if (data.julieResponse) {
            addMessage(data.julieResponse, "Julie", data.julieRetrievedInfo);
        } else {
            console.warn("No julieResponse in data:", data);
            addMessage("Julie tried to respond but no message was returned from the server", "System");
        }
        
        if (data.response) {
            addMessage(data.response, selectedPerson, data.retrieved_info);
        } else {
            console.warn("No response in data:", data);
        }
        
        // Check if conversation has ended
        if (data.conversation_ended) {
            addMessage("This conversation has reached its conclusion.", "System");
            
            // Disable further interaction
            isAutoJulie = false;
            
            // Update UI to show conversation has ended
            const statusElement = document.createElement('div');
            statusElement.classList.add('conversation-ended');
            statusElement.innerHTML = '<div class="alert alert-info">The conversation has ended. You may start a new conversation or review this one.</div>';
            chatWindow.appendChild(statusElement);
            chatWindow.scrollTop = chatWindow.scrollHeight;
            
            // Switch back to direct mode
            directModeBtn.click();
        }
        
    } catch (error) {
        console.error('Error generating Julie response:', error);
        
        // Remove loading indicator if it exists
        const loadingElement = document.getElementById('julieStatus');
        if (loadingElement) loadingElement.remove();
        
        // Show error message
        addMessage(`Error: ${error.message}`, "System");
    }
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

// Function to add a message to the chat - updated to store retrievedInfo and make messages clickable
function addMessage(text, sender, retrievedInfo) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender.toLowerCase()}`;
    // Add clickable cursor style
    messageDiv.style.cursor = 'pointer';
    
    // Add line number
    const lineNumber = document.createElement('span');
    lineNumber.className = 'line-number';
    lineNumber.textContent = lineCounter++;
    messageDiv.appendChild(lineNumber);
    
    // Add message content
    const content = document.createElement('div');
    content.innerHTML = `<strong>${sender}:</strong> ${text}`;
    messageDiv.appendChild(content);
    
    // Store the message data including retrievedInfo for later access
    const messageIndex = messages.length;
    messages.push({ 
        sender, 
        text, 
        lineNumber: lineCounter - 1,
        retrievedInfo: retrievedInfo // Store the retrievedInfo with the message
    });
    
    // Add click handler to show this message's retrieved info
    messageDiv.addEventListener('click', () => {
        // Highlight the clicked message
        document.querySelectorAll('.message').forEach(msg => {
            msg.classList.remove('selected');
        });
        messageDiv.classList.add('selected');
        
        // Show retrieved info for this message
        if (messages[messageIndex].retrievedInfo) {
            showRetrievedInfo(messages[messageIndex].retrievedInfo);
        } else {
            // If no retrieved info, show a message
            retrievedContent.innerHTML = `<div class="retrieved-item">
                <p>No generation details available for this message.</p>
            </div>`;
            retrievedInfo.style.display = 'block';
            retrievedInfo.classList.add('visible');
        }
    });
    
    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    // Show retrieved info if available when message is first added
    if (retrievedInfo) {
        showRetrievedInfo(retrievedInfo);
    }
}

// Function to show retrieved info
function showRetrievedInfo(info) {
    if (!info) {
        console.log('No retrieved info provided');
        return;
    }
    console.log('Retrieved info object:', JSON.stringify(info, null, 2));
    
    retrievedContent.innerHTML = '';
    
    // Check if this is a character response with full prompt
    const speakerLower = info.speaker?.toLowerCase() || '';
    console.log(`Speaker detected: ${speakerLower}`);
    
    // Always show the full prompt if available
    if (info.full_prompt) {
        console.log(`Showing full prompt for ${speakerLower}`);
        
        let title = speakerLower === 'julie' ? 'Julie\'s Generation Prompt:' : `${speakerLower}'s Response Generation:`;
        
        let html = `
            <div class="retrieved-item">
                <h4>${title}</h4>
                <div class="full-prompt">
                    <pre>${info.full_prompt.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
                </div>
            </div>`;
        
        retrievedContent.innerHTML = html;
    } else {
        console.log(`No full prompt available, showing examples instead`);
        // For other town people or if no full prompt, show other retrieved info
        let html = '<div class="retrieved-item">'; // Initialize html variable
        
        if (info.context && typeof info.context === 'string') {
            html += `<div class="retrieved-context">
                <h4>Context:</h4>
                <pre>${info.context.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
            </div>`;
        }
        
        html += '</div>'; // Close the retrieved-item div
        retrievedContent.innerHTML = html;
    }
    
    retrievedInfo.style.display = 'block';
    retrievedInfo.classList.add('visible');
}

// Function to reset line counter
function resetLineCounter() {
    lineCounter = 1;
}

// Function to reset decision status
function resetDecisionStatus() {
    const decisionIndicator = document.getElementById('decision-indicator');
    if (decisionIndicator) {
        decisionIndicator.textContent = 'Decision Status: Waiting for conversation';
    }
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
        
        // Update decision status if available
        if (data.decision_response) {
            console.log("Decision response received:", data.decision_response);
            updateDecisionStatus(data.decision_response);
        } else {
            console.log("No decision response received");
            updateDecisionStatus(data.decision_response);
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
            
            // Log retrieved info for debugging
            console.log(`Response from ${selectedPerson}:`, data.response);
            console.log(`Retrieved info for ${selectedPerson}:`, data.retrieved_info);
            
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

            // Update decision status if available in auto mode
            if (data.decision) {
                console.log("Decision received in auto mode:", data.decision);
                updateDecisionStatus(data.decision);
            } else {
                console.log("No decision received in auto mode response");
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
    resetDecisionStatus();
    
    // Add initial system message
    addMessage(`You're now in interactive mode. You can chat with ${selectedPerson} directly.`, 'System');
}

// Function to switch to auto mode
function enableAutoMode() {
    isAutoMode = true;
    chatInputSection.style.display = 'none';
    interactiveModeBtn.classList.remove('active');
    autoModeBtn.classList.add('active');
    chatWindow.innerHTML = '';
    messages = [];
    resetLineCounter();
    resetDecisionStatus();
    
    // Generate auto chat
    generateAutoChat();
}

// Function to restart conversation
async function restartConversation() {
    // Show confirmation dialog
    const confirmed = confirm('Are you sure you want to restart the conversation? All messages will be cleared.');
    
    if (!confirmed) {
        return;
    }
    
    console.log('Restarting conversation...');
    
    // Clear frontend state
    chatWindow.innerHTML = '';
    retrievedContent.innerHTML = '';
    retrievedInfo.classList.remove('visible');
    retrievedInfo.style.display = 'none';
    messages = [];
    resetLineCounter();
    resetDecisionStatus();
    
    // Reset speaker to Operator
    currentSpeaker = 'Operator';
    speakerToggleBtn.textContent = `Switch to ${selectedPerson}`;
    chatInput.placeholder = 'Type Operator\'s message...';
    
    // Clear backend conversation history (if needed)
    try {
        // You can add an API call here to clear backend session if implemented
        // For now, the backend will handle new session on next message
        console.log('Frontend state cleared');
    } catch (error) {
        console.error('Error clearing backend session:', error);
    }
    
    // Restart based on current mode
    if (isAutoMode) {
        // If in auto mode, regenerate the conversation
        addMessage('Restarting conversation in Auto Mode...', 'System');
        setTimeout(() => {
            generateAutoChat();
        }, 500);
    } else {
        // If in interactive mode, show a fresh start message
        if (useJulie && isAutoJulie) {
            addMessage('Conversation restarted. Auto Julie will begin the conversation.', 'System');
            // Restart Julie if in auto Julie mode
            setTimeout(() => {
                generateJulieResponse();
            }, 500);
        } else {
            addMessage(`Conversation restarted. You can now chat with ${selectedPerson}.`, 'System');
        }
    }
}

// Event listeners
interactiveModeBtn.addEventListener('click', enableInteractiveMode);
autoModeBtn.addEventListener('click', enableAutoMode);
restartBtn.addEventListener('click', restartConversation);
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
speakerToggleBtn.addEventListener('click', toggleSpeaker);

// Start in interactive mode by default
enableInteractiveMode();

// Function to update the decision status display
function updateDecisionStatus(decision) {
    const decisionIndicator = document.getElementById('decision-indicator');
    
    if (!decisionIndicator) {
        console.error("Could not find decision-indicator element");
        return;
    }
    
    console.log("Updating decision status:", decision);
    
    // Remove any existing classes
    decisionIndicator.classList.remove('positive', 'negative', 'neutral', 'trending-positive', 'trending-negative', 'undecided');
    
    // Determine sentiment of decision
    if (typeof decision === 'string') {
        // Positive outcomes - Likely to evacuate
        if (decision.toLowerCase().includes('evacuate') || 
            decision.toLowerCase().includes('agree') || 
            decision.toLowerCase().includes('positive') ||
            decision.toLowerCase().includes('likely')) {
            decisionIndicator.classList.add('positive');
        } 
        // Trending positive
        else if (decision.toLowerCase().includes('trending toward evacuation') || 
                 decision.toLowerCase().includes('initially positive')) {
            decisionIndicator.classList.add('trending-positive');
        }
        // Negative outcomes - Unlikely to evacuate
        else if (decision.toLowerCase().includes('refuse') || 
                 decision.toLowerCase().includes('reject') || 
                 decision.toLowerCase().includes('negative') ||
                 decision.toLowerCase().includes('resist') ||
                 decision.toLowerCase().includes('unlikely')) {
            decisionIndicator.classList.add('negative');
        }
        // Trending negative
        else if (decision.toLowerCase().includes('trending toward refusing') || 
                 decision.toLowerCase().includes('initially resistant')) {
            decisionIndicator.classList.add('trending-negative');
        }
        // Undecided but engaging
        else if (decision.toLowerCase().includes('showing resistance but still engaging') ||
                 decision.toLowerCase().includes('undecided')) {
            decisionIndicator.classList.add('undecided');
        }
        // Default neutral
        else {
            decisionIndicator.classList.add('neutral');
        }
        
        // Update text
        decisionIndicator.textContent = `Decision Status: ${decision}`;
    } else {
        // Handle case where decision is not a string
        decisionIndicator.classList.add('neutral');
        decisionIndicator.textContent = 'Decision Status: Unknown';
    }
}

// Debug button and trigger function removed for cleaner interface  