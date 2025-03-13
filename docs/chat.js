// frontend/chat.js

// Helper function to get URL query parameters
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
  }
  
  const person = getQueryParam("person") || "Unknown";
  document.getElementById("chat-title").textContent = `Chat with Fire Department Operator as ${person}`;
  
  const chatWindow = document.getElementById("chat-window");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const autoModeBtn = document.getElementById("auto-mode-btn");
  const interactiveModeBtn = document.getElementById("interactive-mode-btn");
  
  let isAuto = false;
  let messages = [];
  
  // Function to render chat messages in the chat window
  function renderMessages() {
    chatWindow.innerHTML = "";
    messages.forEach(msg => {
      const p = document.createElement("p");
      // Add a class based on the sender for styling
      p.className = "message " + msg.sender.toLowerCase();
      p.innerHTML = `<strong>${msg.sender}:</strong> ${msg.text}`;
      chatWindow.appendChild(p);
    });
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
  
  // Function for sending a message in interactive mode
  function sendMessage() {
    const userInput = chatInput.value.trim();
    if (!userInput) return;
    
    messages.push({sender: "You", text: userInput});
    renderMessages();
    chatInput.value = "";
    
    // Send the interactive chat request to the backend
    fetch("http://localhost:8001/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        townPerson: person,
        userInput: userInput,
        mode: "interactive"
      })
    })
    .then(response => response.json())
    .then(data => {
      messages.push({sender: "Operator", text: data.response});
      renderMessages();
    })
    .catch(error => {
      console.error("Error:", error);
    });
  }
  
  // Function to generate an auto conversation
  function autoGenerateChat() {
    fetch("http://localhost:8001/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        townPerson: person,
        userInput: "",
        mode: "auto"
      })
    })
    .then(response => response.json())
    .then(data => {
      // Assume the backend returns a newline-separated string of messages
      const lines = data.response.split("\n").filter(line => line.trim() !== "");
      messages = lines.map((line, index) => ({
        sender: index % 2 === 0 ? "Operator" : person,
        text: line.trim()
      }));
      renderMessages();
    })
    .catch(error => {
      console.error("Error in auto chat:", error);
    });
  }
  
  // Event listeners for the buttons
  sendBtn.addEventListener("click", sendMessage);
  
  autoModeBtn.addEventListener("click", () => {
    isAuto = true;
    // Hide the input section when in auto mode
    document.getElementById("chat-input-section").style.display = "none";
    autoGenerateChat();
  });
  
  interactiveModeBtn.addEventListener("click", () => {
    isAuto = false;
    // Show the input section when in interactive mode
    document.getElementById("chat-input-section").style.display = "flex";
  });
  
  // Start in interactive mode by default
  document.getElementById("chat-input-section").style.display = "flex";
  