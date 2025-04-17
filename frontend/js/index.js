// Function to handle person selection
function selectPerson(person) {
    // Store selected person in session storage
    sessionStorage.setItem('selectedPerson', person);
    
    // Redirect to chat page on the same port
    window.location.href = 'http://localhost:8000/chat.html';
} 