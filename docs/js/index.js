// Handle town person selection and navigation
function selectPerson(person) {
    // Store the selected person in session storage
    sessionStorage.setItem('selectedPerson', person);
    
    // Fetch the persona data before navigating
    fetch('https://a2i2-chatbot-1.onrender.com/persona/' + encodeURIComponent(person))
        .then(response => response.json())
        .then(data => {
            // Store the persona data
            sessionStorage.setItem('personaData', JSON.stringify(data));
            // Navigate to the chat page
            window.location.href = 'chat.html';
        })
        .catch(error => {
            console.error('Error fetching persona data:', error);
            alert('Error loading persona data. Please try again.');
        });
}

// Add hover effects for person cards
document.querySelectorAll('.person-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-5px)';
    });
    
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0)';
    });
}); 