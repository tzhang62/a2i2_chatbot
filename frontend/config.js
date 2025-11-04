// API Configuration
// Change this to your deployed backend URL after deployment
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8001'  // Local development
    : 'https://your-backend-app.onrender.com';  // Production - UPDATE THIS after deploying backend

// Export for use in other files
window.API_CONFIG = {
    BASE_URL: API_BASE_URL
};

