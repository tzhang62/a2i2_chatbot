# Emergency Response Chatbot

An interactive dialogue system for emergency response training and simulation.

## Features

- Interactive chat mode with AI-powered responses
- Auto-generation mode for example conversations
- Multiple town person personas
- Line-by-line conversation display
- Real-time response generation

## Setup

### Frontend (GitHub Pages)
The frontend is hosted on GitHub Pages and can be accessed at: `https://[your-username].github.io/[repo-name]`

### Backend Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Install Ollama and pull the required model:
   ```bash
   ollama pull llama3.2:latest
   ```
4. Start the backend server:
   ```bash
   python server.py
   ```

## Development

### Frontend
- HTML/CSS/JavaScript
- Located in the `/docs` directory
- Automatically deployed to GitHub Pages

### Backend
- Python/FastAPI
- Ollama for LLM integration
- Located in the `/backend` directory

## Deployment
- Frontend is automatically deployed via GitHub Pages
- Backend needs to be hosted separately (e.g., on a VPS or cloud service)

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 