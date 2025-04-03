# Emergency Response Chatbot

An interactive chatbot system that simulates conversations between Fire Department Operators and town residents during emergency situations.

## Prerequisites

- Python 3.8 or higher
- Node.js and npm (for frontend)
- Git

## Installation

1. Install Python 3.8:
```bash
# On macOS (using Homebrew):
brew install python@3.8

# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3.8 python3.8-venv

# On Windows:
# Download and install Python 3.8 from https://www.python.org/downloads/release/python-380/
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/A2I2.git
cd A2I2
```

3. Create and activate a Python virtual environment with Python 3.8:
```bash
# Create virtual environment with Python 3.8
python3.8 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

4. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

5. Install frontend dependencies:
```bash
cd ../frontend
npm install
```

## Running the Application

1. Make sure your virtual environment is activated:
```bash
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Start the backend server:
```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001
```

3. Start the frontend server:
```bash
cd frontend
python -m http.server 8000
```

4. Open your browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
A2I2/
├── backend/
│   ├── server.py
│   ├── ollama_0220.py
│   ├── requirements.txt
│   └── data_for_train/
├── frontend/
│   ├── js/
│   │   └── chat.js
│   ├── css/
│   │   └── styles.css
│   └── index.html
└── README.md
```

## Features

- Interactive mode: Chat with town residents as a Fire Department Operator
- Auto mode: Generate complete conversations automatically
- Real-time message display
- Conversation history tracking
- Retrieved information display

## Configuration

- Backend API URL: Configure in `frontend/js/chat.js`
- Port settings: Backend runs on port 8001, frontend on port 3000

## next time using server
'''bash
cd A2I2
source venv/bin/activate
git pull
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001
cd frontend
python -m http.server 8000
'''
then go to this website: http://localhost:8000

## Git Commands

To commit your changes to the repository:

```bash
# Check status of your changes
git status

# Add all changed files to staging
git add .

# Or add specific files
git add filename

# Create a commit with a descriptive message
git commit -m "Your commit message describing the changes"

# Push changes to remote repository
git push origin main  # or your branch name
```

Example commit messages:
- "Fix: Update virtual environment setup instructions"
- "Feature: Add new town person character"
- "Update: Modify conversation flow logic"
- "Fix: Resolve error message display issue"

## License

[Your License Here]