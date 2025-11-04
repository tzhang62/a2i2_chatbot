# Emergency Response Chatbot

An interactive chatbot system that simulates conversations between Fire Department Operators and town residents during emergency situations.

## Prerequisites

- Python 3.8 or higher
- Node.js and npm (for frontend)
- Git
- OpenAI API key (sign up at https://platform.openai.com/)

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
git clone https://github.com/tzhang62/a2i2_chatbot
cd a2i2_chatbot
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

5. Set up environment variables:
Create a `.env` file in the project root directory:
```bash
# In the project root directory (A2I2/)
touch .env
```

Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** Never commit your `.env` file to version control. It should be listed in `.gitignore`.

To get your OpenAI API key:
- Go to https://platform.openai.com/
- Sign up or log in
- Navigate to API keys section
- Create a new API key
- Copy and paste it into your `.env` file

6. Install frontend dependencies:
```bash
cd ../frontend
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
│   ├── ollama_0220.py (deprecated - uses Ollama)
│   ├── ollama_0220_openai.py (uses OpenAI API)
│   ├── requirements.txt
│   └── data_for_train/
├── frontend/
│   ├── js/
│   │   └── chat.js
│   ├── css/
│   │   └── styles.css
│   └── index.html
├── .env (not tracked in git)
└── README.md
```

## Features

- Interactive mode: Chat with town residents as a Fire Department Operator
- Auto mode: Generate complete conversations automatically
- Real-time message display
- Conversation history tracking
- Retrieved information display

## Configuration

- **Backend API URL:** Configure in `frontend/js/chat.js`
- **Port settings:** Backend runs on port 8001, frontend on port 8000
- **OpenAI Model:** The system uses `gpt-4o-mini` by default. You can change this in `backend/ollama_0220_openai.py` in the `send_to_openai()` function
- **Environment Variables:**
  - `OPENAI_API_KEY`: Your OpenAI API key (required)
  - `A2I2_BASE_DIR`: Optional base directory for the application

## next time using server
```bash
cd A2I2
source venv/bin/activate (for activate environment)
git pull
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001
```
Open another terminal:
```
cd frontend
python -m http.server 8000
```
then go to this website: 
http://localhost:8000

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


## Migration from Ollama to OpenAI

This project has been updated to use OpenAI API instead of Ollama for the language model backend. Here are the key changes:

### What Changed:
- **New file:** `backend/ollama_0220_openai.py` replaces the Ollama implementation
- **Updated dependencies:** `openai` package added to `requirements.txt`
- **Environment variable required:** `OPENAI_API_KEY` must be set in `.env` file
- **Old file:** `backend/ollama_0220.py` is deprecated but kept for reference

### Why OpenAI?
- More reliable API with better uptime
- Access to advanced GPT models (gpt-4o-mini by default)
- No need to run a local model server
- Better performance and response quality

### Cost Considerations:
OpenAI API is a paid service. The `gpt-4o-mini` model is cost-effective for most use cases. Monitor your usage at https://platform.openai.com/usage.

### Switching Back to Ollama (Optional):
If you need to use Ollama instead:
1. Change the import in `backend/server.py` from `ollama_0220_openai` to `ollama_0220`
2. Install and run Ollama locally
3. No API key needed

## License

[Your License Here]
