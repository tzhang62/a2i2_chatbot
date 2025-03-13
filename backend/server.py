# backend/server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama_0220 import simulate_dual_role_conversation, simulate_interactive_conversation
import subprocess
import os
import json
import traceback
from pathlib import Path
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

# Enable CORS with proper configuration for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "your-domain.com",
        "www.your-domain.com"
    ]
)

# Get base directory from environment variable or use default
BASE_DIR = os.getenv('A2I2_BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure paths relative to base directory
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "results/answer_80.jsonl")
PERSONA_FILE_PATH = os.path.join(BASE_DIR, "data_for_train/persona.json")
DIAL_FILE_PATH = os.path.join(BASE_DIR, "data_for_train/dialogue_1.json")
PYTHON_SCRIPT = os.path.join(BASE_DIR, "ollama_0220.py")

# Load persona and dialogue data
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}

persona_data = load_json_file(PERSONA_FILE_PATH)
dialogue_data = load_json_file(DIAL_FILE_PATH)

# Request body model
class ChatRequest(BaseModel):
    townPerson: str
    userInput: str
    mode: str  # "interactive" or "auto"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Emergency Response Chatbot Backend is running"}

@app.get("/persona/{town_person}")
async def get_persona(town_person: str):
    """Get persona information for a specific town person"""
    if town_person not in persona_data:
        raise HTTPException(status_code=404, detail="Town person not found")
    return {
        "persona": persona_data.get(town_person, ""),
        "dialogue_example": dialogue_data.get(town_person, "")
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests in both auto and interactive modes"""
    try:
        print("Received request:", request)
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        town_person = request.townPerson.lower()
        user_input = request.userInput
        mode = request.mode.lower()

        # Validate town person exists
        if town_person not in persona_data:
            raise HTTPException(status_code=404, detail="Town person not found")

        persona = persona_data[town_person]
        dialogue = dialogue_data[town_person]

        if mode == "auto":
            print('generating_auto')
            transcript = simulate_dual_role_conversation(persona, town_person, dialogue)
            return {"response": transcript}
        elif mode == "interactive_start":
            # Generate initial operator message
            print('generating_initial_message')
            response = simulate_dual_role_conversation(
                persona=persona,
                name=town_person,
                dialogue=dialogue,
                get_initial_message=True
            )
            return {"response": response}
        else:
            # Interactive mode - use the dedicated function
            print('generating_interactive')
            response = simulate_interactive_conversation(
                persona=persona,
                name=town_person,
                dialogue=dialogue,
                user_input=user_input
            )
            return {"response": response}
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

    # # Build the command to run your Python script.
    # # You can modify the command to pass additional parameters if needed.
    # command = [
    #     "python3", PYTHON_SCRIPT,
    #     "-persona", PERSONA_FILE_PATH,
    #     "-dialogue", DIAL_FILE_PATH,
    #     "--use-mps",
    #     "--town-person", town_person
    # ]

    # try:
    #     result = subprocess.run(command, capture_output=True, text=True, check=True)
    #     response_text = result.stdout
    # except subprocess.CalledProcessError as e:
    #     response_text = f"Error: {e.stderr}"

    # return {"response": response_text}

    