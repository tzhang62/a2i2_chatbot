# backend/server.py
from fastapi import FastAPI, HTTPException, Request
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
        "*.onrender.com",  # Allow all Render.com subdomains
        "emergency-chatbot-backend.onrender.com"  # Your specific Render domain
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
    # Convert to lowercase for case-insensitive lookup
    town_person_lower = town_person.lower()
    if town_person_lower not in persona_data:
        raise HTTPException(status_code=404, detail="Town person not found")
    return {
        "persona": persona_data.get(town_person_lower, ""),
        "dialogue_example": dialogue_data.get(town_person_lower, "")
    }

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        town_person = data.get("townPerson")
        # Convert to lowercase for case-insensitive lookup
        town_person_lower = town_person.lower()
        user_input = data.get("userInput", "")
        mode = data.get("mode", "interactive")

        print(f"Received request - town_person: {town_person}, mode: {mode}")

        if not town_person or town_person_lower not in persona_data:
            print(f"Invalid town person: {town_person}")
            return {"error": "Invalid town person"}

        if mode == "auto":
            try:
                print(f"Starting auto mode generation for {town_person}")
                # Generate the entire conversation at once
                transcript, retrieved_info = simulate_dual_role_conversation(
                    persona_data[town_person_lower],
                    town_person # Keep original case for display
                )
                
                print(f"Generated transcript: {transcript}")
                print(f"Retrieved info: {retrieved_info}")
                
                return {
                    "transcript": transcript,
                    "retrieved_info": retrieved_info,
                    "is_complete": True
                }
            except Exception as e:
                print(f"Error in auto mode generation: {str(e)}")
                traceback.print_exc()
                return {"error": f"Error generating conversation: {str(e)}"}
            
        elif mode == "interactive":
            # Handle interactive mode
            response, retrieved_info = simulate_interactive_conversation(
                persona_data[town_person_lower],
                town_person,  # Keep original case for display
                dialogue_data[town_person_lower],
                user_input
            )
            
            return {
                "response": response,
                "retrieved_info": retrieved_info
            }
            
        else:
            return {"error": "Invalid mode"}
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

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

    