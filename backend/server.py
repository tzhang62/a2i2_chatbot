# backend/server.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama_0220 import simulate_dual_role_conversation, simulate_interactive_single_turn, conversation_manager
import subprocess
import os
import json
import traceback
from pathlib import Path
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time

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
        #"*.onrender.com",  # Allow all Render.com subdomains
        #"emergency-chatbot-backend.onrender.com"  # Your specific Render domain
    ]
)

# Get base directory from environment variable or use default
BASE_DIR = os.getenv('A2I2_BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure paths relative to base directory
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "results/answer_80.jsonl")
PERSONA_FILE_PATH = os.path.join("/Users/tzhang/projects/A2I2/data_for_train/persona.json")
DIAL_FILE_PATH = os.path.join("/Users/tzhang/projects/A2I2/data_for_train/dialogue_1.json")
PYTHON_SCRIPT = os.path.join("/Users/tzhang/projects/A2I2/backend/ollama_0220.py")

# Load persona and dialogue data
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        import pdb; pdb.set_trace()
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
    """Get persona data for a specific town person."""
    try:
        # Load persona data
        persona_data = load_json_file(PERSONA_FILE_PATH)
        
        # Convert town_person to lowercase for case-insensitive matching
        town_person_lower = town_person.lower()
        
        if town_person_lower not in persona_data:
            raise HTTPException(status_code=404, detail=f"Persona not found for {town_person}")
            
        return {"persona": persona_data[town_person_lower]}
    except Exception as e:
        import pdb; pdb.set_trace()
        print(f"Error in get_persona: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
            # Handle interactive mode - In this mode, we only generate a response from the town person
            # The Operator's messages always come from the user
            
            # Use a stable session ID that doesn't change between requests
            session_id = f"{town_person_lower}_session"
            print(f"Using session ID: {session_id}")
            
            # Get user input speaker from request
            speaker = data.get("speaker", "Operator")
            print(f"User input speaker: {speaker}")
            
            # If the speaker is not Operator, we don't generate a response
            if speaker != "Operator":
                return {"error": "In interactive mode, only Operator messages can be sent by the user"}
            
            # Save user input to conversation history if provided
            if user_input:
                conversation_manager.add_message(session_id, speaker, user_input)
                print(f"Added user input to history: {user_input}")
            
            # Get the history to determine conversation state
            history = conversation_manager.get_history(session_id, max_turns=10)  # Increase max_turns to get full history
            print(f"Current history: {history}")
            
            # Determine which turn to use based on the conversation history
            # Get more accurate message count by counting newlines plus 1 (if history exists)
            message_count = history.count("\n") + 1 if history else 0
            print(f"Message count: {message_count}")
            
            # Pick the appropriate turn based on message count
            if message_count == 1:  # Just user's first message
                turn_index = 0  # Initial resistance
            elif message_count == 2:  # User + town person's first response
                turn_index = 1  # Continued resistance
            elif message_count == 3:  # Second user message
                turn_index = 1  # Continued resistance
            elif message_count == 4:  # User + town person's second response
                turn_index = 2  # Starting agreement
            elif message_count == 5:  # Third user message
                turn_index = 2  # Starting agreement
            else:  # Final rounds
                turn_index = 3  # Final agreement
            
            # Get the appropriate turn
            turn = [
                {
                    "speaker": town_person_lower,
                    "prompt": """System: You are {name} responding to a Fire Department Agent during an emergency.
                    Based on your background: {persona}
                    
                    Relevant examples:
                    {context}
                    
                    Respond appropriately to the operator's input. For example, if the operator says 'hello', respond with 'hi'.
                    Be flexible and creative in your response, considering the context and examples provided.
                    
                    Current conversation:
                    {history}
                    
                    Format your output as a direct response without any prefix.""",
                    "category": "response_to_operator_greetings"
                },
                {
                    "speaker": town_person_lower,
                    "prompt": """System: You are {name} still showing resistance to evacuation.
                    Based on your background: {persona}
                    
                    Relevant examples:
                    {context}
                    
                    Respond appropriately to the operator's input. For example, if the operator expresses concern, address it with a relevant response.
                    Be flexible and creative in your response, considering the context and examples provided.
                    
                    Current conversation:
                    {history}
                    
                    Format your output as a direct response without any prefix.""",
                    "category": "response_to_operator_greetings"
                },
                {
                    "speaker": town_person_lower,
                    "prompt": """System: You are {name} starting to agree to evacuate.
                    Based on your background: {persona}
                    
                    Relevant examples:
                    {context}
                    
                    Respond appropriately to the operator's input. For example, if the operator suggests evacuation, show agreement.
                    Be flexible and creative in your response, considering the examples provided.
                    
                    Current conversation:
                    {history}
                    
                    Format your output as a direct response without any prefix.""",
                    "category": "progression"
                },
                {
                    "speaker": town_person_lower,
                    "prompt": """System: You are {name} finally agreeing to evacuate.
                    Based on your background: {persona}
                    
                    Relevant examples:
                    {context}  // Ensure this context is specific to Bob
                    
                    Respond with clear agreement to evacuate. Acknowledge the operator's instructions and express readiness to leave.
                    Ensure the conversation is closed with a final statement of agreement.
                    
                    Current conversation:
                    {history}
                    
                    Format your output as a direct response without any prefix.""",
                    "category": "closing"
                }
            ][turn_index]
            
            print(f"Using turn index {turn_index}, category: {turn['category']}")
            
            # Generate the town person's response
            try:
                response, retrieved_info = simulate_interactive_single_turn(
                    town_person_lower,
                    user_input,
                    speaker=town_person_lower,  # The input speaker is always Operator
                    persona=persona_data[town_person_lower],
                    turn=turn,
                    session_id=session_id
                )
                
                print(f"Generated response: {response}")
                conversation_manager.add_message(session_id, town_person_lower, response)
                return {
                    "response": response,
                    "retrieved_info": retrieved_info,
                    "category": turn["category"]
                }
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                traceback.print_exc()
                return {"error": f"Error generating response: {str(e)}"}
    # except:
    #     #import pdb; pdb.set_trace()
    #     print("Error in chat endpoint")
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}
    #     continue

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

    