# backend/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

# Enable CORS so your frontend (which might run on a different port) can call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update these paths as needed
TRAIN_DATA_PATH = "/Users/tzhang/projects/A2I2/data_for_train/train_data_80.jsonl"
OUTPUT_FILE_PATH = "/Users/tzhang/projects/A2I2/results/answer_80.jsonl"
PYTHON_SCRIPT = "/Users/tzhang/projects/A2I2/ollama_0220.py"  # Ensure this script is in your backend directory or provide the correct path

# Request body model
class ChatRequest(BaseModel):
    townPerson: str
    userInput: str
    mode: str  # "interactive" or "auto"

@app.post("/chat")
async def chat(request: ChatRequest):
    # You can use townPerson and userInput here as needed.
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Build the command to run your Python script.
    # You can modify the command to pass additional parameters if needed.
    command = [
        "python3", PYTHON_SCRIPT,
        "-i", TRAIN_DATA_PATH,
        "-o", OUTPUT_FILE_PATH,
        "--use-mps"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        response_text = result.stdout
    except subprocess.CalledProcessError as e:
        response_text = f"Error: {e.stderr}"

    return {"response": response_text}

# To run the server, execute:
# uvicorn server:app --reload --host 0.0.0.0 --port 8000
