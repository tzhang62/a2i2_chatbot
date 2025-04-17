# backend/server.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama_0220 import simulate_interactive_single_turn, conversation_manager
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
DIAL_FILE_PATH = os.path.join("/Users/tzhang/projects/A2I2/data_for_train/character_lines.jsonl")
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

# Function to load JSONL file
bob_data = None
niki_data = None
lindsay_data = None
ross_data = None
michelle_data = None
with open(DIAL_FILE_PATH, 'r') as f:
    for line in f:
        dialogue_data_line = json.loads(line)
        if dialogue_data_line ['character'] == 'bob':
            bob_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'niki':
            niki_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'lindsay':
            lindsay_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'ross':
            ross_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'michelle':
            michelle_data = dialogue_data_line


persona_data = load_json_file(PERSONA_FILE_PATH)
dialogue_data = bob_data

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
        user_input = data.get("userInput", "")
        mode = data.get("mode", "interactive")
        speaker = data.get("speaker", "")
        auto_julie = data.get("autoJulie", False)
        town_person_lower = town_person.lower()
        
        # Simulating session tracking on the server side
        session_id = f"{town_person_lower}_session"

        # If in interactive mode, the session ID is stable across requests
        if mode == "interactive":
            print(f"Interactive mode: session_id = {session_id}")
            
            if auto_julie:
                # Get the conversation history
                history = conversation_manager.get_history(session_id, max_turns=11)
                # Count messages to determine conversation stage
                message_count = 0
                if history:
                    # Count messages based on the number of entries in the history
                    message_count = len(history.split('\n'))
                
                print(f"Auto Julie mode: message count = {message_count}")
                
                # Determine conversation stage based on message count
               
                
                # Stage-specific prompts for Julie
                julie_prompts = {
                    "initial": f"""System: You are Julie, a virtual assistant specializing in emergency evacuations.
                    You are initiating contact with {town_person} who needs to evacuate due to a dangerous wildfire approaching within 5 miles.
                    
                    About {town_person}: {persona_data[town_person_lower]}
                    
                    Generate a brief, clear first message to {town_person} that:
                    1. Introduces yourself as Julie, the emergency evacuation assistant
                    2. Clearly states there is an urgent wildfire evacuation
                    3. Is direct but not alarming
                    4. Is only 1-3 sentences long
                    
                    Format your response as a direct message without any prefix or quotation marks.""",
                    
                    "followup": f"""System: You are Julie, continuing an emergency evacuation conversation with {town_person}.
                    The wildfire is now 3 miles away and approaching quickly.
                    
                    About {town_person}: {persona_data[town_person_lower]}
                    
                    Previous conversation:
                    {history}
                    
                    Generate a short, persuasive follow-up response that:
                    1. Acknowledges any concerns expressed by {town_person}
                    2. Provides specific, relevant information to address those concerns
                    3. Increases the sense of urgency without causing panic
                    4. Is only 2-3 sentences long
                    
                    Format your response as a direct message without any prefix or quotation marks.""",
                    
                    "urgent": f"""System: You are Julie, in an urgent evacuation conversation with {town_person}.
                    The wildfire is now just 1 mile away and the situation is critical.
                    
                    About {town_person}: {persona_data[town_person_lower]}
                    
                    Previous conversation:
                    {history}
                    
                    Generate an urgent, compelling response that:
                    1. Emphasizes the immediate danger with specific details (fire distance, time left)
                    2. Directly addresses {town_person}'s specific concerns or objections
                    3. Offers practical solutions to their stated obstacles
                    4. Uses stronger language that conveys urgency
                    5. Is only 2-3 sentences long
                    
                    Format your response as a direct message without any prefix or quotation marks.""",
                    
                    "final": f"""System: You are Julie making a final appeal to {town_person} who is resisting evacuation.
                    The wildfire is now at the edge of town, and this is your final attempt to convince them.
                    
                    About {town_person}: {persona_data[town_person_lower]}
                    
                    Previous conversation:
                    {history}
                    
                    Generate a final, emotionally impactful appeal that:
                    1. States this is the absolute last chance to evacuate safely
                    2. Uses specific personal details about {town_person} that would motivate them
                    3. Appeals to their emotions (family, safety, responsibility)
                    4. Offers very specific help (I can call someone, I've arranged transportation)
                    5. Is only 2-3 sentences long
                    
                    Format your response as a direct message without any prefix or quotation marks."""
                }
                
                # Select appropriate prompt
                julie_prompt = {
                    "speaker": "julie",
                    "prompt": julie_prompts,
                    "category": f""
                }
                
                
                
                # Generate Julie's message first
                try:
                    julie_response = simulate_interactive_single_turn(
                        "julie",
                        "",
                        speaker="System",
                        persona=persona_data.get("julie", "A helpful virtual assistant for evacuations"),
                        turn=julie_prompt,
                        session_id=session_id
                    )[0]
                    # Add Julie's message to conversation history
                    conversation_manager.add_message(session_id, "Julie", julie_response)
                except Exception as e:
                    return {"error": f"Error generating Julie's response: {str(e)}"}
                    
            
                try:
                    # Generate town person's response
                    response, retrieved_info = simulate_interactive_single_turn(
                        town_person_lower,
                        user_input,
                        speaker=speaker,
                        persona=persona_data[town_person_lower],
                        turn=turn,
                        session_id=session_id
                    )
                    
                    # Check if response is in history and add it if not
                    history_after = conversation_manager.get_history(session_id, max_turns=11)
                    if not response in history_after:
                        # If response isn't in history already, add it explicitly
                        conversation_manager.add_message(session_id, town_person, response)
                        print(f"Explicitly added response to history: {town_person}: {response}")
                    
                    # Return town person's response
                    if isinstance(retrieved_info, dict):
                        # Include the full prompt in the retrieved info for all characters
                        full_prompt = turn["prompt"]
                        retrieved_info["full_prompt"] = full_prompt
                        
                        # Make sure speaker is set correctly
                        if town_person_lower == "niki":
                            retrieved_info["speaker"] = "niki"
                            print(f"Set Niki's speaker in retrieved_info: {retrieved_info['speaker']}")
                        elif town_person_lower == "lindsay":
                            retrieved_info["speaker"] = "lindsay"
                            print(f"Set Lindsay's speaker in retrieved_info: {retrieved_info['speaker']}")
                        elif town_person_lower == "ross":
                            retrieved_info["speaker"] = "ross"
                            print(f"Set Ross's speaker in retrieved_info: {retrieved_info['speaker']}")
                        elif town_person_lower == "michelle":
                            retrieved_info["speaker"] = "michelle"
                            print(f"Set Michelle's speaker in retrieved_info: {retrieved_info['speaker']}")
                    else:
                        # If retrieved_info is not a dict, create a new one
                        retrieved_info = {
                            "full_prompt": turn["prompt"],
                            "speaker": town_person_lower
                        }
                        print(f"Created new retrieved_info with speaker: {retrieved_info['speaker']}")
                    
                    print(f"Final retrieved info for {town_person_lower}: {retrieved_info}")
                    return {
                        "julieResponse": julie_response,
                        "response": response,
                        "retrieved_info": retrieved_info,
                        "category": turn["category"]
                    }
                except Exception as e:
                    print(f"Error in interactive mode: {str(e)}")
                    traceback.print_exc()
                    return {"error": f"Error generating response: {str(e)}"}
            else:
                # Handle regular interactive mode (not auto Julie)
                # First, add the user's message to the conversation history
                if user_input:
                   
                    conversation_manager.add_message(session_id, speaker, user_input)
                    print(f"Added user input to history: {speaker}: {user_input}")
                
                # Get the conversation history to determine stage
                history = conversation_manager.get_history(session_id, max_turns=11)
                message_count = 0
                if history:
                    message_count = len(history.split('\n'))
                
                print(f"Interactive mode: message count = {message_count}")
                
                # Special structure for Bob with branching logic
                if town_person_lower == "bob":
                    # Check if the operator is speaking personally
                    is_operator_personal = False
                    emphasizes_danger = False
                    emphasizes_value_of_life = False
                    ending_conversation = False
                    # Analyze last message if it exists and is from the Operator
                    if history and message_count > 0 and speaker == "Operator":
                        last_message = user_input.lower()
                        # Check if message emphasizes fire danger
                        if any(keyword in last_message for keyword in ["danger", "fire", "emergency", "threatening",'die']):
                            emphasizes_danger = True
                        # Check if message emphasizes that work is not worth the risk
                        if any(keyword in last_message for keyword in ["not worth", "work isn't worth", "nothing is worth", "life"]):
                            emphasizes_value_of_life = True
                        if any(keyword in last_message for keyword in ["fine", "alright", "sure", "ok","sounds good","thank",'thanks',"bye","goodbye","see you"]):
                            ending_conversation = True
                        # If both conditions are met, the operator is being personally persuasive
                        is_operator_personal = emphasizes_danger or emphasizes_value_of_life
                    
                    # Determine conversation stage based on message count and operator interaction
                    category = ""
                    if message_count == 1:
                        # Initial dismissal - Bob ignores evacuation warnings
                        category = "greetings"
                        # Find Bob's responses in the dialogue data
                        context = bob_data[category]
                        
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting. Use or adapt lines from this {category}:{context}. If the message came from Julie, show reluctance to even acknowledge her. If the message came from the Operator, be slightly more responsive but still resistant."
                    
                    elif message_count ==3 :
                        # Work-focused resistance - Bob emphasizes his work is too important
                        category = "work_resistance"
                        # Find Bob's responses in the dialogue data
                        context = bob_data[category]
                        
                        prompt_content = f"Generate a response focusing heavily on your work being too important to leave behind. Use or adapt lines from this {category}: {context}. If the previous message tried to emphasize danger, respond with skepticism. If the previous message tried to be empathetic, still refuse but with slightly less hostility."
                    
                    elif message_count == 5:
                        # Decision point - Either minimal engagement or beginning to consider evacuation
                        # category = "decision_point" if is_operator_personal else "minimal_engagement"
                        # print(f"Category: {category}")
                        
                        # # Find Bob's responses in the dialogue data
                        # context = bob_data[category]
                        
                        if emphasizes_danger:
                            category = "decision_point"
                            context = bob_data[category]
                            prompt_content = f"Generate a response showing that you're beginning to consider the evacuation warning. The operator has personally emphasized the danger of the fire. Choose from: {context} to show that you're starting to take the threat seriously."
                        else:
                            category = "minimal_engagement"
                            context = bob_data[category]
                            prompt_content = f"Generate a response with minimal engagement. Showing frustration at continued persuasion attempts. Use lines from this {category}: {context} that show resistance. Keep your response very brief and show you're disengaging from the conversation."
                    
                    elif message_count == 7:
                        # Final resolution - Either evacuation agreement or final refusal
                        if emphasizes_value_of_life or emphasizes_danger:
                            category = "progression"
                            print(f"Category: {category}")
                            # Find Bob's responses in the dialogue data
                            context = bob_data[category]
                            
                            prompt_content = f"Generate a response agreeing to evacuate. Please be flexible based on the previous message. The operator has personally convinced you that the danger is real and no work is worth risking your life. Choose from this {category}: {context} like \"Okay, I'm not stupid. Let me just grab my bag and I'll head out.\" Show that you've been convinced to prioritize your safety."
                        else:
                            category = "final_refusal"
                            print(f"Category: {category}")
                            context = bob_data[category]
                            
                            prompt_content = f"Generate your final response refusing to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context} to emphasize that you will not leave your work behind. This is your final decision and nothing will change your mind."
                    else:
                        if ending_conversation:
                            print(f"Ending conversation")
                            # category = "ending_conversation"
                            # print(f"Category: {category}")
                            # context = bob_data[category]
                
                            # prompt_content = f"Generate a response ending the conversation. Choose from this {category}: {context} to show that you're done talking."
                
                    # Create turn object with the selected category and prompt - fix formatting to avoid string format issues
                    turn = {
                        "speaker": "bob",
                        "prompt": f"You are roleplaying as Bob, \nBob's background: {persona_data['bob']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                elif town_person_lower == "niki":
                    keep_asking_questions = False
                    mentions_fire = False
                    ending_conversation = False
                    if history and message_count > 2 and speaker == "Operator":
                        last_message = user_input.lower()
                        # Check if message emphasizes fire danger
                        if any(keyword in last_message for keyword in ["?"]):
                            keep_asking_questions = True
                        if any(keyword in last_message for keyword in ["fine", "alright", "sure", "ok","sounds good","thank",'thanks',"bye","goodbye","see you"]):
                            ending_conversation = True
                        if any(keyword in last_message for keyword in ["fire", "danger", "emergency", "threatening",'die','evacuate','safety','drone', 'drones']):
                            mentions_fire = True
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = niki_data[category]
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting. Use or adapt lines from this {category}:{context}. If the message came from Julie, show reluctance to even acknowledge her. If the message came from the Operator, be slightly more responsive but shows uncertainty and unware of the danger."
                    elif message_count == 3:
                        if mentions_fire:
                            category = "progression"
                            context = niki_data[category]
                            prompt_content = f"Generate a response acknowledging the danger and agreeing to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context}"
            
                        else:
                            category = "response_to_operator_greetings"
                            context = niki_data[category]
                            prompt_content = f"Generate a response to the operator's greeting or answer the operator's question. Use or adapt lines from this {category}:{context}. If the message came from Julie, show reluctance to even acknowledge her. If the message came from the Operator, be slightly more responsive but try to confirm the danger."
                    elif message_count == 5:
                        if mentions_fire:
                            category = "progression"
                            context = niki_data[category]
                            prompt_content = f"Generate a final response acknowledging the danger and agreeing to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context}"
                        elif ending_conversation:
                            category = "closing"
                            context = niki_data[category]
                            prompt_content = f"Generate a final response to the operator or julie that agrees to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context}"
                        else:
                            category = "observations"
                            context = niki_data[category]
                        prompt_content = f"Generate a response to answer the operator's or julie's question. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 7:
                        if mentions_fire:
                            category = "progression"
                            context = niki_data[category]
                            prompt_content = f"Generate your response acknowledging the danger and agreeing to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context}"
                        elif keep_asking_questions:
                            category = "observation_2"
                            context = niki_data[category]
                            prompt_content = f"Generate your response to answer the operator's or julie's question. Please be flexible based on the previous message. Choose from this {category}: {context} "
                        else:
                            category = "progression"
                            context = niki_data[category]
                            prompt_content = f"Generate your final response finally agreeing to evacuate. Choose from this {category}: {context} "

                    else:
                        if ending_conversation:
                            category = "closing"
                            context = niki_data[category]
                            prompt_content = f"Generate your response ending the conversation. Choose from this {category}: {context} "
                            breakpoint()

                        else:
                            category = "progression"
                            context = niki_data[category]
                            prompt_content = f"Generate your final response finally agreeing to evacuate. Choose from this {category}: {context} "
                    turn = {
                        "speaker": "niki",
                        "prompt": f"You are roleplaying as Niki, \nNiki's background: {persona_data['niki']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                elif town_person_lower == "lindsay":
                    mentions_children = False
                    mentions_parents = False
                    ending_conversation = False
                    mentions_fire = False
                    if history and message_count > 0 and speaker == "Operator":
                        last_message = user_input.lower()
                        # Check if message emphasizes fire danger
                        if any(keyword and '?' in last_message for keyword in ["children", "kid", "kids"]):
                            mentions_children = True
                        if any(keyword and '?' in last_message for keyword in ["parents", "parent", "mom", "dad"]):
                            mentions_parents = True
                        if any(keyword in last_message for keyword in ["fine", "alright", "sure", "ok","sounds good","thank",'thanks',"bye","goodbye","see you"]):
                            ending_conversation = True
                        if any(keyword in last_message for keyword in ["fire", "danger", "emergency", "threatening",'die','evacuate','safety', 'drone', 'drones']):
                            mentions_fire = True
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = lindsay_data[category]
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 3:
                        if mentions_fire:
                            category = "progression"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response acknowledging the danger and agreeing to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context}"
                        else:
                            category = "response_to_operator_greetings"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response to the operator's greeting or answer the operator's question. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 5:
                        if mentions_children:
                            category = "children"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response to answer the operator's or julie's question about the children. Use or adapt lines from this {category}:{context}. "
                        elif mentions_parents:
                            category = "parents"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response to answer the operator's or julie's question about the parents. Use or adapt lines from this {category}:{context}. "
                        elif mentions_fire:
                            category = "progression"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                        elif ending_conversation:
                            category = "closing"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a final response to operator or julie. Use or adapt lines from this {category}:{context}. "
            
                        else:
                            category = "observations"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response to answer the operator's or julie's question about the fire. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 7:
                        if mentions_children:
                            category = "children"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response to answer the operator's or julie's question. Use or adapt lines from this {category}:{context}. "   
                        elif mentions_parents:
                            category = "parents"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response to answer the operator's or julie's question. Use or adapt lines from this {category}:{context}. "
                        elif mentions_fire:
                            category = "progression"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a final response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                        elif ending_conversation:
                            category = "closing"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a final response to operator or julie. Use or adapt lines from this {category}:{context}. "
                        else:
                            category = "progression"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                    else:
                        if ending_conversation:
                            category = "closing"
                            context = lindsay_data[category]
                            prompt_content = f"Generate a final response to operator or julie. Use or adapt lines from this {category}:{context}. "
                        else:
                            category = "progression"
                            context = lindsay_data[category]
                            prompt_content = f"Generate your final response finally agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                    turn = {
                        "speaker": "lindsay",
                        "prompt": f"You are roleplaying as Lindsay, \nLindsay's background: {persona_data['lindsay']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                elif town_person_lower == "ross":
                    ending_conversation = False
                    if history and message_count > 5 and speaker == "Operator":
                        last_message = user_input.lower()
                        if any(keyword in last_message for keyword in ["fine", "alright", "sure", "ok","sounds good","thank",'thanks',"bye","goodbye","see you"]):
                            ending_conversation = True
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = ross_data[category]
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 3:
                        category = "response_to_operator_greetings"
                        context = ross_data[category]
                        prompt_content = f"Generate a response to the operator's greeting or answer the operator's question. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 5:
                        category = "progression"
                        context = ross_data[category]
                        prompt_content = f"Generate a response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 7:
                        if ending_conversation:
                            category = "closing"
                            context = ross_data[category]
                            prompt_content = f"Generate a final response to operator or julie. Use or adapt lines from this {category}:{context}. "
                        else:
                            category = "progression"
                            context = ross_data[category]
                            prompt_content = f"Generate a response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                    else:
                        if ending_conversation:
                            category = "closing"
                            context = ross_data[category]
                            prompt_content = f"Generate a final response to operator or julie. Use or adapt lines from this {category}:{context}. "
                        else:
                            category = "progression"
                            context = ross_data[category]
                            prompt_content = f"Generate a final response finally agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                    turn = {
                        "speaker": "ross",
                        "prompt": f"You are roleplaying as Ross, \nRoss's background: {persona_data['ross']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                elif town_person_lower == "michelle":
                    engagement=False
                    if history and message_count > 0 and speaker == "Operator":
                        last_message = user_input.lower()
                        if any(keyword in last_message for keyword in ["worry","worried","understand","best"]):
                            engagement = True
                    ending_conversation = False
                    if any(keyword in last_message for keyword in ["fine", "alright", "sure", "ok","good","thank",'thanks',"bye","goodbye","see you"]):
                        ending_conversation = True
                    
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = michelle_data[category]
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 3:
                        category = "response_to_operator_greetings"
                        context = michelle_data[category]
                        prompt_content = f"Generate a response to the operator's greeting or answer the operator's question. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 5:
                        if engagement:
                            category = "progression"
                            context = michelle_data[category]
                            prompt_content = f"Generate a response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                        else:
                            category = "refuse_assistance"
                            context = michelle_data[category]
                            prompt_content = f"Generate a response refusing to evacuate. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 7:
                        if ending_conversation:
                            category = "progression"
                            context = michelle_data[category]
                            prompt_content = f"Generate a final response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                        else:
                            category = "refuse_assistance"
                            context = michelle_data[category]
                            prompt_content = f"Generate a final response refusing to evacuate. Use or adapt lines from this {category}:{context}. "
                    else:
                        category = "closing"
                        context = michelle_data[category]
                        prompt_content = f"Generate a final response to operator or julie. Use or adapt lines from this {category}:{context}. "
                    
                    turn = {
                        "speaker": "michelle",
                        "prompt": f"You are roleplaying as Michelle, \nMichelle's background: {persona_data['michelle']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                        
                try:
                    # Generate town person's response
                    response, retrieved_info = simulate_interactive_single_turn(
                        town_person_lower,
                        user_input,
                        speaker=speaker,
                        persona=persona_data[town_person_lower],
                        turn=turn,
                        session_id=session_id
                    )
                    
                    # Check if response is in history and add it if not
                    history_after = conversation_manager.get_history(session_id, max_turns=12)
                    if not response in history_after:
                        # If response isn't in history already, add it explicitly
                        conversation_manager.add_message(session_id, town_person, response)
                        print(f"Explicitly added response to history: {town_person}: {response}")
                    
                    # Return town person's response
                    if isinstance(retrieved_info, dict):
                        # Include the full prompt in the retrieved info for all characters
                        full_prompt = turn["prompt"]
                        retrieved_info["full_prompt"] = full_prompt
                        print(f"Added full_prompt to retrieved_info for {town_person_lower}")
                        
                        # Make sure speaker is set correctly
                        if town_person_lower == "niki":
                            retrieved_info["speaker"] = "niki"
                            print(f"Set Niki's speaker in retrieved_info: {retrieved_info['speaker']}")
                        elif town_person_lower == "lindsay":
                            retrieved_info["speaker"] = "lindsay"
                            print(f"Set Lindsay's speaker in retrieved_info: {retrieved_info['speaker']}")
                        elif town_person_lower == "ross":
                            retrieved_info["speaker"] = "ross"
                            print(f"Set Ross's speaker in retrieved_info: {retrieved_info['speaker']}")
                        elif town_person_lower == "michelle":
                            retrieved_info["speaker"] = "michelle"
                            print(f"Set Michelle's speaker in retrieved_info: {retrieved_info['speaker']}")
                    else:
                        # If retrieved_info is not a dict, create a new one
                        retrieved_info = {
                            "full_prompt": turn["prompt"],
                            "speaker": town_person_lower
                        }
                    #print(retrieved_info)
                    #print(f"Retrieved info for {town_person_lower}: {retrieved_info}")
                    return {
                        "response": response,
                        "retrieved_info": retrieved_info,
                        "category": turn["category"]
                    }
                    
                except Exception as e:
                    print(f"Error in interactive mode: {str(e)}")
                    traceback.print_exc()
                    return {"error": f"Error generating response: {str(e)}"}
            
        elif mode == "auto":
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

    