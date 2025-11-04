from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama_0220_openai import simulate_interactive_single_turn, conversation_manager, decision_making, emphasize_danger_check, emphasize_value_of_life_check, ending_conversation_check, mentions_fire_check, keep_asking_questions_check, simulate_dual_role_conversation, ask_about_children_check, ask_about_parents_check, engagement_check
import subprocess
import os
import json
import traceback
from pathlib import Path
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time

app = FastAPI()

# Enable CORS with proper configuration for production
# TODO: After deployment, replace "*" with your actual Netlify URL
# Example: allow_origins=["https://your-app.netlify.app", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # IMPORTANT: Update this with your Netlify URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
# TODO: After deployment, add your actual Render and Netlify domains
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "*.onrender.com",  # Allow all Render.com subdomains
        # Add your specific domains after deployment:
        # "your-app.onrender.com",
        # "your-app.netlify.app"
    ]
)

# Get base directory from environment variable or use default
BASE_DIR = os.getenv('A2I2_BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure paths relative to base directory
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "results/answer_80.jsonl")
PERSONA_FILE_PATH = os.path.join("data_for_train/persona.json")
DIAL_FILE_PATH = os.path.join("data_for_train/character_lines.jsonl")
PYTHON_SCRIPT = os.path.join("backend/ollama_0220.py")

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
mary_data = None
ben_data = None
ana_data = None
tom_data = None
mia_data = None
with open(DIAL_FILE_PATH, 'r') as f:
    for line in f:
        dialogue_data_line = json.loads(line)
        #import pdb; pdb.set_trace()
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
        if dialogue_data_line ['character'] == 'mary':
            mary_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'ben':
            ben_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'ana':
            ana_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'tom':
            tom_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'mia':
            mia_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'operator':
            operator_data = dialogue_data_line
        if dialogue_data_line ['character'] == 'julie':
            julie_data = dialogue_data_line


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
        
        # Debug print to verify parameters
        print(f"REQUEST: town_person={town_person}, mode={mode}, speaker={speaker}, auto_julie={auto_julie}")
        
        # Simulating session tracking on the server side
        session_id = f"{town_person_lower}_session"
        decision_response = None  # Initialize decision_response for all town people

        # If in interactive mode, the session ID is stable across requests
        if mode == "interactive":
            print(f"Interactive mode: session_id = {session_id}")
            
            if auto_julie:
                # Get the conversation history
                history = conversation_manager.get_history(session_id, max_turns=11)
                
                # Count messages to determine conversation stage
                message_count = 0
                if history:
                    message_count = len(history.split('\n'))
                
                print(f"Auto Julie mode: message count = {message_count}")
                
                # Check if conversation has ended due to message count
                conversation_ended = message_count > 10
                
                # If conversation has ended, return early with indication
                if conversation_ended:
                    print("Conversation has ended due to message count exceeding limit")
                    return {
                        "julieResponse": "Thank you for your time. Stay safe!",
                        "response": "Goodbye, thank you for your help.",
                        "conversation_ended": True,
                        "message": "Conversation has ended."
                    }
                
                # Determine which category to use for Julie based on conversation stage
                if message_count <= 1:
                    julie_category = "greetings"
                elif message_count <= 5:
                    if town_person_lower == "bob" or town_person_lower == "michelle":
                        julie_category = "emphasize_danger"
                    else:
                        julie_category = "progression"
                elif message_count <= 7:
                    julie_category = "progression"
                elif message_count <= 9:
                    julie_category = "closing"
                else:
                    print(f"Conversation has ended due to message count exceeding limit")
                    return {
                        "julieResponse": "Thank you for your time. Stay safe!",
                        "response": "Goodbye, thank you for your help.",
                        "conversation_ended": True,
                        "message": "Conversation has ended."
                    }
                
                print(f"Selected Julie category: {julie_category}")
                
                # Get Julie's dialogue lines for the selected category
                # Get the lines for Julie from operator_data
                julie_context = julie_data.get(julie_category, [])
                
                # If category doesn't exist or is empty, use general as fallback
                if not julie_context:
                    julie_category = "general"
                    julie_context = julie_data.get("general", [])
                    print(f"Fallback to Julie category: {julie_category}")
                
                # # Ensure we have context
                # if not julie_context:
                #     julie_context = ["Hi, I'm Julie. I'm here to help you evacuate safely."]
                #     print("Using default Julie context")
                
                # Tailor the persuasion approach based on town person
                if town_person_lower == "bob":
                    persuasion_focus = "Focus on how his work can be continued later or recovered, but his life cannot be replaced."
                elif town_person_lower == "niki":
                    persuasion_focus = "Explain the fire danger clearly and directly to address her confusion and uncertainty."
                elif town_person_lower == "lindsay":
                    persuasion_focus = "Emphasize the safety of her family and children, offer specific help with evacuating them."
                elif town_person_lower == "ross":
                    persuasion_focus = "Use logical arguments about the fire's trajectory and timing to appeal to his practical nature."
                elif town_person_lower == "michelle":
                    persuasion_focus = "Be respectful of her independence, provide factual information about the fire rather than giving commands."
                else:
                    persuasion_focus = "Emphasize the imminent danger and the need to evacuate immediately."
                
                # Create the prompt for Julie's response
                if julie_category == "closing":
                    # Make the closing instruction much more explicit when the category is "closing"
                    julie_prompt_content = f"This conversation is now ending. Generate ONLY a brief goodbye message to {town_person} that clearly ends the conversation. Choose from these closing lines: {julie_context}. Do not ask any questions or continue the conversation."
                else:
                    julie_prompt_content = f"Generate a message to respond {town_person}. Use or adapt lines from this category: {julie_category}: {julie_context}."
                
                # Configure Julie's turn
                julie_turn = {
                    "speaker": "julie",
                    "prompt": f"You are roleplaying as Julie, an emergency evacuation virtual assistant.\nPrevious conversation:\n{history}\n{julie_prompt_content}\nKeep your response in one short sentence. Only generate utterances, no system messages.",
                    "category": julie_category
                }
                
                try:
                    print("Generating Julie's response...")
                    # Generate Julie's persuasive message
                    julie_response, julie_retrieved_info = simulate_interactive_single_turn(
                        "julie",
                        "",
                        speaker="Julie",
                        persona=persona_data.get("julie", "A virtual assistant specializing in emergency evacuations"),
                        turn=julie_turn,
                        session_id=session_id
                    )
                    
                    print(f"Julie's response generated: {julie_response}")
                    print(f"Julie's retrieved info: {julie_retrieved_info}")
                    
                    # Add Julie's message to conversation history
                    # conversation_manager.add_message(session_id, "Julie", julie_response)
                    
                    # Prepare Julie's retrieved info with full prompt
                    if isinstance(julie_retrieved_info, dict):
                        julie_retrieved_info["full_prompt"] = julie_turn["prompt"]
                        julie_retrieved_info["speaker"] = "julie"
                    else:
                        julie_retrieved_info = {
                            "full_prompt": julie_turn["prompt"],
                            "speaker": "julie"
                        }
                    
                    # Now generate town person's response to Julie
                    # Create appropriate turn for town person based on their character
                    town_person_category = ""
                    
                    # Select appropriate category for town person's response
                    if town_person_lower == "bob":
                        if message_count <= 2:
                            town_person_category = "greetings"
                        elif message_count <= 5:
                            town_person_category = "work_resistance"
                        else:
                            town_person_category = "minimal_engagement"
                        context = bob_data[town_person_category]
                    elif town_person_lower == "niki":
                        if message_count <= 2:
                            town_person_category = "greetings"
                        elif message_count <= 5:
                            town_person_category = "observations"
                        else:
                            town_person_category = "progression"
                        context = niki_data[town_person_category]
                    elif town_person_lower == "lindsay":
                        if message_count <= 2:
                            town_person_category = "greetings"
                        elif message_count <= 5:
                            town_person_category = "observations"
                        else:
                            town_person_category = "progression"
                        context = lindsay_data[town_person_category]
                    elif town_person_lower == "ross":
                        if message_count <= 2:
                            town_person_category = "greetings"
                        elif message_count <= 5:
                            town_person_category = "response_to_operator_greetings"
                        else:
                            town_person_category = "progression"
                        context = ross_data[town_person_category]
                    elif town_person_lower == "michelle":
                        if message_count <= 2:
                            town_person_category = "greetings"
                        elif message_count <= 5:
                            town_person_category = "response_to_operator_greetings"
                        else:
                            town_person_category = "refuse_assistance"
                        context = michelle_data[town_person_category]
                    
                    print(f"Selected town person category: {town_person_category}")
                    
                    prompt_content = f"Generate a response to Julie's persuasive message. Use or adapt lines from this {town_person_category}: {context}."
                    
                    # Create turn for town person
                    town_person_turn = {
                        "speaker": town_person_lower,
                        "prompt": f"You are roleplaying as {town_person}, \n{town_person}'s background: {persona_data[town_person_lower]}\nPrevious conversation:\n{history}\n{prompt_content}\nJulie just said: {julie_response}\nPlease generate a response based on this message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": town_person_category
                    }
                    
                    print("Generating town person's response...")
                    # Generate town person's response to Julie
                    response, retrieved_info = simulate_interactive_single_turn(
                        town_person_lower,
                        julie_response,  # Using Julie's message as the input
                        speaker="Julie",
                        persona=persona_data[town_person_lower],
                        turn=town_person_turn,
                        session_id=session_id
                    )
                    
                    print(f"Town person's response generated: {response}")
                    
                    # Add town person's response to history
                    # conversation_manager.add_message(session_id, town_person, response)
                    
                    # Prepare retrieved info
                    if isinstance(retrieved_info, dict):
                        retrieved_info["full_prompt"] = town_person_turn["prompt"]
                        retrieved_info["speaker"] = town_person_lower
                    else:
                        retrieved_info = {
                            "full_prompt": town_person_turn["prompt"],
                            "speaker": town_person_lower
                        }
                    
                    # Get decision response if appropriate
                    updated_history = conversation_manager.get_history(session_id, max_turns=11)
                    if updated_history and len(updated_history.split('\n')) >= 0:
                        decision_response = decision_making(updated_history,town_person_lower)
                    
                    print("Returning Auto Julie response")
                    # Return both Julie's message, retrieved info, and town person's response
                    return {
                        "julieResponse": julie_response,
                        "julieRetrievedInfo": julie_retrieved_info,
                        "response": response,
                        "retrieved_info": retrieved_info,
                        "category": town_person_turn["category"],
                        "decision_response": decision_response,
                        "conversation_ended": message_count > 10
                    }
                    
                except Exception as e:
                    print(f"Error in 'Auto Julie' mode: {str(e)}")
                    traceback.print_exc()
                    return {"error": f"Error processing Julie's persuasion: {str(e)}"}
            
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
                
                # Get decision response if we have enough messages
                if history and message_count >= 0:
                    decision_response = decision_making(history,town_person_lower)
                    print(f"Decision response: {decision_response}")
                
                # Special structure for Bob with branching logic
                if town_person_lower == "bob":
                    # Check if the operator is speaking personally
                    is_operator_personal = False
                    emphasizes_danger = False
                    emphasizes_value_of_life = False
                    ending_conversation = False
                    emphasizes_danger_final = False
                    if history and message_count >= 5:
                        i=0
                        for line in history.split('\n'):
                            i+=1
                            if i >=4:
                                ##check if the current line is from operator
                                if "operator" == line.split(':')[0].lower():
                                    if "yes" in emphasize_danger_check(line).lower():
                                        emphasizes_danger_final = True
            
                                        break
                        
                    if history and message_count >= 0:
                        decision_response = decision_making(history,town_person_lower)
                        print(f"Decision response: {decision_response}")
                        
                        last_message = user_input.lower()
                        # Check if message emphasizes fire danger
                        emphasizes_danger_response = emphasize_danger_check(last_message)
                        print(f"emphasize_danger_response: {emphasizes_danger_response}")
                        emphasizes_value_of_life_response = emphasize_value_of_life_check(last_message)
                        print(f"emphasize_value_of_life_response: {emphasizes_value_of_life_response}")
                        # if emphasizes_danger == "yes":
                        #     emphasizes_danger = True
                        if "yes" in emphasizes_danger_response.lower():
                            emphasizes_danger = True
                        if "yes" in emphasizes_value_of_life_response.lower():  
                            emphasizes_value_of_life = True
                        # 
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
                        print(f"emphasizes_danger_response: {emphasizes_danger_response}")
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
                        print(f"emphasizes_value_of_life_response: {emphasizes_value_of_life_response}")
                        print(f"emphasizes_danger_final: {emphasizes_danger_final}")
                        if emphasizes_value_of_life or emphasizes_danger_final:
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
                            category = "closing"
                            context = bob_data[category]
                            prompt_content = f"Generate a response ending the conversation. Choose from this {category}: {context} to show that you're done talking."
                            print(f"Ending conversation")
                        else:
                            category = "closing"
                            context = bob_data[category]
                            prompt_content = f"Generate a response ending the conversation. Please be flexible based on the previous message. Choose from this {category}: {context} "
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
                        last_message = history
                        # Check if message emphasizes fire danger
                        keep_asking_questions_response = keep_asking_questions_check(history)
                        if "yes" in keep_asking_questions_response.lower():
                            keep_asking_questions = True
                        if ending_conversation_check(history):
                            if "yes" in ending_conversation_check(history).lower():
                                ending_conversation = True
                            else:
                                ending_conversation = False
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
                        print(f"mentions_fire_response: {mentions_fire_response}")
                        if mentions_fire:
                            category = "progression"
                            context = niki_data[category]
                            prompt_content = f"Generate a final response acknowledging the danger and agreeing to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context}"
                        elif ending_conversation:
                            category = "closing"
                            context = niki_data[category]
                            prompt_content = f"Generate a final response to the operator or julie that agrees to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context}"
                        elif keep_asking_questions:
                            category = "observation"
                            context = niki_data[category]
                            prompt_content = f"Generate a response to answer the operator's or julie's question. Use or adapt lines from this {category}:{context}. "
                        else:
                            category = "progression"
                            context = niki_data[category]
                        prompt_content = f"Generate a response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 7:
                        if mentions_fire:
                            category = "progression"
                            context = niki_data[category]
                            prompt_content = f"Generate your response acknowledging the danger and agreeing to evacuate. Please be flexible based on the previous message. Choose from this {category}: {context}"
                        elif keep_asking_questions:
                            print(f"keep_asking_questions_response: {keep_asking_questions_response}")
                            category = "observation_2"
                            context = niki_data[category]
                            prompt_content = f"Generate your response to answer the operator's or julie's question. Please be flexible based on the previous message. Choose from this {category}: {context} "
                        elif ending_conversation:
                            category = "closing"
                            context = niki_data[category]
                            prompt_content = f"Generate your response ending the conversation. Please be flexible based on the previous message. Choose from this {category}: {context} "
                        else:
                            category = "progression"
                            context = niki_data[category]
                            prompt_content = f"Generate your final response finally agreeing to evacuate. Choose from this {category}: {context} "

                    else:
                        if ending_conversation:
                            category = "closing"
                            context = niki_data[category]
                            prompt_content = f"Generate your response ending the conversation. Choose from this {category}: {context} "

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
                    mentions_fire_final=False
                    if history and message_count >= 3:
                        for line in history.split('\n'):
                            if "yes" in mentions_fire_check(line).lower():
                                mentions_fire_final = True
                                break
                    if history and message_count > 0 and speaker == "Operator":
                        last_message = user_input.lower()
                        ask_about_children_response = ask_about_children_check(last_message)
                        if "yes" in ask_about_children_response.lower():
                            mentions_children = True
                        ask_about_parents_response = ask_about_parents_check(last_message)
                        if "yes" in ask_about_parents_response.lower():
                            mentions_parents = True
                        if any(keyword in last_message for keyword in ["fine", "alright", "sure", "ok","sounds good","thank",'thanks',"bye","goodbye","see you"]):
                            ending_conversation = True
                        mentions_fire_response = mentions_fire_check(last_message)
                        if "yes" in mentions_fire_response.lower():
                            mentions_fire = True
                            mentions_fire_final = True
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
                            mentions_fire_final = True
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
                        elif mentions_fire or mentions_fire_final:
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
                        elif mentions_fire or mentions_fire_final:
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
                    final_engagement=False
                    engagement=False
                    if history:
                        for line in history.split('\n'):
                            if "yes" in engagement_check(line).lower():
                                final_engagement = True
                                break
                    if history and message_count > 0 and speaker == "Operator":
                        last_message = user_input.lower()
                        engagement_response = engagement_check(last_message)
                        if "yes" in engagement_response.lower():
                            engagement = True
                            
                    ending_conversation = False
                    if ending_conversation_check(history):
                        if "yes" in ending_conversation_check(history).lower():
                            ending_conversation = True
                        else:
                            ending_conversation = False
                    
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = michelle_data[category]
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 3:
                        category = "response_to_operator_greetings"
                        context = michelle_data[category]
                        prompt_content = f"Generate a response to ask the operator if he would like to leave in the situation. Refer to lines from this {category}:{context}."
                    elif message_count == 5:
                        print(f'engagement_response: {engagement_response}')
                        if engagement or final_engagement:
                            category = "progression"
                            context = michelle_data[category]
                            prompt_content = f"Generate a response agreeing to evacuate. Use or adapt lines from this {category}:{context}. "
                        else:
                            category = "refuse_assistance"
                            context = michelle_data[category]
                            prompt_content = f"Generate a response refusing to evacuate. Use or adapt lines from this {category}:{context}. "
                    elif message_count == 7:
                        if ending_conversation or final_engagement:
                            category = "closing"
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
                elif town_person_lower == "mary":
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = ''
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting."
                    elif message_count == 3:
                        category = "response_to_operator_greetings"
                        context = ''
                        prompt_content = f"Generate a response to the operator's greeting or answer the operator's question."
                    elif message_count == 5:
                        category = "progression"
                        context = ''
                        prompt_content = f"Generate a response to the operator."
                    elif message_count == 7:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to determine whether you want to be evacuated or not."
                    else:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to operator or julie."
                    turn = {
                        "speaker": "mary",
                        "prompt": f"You are roleplaying as Mary, \nMary's background: {persona_data['mary']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                elif town_person_lower == "ben":
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = ''
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting."
                    elif message_count == 3:
                        category = "response_to_operator_greetings"
                        context = ''
                        prompt_content = f"Generate a response to the operator's greeting or answer the operator's question."
                    elif message_count == 5:
                        category = "progression"
                        context = ''
                        prompt_content = f"Generate a response to the operator."
                    elif message_count == 7:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to determine whether you want to be evacuated or not."
                    else:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to operator or julie."
                    turn = {
                        "speaker": "ben",
                        "prompt": f"You are roleplaying as Ben, \nBen's background: {persona_data['ben']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                elif town_person_lower == "ana":
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = ''
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting."
                    elif message_count == 3:
                        category = "response_to_operator_greetings"
                        context = ''
                        prompt_content = f"Generate a response to the operator's greeting or answer the operator's question."
                    elif message_count == 5:
                        category = "progression"
                        context = ''
                        prompt_content = f"Generate a response to the operator."
                    elif message_count == 7:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to determine whether you want to be evacuated or not."
                    else:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to operator or julie."
                    turn = {
                        "speaker": "ana",
                        "prompt": f"You are roleplaying as Ana, \nAna's background: {persona_data['ana']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                elif town_person_lower == "tom":
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = ''
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting."
                    elif message_count == 3:
                        category = "response_to_operator_greetings"
                        context = ''
                        prompt_content = f"Generate a response to the operator's greeting or answer the operator's question."
                    elif message_count == 5:
                        category = "progression"
                        context = ''
                        prompt_content = f"Generate a response to the operator."
                    elif message_count == 7:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to determine whether you want to be evacuated or not."
                    else:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to operator or julie."
                    turn = {
                        "speaker": "tom",
                        "prompt": f"You are roleplaying as Tom, \nTom's background: {persona_data['tom']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
                        "category": category
                    }
                elif town_person_lower == "mia":
                    category = ""
                    if message_count == 1:
                        category = "greetings"
                        context = ''
                        prompt_content = f"Generate an initial response to the operator's or julie's greeting."
                    elif message_count == 3:
                        category = "response_to_operator_greetings"
                        context = ''
                        prompt_content = f"Generate a response to the operator's greeting or answer the operator's question."
                    elif message_count == 5:
                        category = "progression"
                        context = ''
                        prompt_content = f"Generate a response to the operator."
                    elif message_count == 7:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to determine whether you want to be evacuated or not."
                    else:
                        category = "closing"
                        context = ''
                        prompt_content = f"Generate a final response to operator or julie."
                    turn = {
                        "speaker": "mia",
                        "prompt": f"You are roleplaying as Mia, \nMia's background: {persona_data['mia']}\nPrevious conversation:\n{history}\n{prompt_content}\n please generate a response based on the last message and keep your response natural and brief. Only generate utterances, no system messages.",
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
                    
                    print(f"Decision response: {decision_response}")    
                    return {
                        "response": response,
                        "retrieved_info": retrieved_info,
                        "category": turn["category"],
                        "decision_response": decision_response
                    }
                    
                except Exception as e:
                    print(f"Error in interactive mode: {str(e)}")
                    traceback.print_exc()
                    return {"error": f"Error generating response: {str(e)}"}
            
        elif mode == "auto":
            try:
                print(f"Starting auto mode generation for {town_person}")
                # Generate the entire conversation at once
                transcript, retrieved_info, decision = simulate_dual_role_conversation(
                    persona_data[town_person_lower],
                    town_person # Keep original case for display
                )
                
                print(f"Generated transcript: {transcript}")
                print(f"Retrieved info: {retrieved_info}")
                print(f"Decision: {decision}")
                
                return {
                    "transcript": transcript,
                    "retrieved_info": retrieved_info,
                    "is_complete": True,
                    "decision": decision}
                 
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

    