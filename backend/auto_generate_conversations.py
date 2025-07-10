import json
import os
from pathlib import Path
from ollama_0220 import simulate_interactive_single_turn, conversation_manager, decision_making

# Get base directory from environment variable or use default
BASE_DIR = os.getenv('A2I2_BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure paths relative to base directory
PERSONA_FILE_PATH = os.path.join("data_for_train/persona.json")
DIAL_FILE_PATH = os.path.join("data_for_train/character_lines.jsonl")
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "results/auto_generated_conversations.json")

def load_json_file(file_path):
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}

def load_dialogue_data():
    """Load dialogue data for all characters."""
    dialogue_data = {}
    with open(DIAL_FILE_PATH, 'r') as f:
        for line in f:
            dialogue_data_line = json.loads(line)
            character = dialogue_data_line['character']
            dialogue_data[character] = dialogue_data_line
    return dialogue_data

def get_julie_category(message_count, town_person):
    """Determine Julie's category based on message count and town person."""
    if message_count <= 1:
        return "greetings"
    elif message_count <= 5:
        if town_person in ["bob", "michelle"]:
            return "emphasize_danger"
        else:
            return "progression"
    elif message_count <= 7:
        return "progression"
    elif message_count <= 9:
        return "closing"
    else:
        return "closing"

def get_town_person_category(message_count, town_person):
    """Determine town person's category based on message count and character."""
    if message_count <= 2:
        return "greetings"
    elif message_count <= 5:
        if town_person == "bob":
            return "work_resistance"
        elif town_person == "niki":
            return "observations"
        elif town_person == "lindsay":
            return "observations"
        elif town_person == "ross":
            return "response_to_operator_greetings"
        elif town_person == "michelle":
            return "response_to_operator_greetings"
    else:
        if town_person == "bob":
            return "minimal_engagement"
        elif town_person == "michelle":
            return "refuse_assistance"
        else:
            return "progression"

def generate_conversation(town_person, persona_data, dialogue_data):
    """Generate a conversation between Julie and a town person."""
    # Use a unique session ID with timestamp to avoid conflicts
    import time
    session_id = f"{town_person}_auto_session_{int(time.time())}"
    conversation_history = []
    decision_responses = []
    
    print(f"Generating conversation for {town_person}...")
    
    # Generate up to 10 messages (5 exchanges between Julie and town person)
    for message_count in range(1, 11):
        print(f"Message {message_count}")
        
        # Get conversation history
        history = conversation_manager.get_history(session_id, max_turns=11)
        
        # Determine Julie's category
        julie_category = get_julie_category(message_count, town_person)
        julie_context = dialogue_data.get("julie", {}).get(julie_category, [])
        
        # Fallback to general if category doesn't exist
        if not julie_context:
            julie_category = "general"
            julie_context = dialogue_data.get("julie", {}).get("general", [])
        
        # Create Julie's prompt
        if julie_category == "closing":
            julie_prompt_content = f"This conversation is now ending. Generate ONLY a brief goodbye message to {town_person} that clearly ends the conversation. Choose from these closing lines: {julie_context}. Do not ask any questions or continue the conversation."
        else:
            julie_prompt_content = f"Generate a message to respond to {town_person}. Use or adapt lines from this category: {julie_category}: {julie_context}."
        
        # Configure Julie's turn
        julie_turn = {
            "speaker": "julie",
            "prompt": f"You are roleplaying as Julie, an emergency evacuation virtual assistant.\nPrevious conversation:\n{history}\n{julie_prompt_content}\nKeep your response in one short sentence. Only generate utterances, no system messages.",
            "category": julie_category
        }
        
        try:
            # Generate Julie's response
            julie_response, julie_retrieved_info = simulate_interactive_single_turn(
                "julie",
                "",
                speaker="Julie",
                persona=persona_data.get("julie", "A virtual assistant specializing in emergency evacuations"),
                turn=julie_turn,
                session_id=session_id
            )
            
            # Add Julie's message to history
            # conversation_manager.add_message(session_id, "Julie", julie_response)
            
            # Record Julie's message
            conversation_history.append({
                "speaker": "Julie",
                "message": julie_response,
                "category": julie_category,
                "retrieved_info": julie_retrieved_info
            })
            
            print(f"Julie: {julie_response}")
            
            # Check if conversation should end
            if julie_category == "closing":
                break
            
            # Generate town person's response
            town_person_category = get_town_person_category(message_count, town_person)
            town_person_data = dialogue_data.get(town_person, {})
            context = town_person_data.get(town_person_category, [])
            
            prompt_content = f"Generate a response to Julie's persuasive message. Use or adapt lines from this {town_person_category}: {context}."
            
            town_person_turn = {
                "speaker": town_person,
                "prompt": f"You are roleplaying as {town_person}, \n{town_person}'s background: {persona_data[town_person]}\nPrevious conversation:\n{history}\n{prompt_content}\nJulie just said: {julie_response}\nPlease generate a response based on this message and keep your response natural and brief. Only generate utterances, no system messages.",
                "category": town_person_category
            }
            
            # Generate town person's response
            response, retrieved_info = simulate_interactive_single_turn(
                town_person,
                julie_response,
                speaker="Julie",
                persona=persona_data[town_person],
                turn=town_person_turn,
                session_id=session_id
            )
            
            # Add town person's response to history
            # conversation_manager.add_message(session_id, town_person, response)
            
            # Record town person's message
            conversation_history.append({
                "speaker": town_person,
                "message": response,
                "category": town_person_category,
                "retrieved_info": retrieved_info
            })
            
            print(f"{town_person}: {response}")
            
            # Get decision response if we have enough messages
            updated_history = conversation_manager.get_history(session_id, max_turns=9)
            if updated_history and len(updated_history.split('\n')) >= 3:
                decision_response = decision_making(updated_history, town_person)
                if decision_response:
                    decision_responses.append({
                        "message_count": message_count,
                        "decision": decision_response
                    })
            
        except Exception as e:
            print(f"Error generating message {message_count} for {town_person}: {str(e)}")
            break
    
    return {
        "town_person": town_person,
        "conversation_history": conversation_history,
        "decision_responses": decision_responses,
        "total_messages": len(conversation_history)
    }

def main():
    """Main function to generate conversations for all town people."""
    # Load data
    persona_data = load_json_file(PERSONA_FILE_PATH)
    dialogue_data = load_dialogue_data()
    
    # Town people to generate conversations for
    town_people = ["bob", "niki", "lindsay", "ross", "michelle"]
    
    all_conversations = []
    
    for town_person in town_people:
        print(f"\n{'='*50}")
        print(f"Generating conversation for {town_person}")
        print(f"{'='*50}")
        
        conversation_result = generate_conversation(town_person, persona_data, dialogue_data)
        all_conversations.append(conversation_result)
        
        print(f"Generated {conversation_result['total_messages']} messages for {town_person}")
    
    # Save results to JSON file
    output_data = {
        "generated_at": str(Path(__file__).stat().st_mtime),
        "total_conversations": len(all_conversations),
        "conversations": all_conversations
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Generated {len(all_conversations)} conversations")
    print(f"Results saved to: {OUTPUT_FILE_PATH}")
    print(f"{'='*50}")
    
    # Print summary
    for conv in all_conversations:
        print(f"{conv['town_person']}: {conv['total_messages']} messages, {len(conv['decision_responses'])} decisions")

if __name__ == "__main__":
    main() 