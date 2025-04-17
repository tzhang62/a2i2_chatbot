import ollama
from GeneratorModel import GeneratorModel
import argparse
import torch
import pickle
#from em_retriever import *
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from typing import List, Dict, Optional
import time
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the prompt template
# prompt_1 = """System: The following conversation is between a Fire Department Agent and a TownPerson {name} who needs to be rescued during a fire emergency. 
# Here is an introduction of {name}:
# {persona}
# Use this example as a guide:
# {dialogue}
# Now, generating a new conversation between the Agent and {name}. Start with Agent's initial message. Do not include any additional utterances or explanations.
# """

prompt_1 = """System: The following conversation is between a Fire Department Agent and a TownPerson {name} who needs to be rescued during a fire emergency. 
Here is the persona of {name}:
{persona}
Use this example as a guide:
{dialogue}
Now, generate a single utterance that Agent said to {name} as the start of the conversation.

Format your output as:
[Name]: Content
"""

prompt_2 = """System: The following conversation is between a Fire Department Agent and a TownPerson {name} during a fire emergency. 

Here is an introduction of {name}:
{persona}
Use this example as a guide:
{dialogue}
Based on the conversation history:
{history} 
Generate one response for the next turn. The next utterance must be solely from the {speaker},and it should start with mentioning {next_speaker}.  Do not include any additional utterances or explanations.

Format your output as:
[Name]: Content
"""

# Add a new prompt template for interactive mode
prompt_interactive = """System: You are a Fire Department Agent speaking with TownPerson {name} during a fire emergency. 

Here is {name}'s background:
{persona}

Use this example dialogue as a guide for the tone and style:
{dialogue}

Current conversation:
{history}

You are the Fire Department Agent. Generate a single response to {name}'s message. Be direct, professional, and focused on ensuring their safety.

Format your output as a direct response without any name prefix or additional context."""

class DialogueVectorStore:
    def __init__(self):
        self.character_responses = {}
        self.operator_responses = {}
        self.response_categories = ['greetings', 'response_to_operator_greetings', 'progression', 'observations', 'general', 'closing']
        # Initialize sentence transformer for semantic similarity
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_dialogues(self, file_path):
        """Load character responses from JSONL file."""
        try:
            with open(file_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    # Skip empty lines
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        if data.get('character') == 'operator':
                            # Store operator responses
                            for category in self.response_categories:
                                if category in data:
                                    if category not in self.operator_responses:
                                        self.operator_responses[category] = []
                                    self.operator_responses[category].extend(data[category])
                        else:
                            # Store character responses
                            for category in self.response_categories:
                                if category not in data:
                                    data[category] = []
                            self.character_responses[data['character']] = data
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON at line {line_num}: {str(e)}")
                        continue
                        
            if not self.character_responses:
                logging.warning("No valid character responses were loaded")
            else:
                logging.info(f"Loaded responses for characters: {list(self.character_responses.keys())}")
                logging.info(f"Loaded operator responses for contexts: {list(self.operator_responses.keys())}")
                
        except Exception as e:
            logging.error(f"Error loading dialogues: {str(e)}")
            raise

    def search(self, query: str, character: str = None, k: int = 3) -> List[Dict]:
        """Search for relevant dialogue entries based on keywords."""
        results = []
        
        # Convert query to lowercase for case-insensitive matching
        query = query.lower()
        
        # Determine which category to search based on keywords
        category = None
        if "greeting" in query:
            category = 'greetings'
        elif "evacuation" in query or "leave" in query:
            category = 'progression'
        elif "see" in query or "smoke" in query or "fire" in query:
            category = 'observations'
        elif "goodbye" in query or "bye" in query:
            category = 'closing'
        elif "operator" in query or "response" in query:
            category = 'response_to_operator_greetings'
        else:
            category = 'general'

        # If character is specified, only search that character's responses
        if character and character in self.character_responses:
            responses = self.character_responses[character].get(category, [])
            for response in responses[:k]:
                results.append({
                    'speaker': character,
                    'content': response,
                    'character': character,
                    'context': category
                })
        else:
            # Search all characters
            for char, data in self.character_responses.items():
                responses = data.get(category, [])
                for response in responses[:k]:
                    results.append({
                        'speaker': char,
                        'content': response,
                        'character': char,
                        'context': category
                    })
                    if len(results) >= k:
                        break
                if len(results) >= k:
                    break

        return results[:k]

    def get_response(self, character, category):
        """Get a response for a specific character and category."""
        if character not in self.character_responses:
            raise ValueError(f"Character {character} not found")
        
        if category not in self.response_categories:
            raise ValueError(f"Category {category} not valid")
            
        responses = self.character_responses[character].get(category, [])
        return random.choice(responses) if responses else ""

    def get_character_context(self, character):
        """Get all responses for a character for context."""
        if character not in self.character_responses:
            return ""
        
        context = []
        for category in self.response_categories:
            responses = self.character_responses[character].get(category, [])
            if responses:
                context.extend(responses)
        return " ".join(context)

    def get_operator_response(self, context: str) -> str:
        """Get a response for the operator/agent based on context."""
        if context not in self.operator_responses or not self.operator_responses[context]:
            context = 'general'  # fallback to general responses
            
        responses = self.operator_responses.get(context, [])
        if not responses:
            return "I understand. Please proceed with evacuation for your safety."
            
        return random.choice(responses)


class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        
    def add_message(self, session_id: str, speaker: str, content: str):
        """Add a message to the conversation history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            
        self.conversations[session_id].append({
            'speaker': speaker,
            'content': content,
            'timestamp': time.time()
        })
        
    def get_history(self, session_id: str, max_turns: int = 5) -> str:
        """Get formatted conversation history."""
        if session_id not in self.conversations:
            print(f"No conversation found for session ID: {session_id}")
            return ""
            
        history = self.conversations[session_id][-max_turns:]
        print(f"Found {len(history)} messages for session ID: {session_id}")
        for i, msg in enumerate(history):
            print(f"Message {i+1}: {msg['speaker']}: {msg['content']}")
        
        return "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in history])

# Initialize global instances
vector_store = DialogueVectorStore()
conversation_manager = ConversationManager()

# Load dialogues if the file exists
dialogue_file = '/Users/tzhang/projects/A2I2/data_for_train/characterlines.jsonl'
if os.path.exists(dialogue_file):
    
    vector_store.add_dialogues(dialogue_file)
   
else:
    logging.error(f"Warning: Dialogue file not found at {dialogue_file}")
    logging.error(f"Current working directory: {os.getcwd()}")


# prompt_rag = """System: You are a Fire Department Agent speaking with TownPerson {name} during a fire emergency.

# {name}'s background:
# {persona}

# Relevant conversation examples:
# {context}

# Current conversation history:
# {history}

# Based on {name}'s background and the conversation examples, generate a natural and contextually appropriate {speaker} response.
# Remember to maintain a professional and reassuring tone while addressing the emergency situation.
# For {name}'s responses, make sure to reference and follow the style of the example responses provided.

# Format your output as a direct response without any name prefix or additional context."""

def send_to_ollama(prompt: str) -> str:
    """Query the Ollama model with the given prompt."""
    response = ollama.chat(model="llama3.2:latest", messages=[{
        'role': 'user',
        'content': prompt,
    }])
    return response['message']['content'].strip()

def clean_response(response: str) -> str:
    """Clean up model response by removing prefixes and system messages."""
    response = response.strip()
    if ":" in response:
        response = response.split(":", 1)[1].strip()
    if "</think>" in response:
        response = response.split("</think>")[1].strip()
    if "Agent:" in response:
        response = response.replace("Agent:", "").strip()
    if "Operator:" in response:
        response = response.replace("Operator:", "").strip()
    return response

# def simulate_dual_role_conversation(
#     persona: str,
#     name: str,
#     session_id: Optional[str] = None
# ) -> str:
#     """
#     Simulate a conversation using LLM while following a specific conversation flow structure.
#     """
#     if session_id is None:
#         session_id = f"{name}_{int(time.time())}"
        
#     # Convert name to lowercase for character matching
#     character = name.lower()
        
#     # Auto mode: generate responses following conversation flow structure
#     history = ""
#     retrieved_info_list = []
    
#     # Initial operator greeting
#     greeting_prompt = prompt_rag.format(
#         name=name,
#         persona=persona,
#         context="Example greeting: Hello hi, this is Fire Department dispatcher Tanay. Are you okay?",
#         history="",
#         speaker="Agent"
#     )
#     initial_response = clean_response(send_to_ollama(greeting_prompt))
#     conversation_manager.add_message(session_id, "Agent", initial_response)
#     history = f"Agent: {initial_response}\n"
    
#     # Add retrieved info for initial greeting
#     operator_greetings = vector_store.operator_responses.get('greetings', [])
#     retrieved_info_list.append({
#         'speaker': 'Agent',
#         'category': 'greetings',
#         'examples': operator_greetings,
#         'context': f"Category: Greetings\nSpeaker: Agent\n\nExample responses:\n" + "\n".join([f"- {greeting}" for greeting in operator_greetings])
#     })
    
#     # Conversation flow structure with prompts
#     conversation_structure = [
#         {
#             "speaker": name,
#             "prompt": """System: You are {name} responding to a Fire Department Agent during an emergency.
#             Based on your background: {persona}
            
#             Relevant examples:
#             {context}
            
#             Generate a single-sentence response showing initial resistance to evacuation.
#             Keep your response to one brief sentence that reflects your character's background.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "response_to_operator_greetings"
#         },
#         {
#             "speaker": "Agent",
#             "prompt": """System: You are a Fire Department Agent responding to {name}'s reluctance to evacuate.
#             Based on these example responses:
#             {context}
            
#             Generate a single-sentence urgent warning about the fire danger.
#             Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "progression"
#         },
#         {
#             "speaker": name,
#             "prompt": """System: You are {name} still showing resistance to evacuation.
#             Based on your background: {persona}
            
#             Relevant examples:
#             {context}
            
#             Generate a single-sentence response expressing specific concerns based on your background.
#             Keep your response to one brief sentence that reflects your character's background.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "response_to_operator_greetings"
#         },
#         {
#             "speaker": "Agent",
#             "prompt": """System: You are a Fire Department Agent making a final plea about life safety.
#             Based on these example responses:
#             {context}
            
#             Generate a single-sentence response emphasizing life over property.
#             Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "progression"
#         },
#         {
#             "speaker": name,
#             "prompt": """System: You are {name} starting to agree to evacuate.
#             Based on your background: {persona}
            
#             Relevant examples:
#             {context}
            
#             Generate a single-sentence response showing your agreement to evacuate.
#             Keep your response to one brief sentence that reflects your character's background.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "progression"
#         },
#         {
#             "speaker": "Agent",
#             "prompt": """System: You are a Fire Department Agent responding to {name}'s agreement to evacuate.
#             Based on these example responses:
#             {context}
            
#             Generate a single-sentence response about the importance of evacuating.
#             Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "progression"
#         },
#         {
#             "speaker": name,
#             "prompt": """System: You are {name} finally agreeing to evacuate.
#             Based on your background: {persona}
            
#             Relevant examples:
#             {context}
            
#             Generate a single-sentence response showing your agreement to evacuate.
#             Keep your response a few words.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "closing"
#         }
#     ]
    
#     # Generate responses following the structure
#     for turn in conversation_structure:
#         # Get relevant examples based on the category
#         if turn["speaker"] == name:
#             # Get character-specific examples
#             responses = vector_store.character_responses[character.lower()].get(turn["category"], [])
#             context = f"Category: {turn['category']}\nSpeaker: {name}\n\nExample responses:\n" + "\n".join([f"- {response}" for response in responses])
#             retrieved_info_list.append({
#                 'speaker': name,
#                 'category': turn["category"],
#                 'examples': responses,
#                 'context': context
#             })
#         else:
#             # Get operator examples
#             responses = vector_store.operator_responses.get(turn["category"], [])
#             context = f"Category: {turn['category']}\nSpeaker: Agent\n\nExample responses:\n" + "\n".join([f"- {response}" for response in responses])
#             retrieved_info_list.append({
#                 'speaker': 'Agent',
#                 'category': turn["category"],
#                 'examples': responses,
#                 'context': context
#             })
        
#         # Generate response using the original prompt
#         prompt = turn["prompt"].format(
#             name=name,
#             persona=persona,
#             context=context,
#             history=history
#         )
        
#         # print(f"\nGenerating response for {turn['speaker']}...")
#         response = clean_response(send_to_ollama(prompt))
#         # print(f"Generated response: {response}")
#         print("-" * 50)
        
#         conversation_manager.add_message(session_id, turn["speaker"], response)
#         history += f"{turn['speaker']}: {response}\n"
        
#         # Add the response to conversation history
#         response_speaker = name  # In interactive mode, response always comes from town person
#         conversation_manager.add_message(session_id, response_speaker, response)
#         # print(f'Added response to history: {response_speaker}: {response}')
    
#     print("\n=== Final Conversation ===")
#     print(history)
#     return history, retrieved_info_list

# name = 'bob'
# conversation_structure = [
#         {
#             "speaker": name,
#             "prompt": """System: You are {name} responding to a Fire Department Agent during an emergency.
#             Based on your background: {persona}
            
#             Relevant examples:
#             {context}
            
#             Generate a single-sentence response showing initial resistance to evacuation.
#             Keep your response to one brief sentence that reflects your character's background.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "response_to_operator_greetings"
#         },
#         {
#             "speaker": name,
#             "prompt": """System: You are {name} still showing resistance to evacuation.
#             Based on your background: {persona}
            
#             Relevant examples:
#             {context}
            
#             Generate a single-sentence response expressing specific concerns based on your background.
#             Keep your response to one brief sentence that reflects your character's background.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "response_to_operator_greetings"
#         },
#         {
#             "speaker": name,
#             "prompt": """System: You are {name} starting to agree to evacuate.
#             Based on your background: {persona}
            
#             Relevant examples:
#             {context}
            
#             Generate a single-sentence response showing your agreement to evacuate.
#             Keep your response to one brief sentence that reflects your character's background.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "progression"
#         },
#         {
#             "speaker": name,
#             "prompt": """System: You are {name} finally agreeing to evacuate.
#             Based on your background: {persona}
            
#             Relevant examples:
#             {context}
            
#             Generate a single-sentence response showing your agreement to evacuate.
#             Keep your response a few words.
            
#             Current conversation:
#             {history}
            
#             Format your output as a direct response without any prefix.""",
#             "category": "closing"
#         }
#     ]


def simulate_interactive_single_turn(town_person, user_input, speaker, persona, turn, session_id=None):
    """Handle interactive conversation mode."""
    # Convert name to lowercase for character matching
    name = town_person.lower()
    character = town_person.lower()
    
    print(f"simulate_interactive_single_turn called for {town_person} with speaker={speaker}")
    
    # Use provided session_id or create a new one
    if session_id is None:
        session_id = f"{name}_{int(time.time())}"
    
    # Then get the complete history INCLUDING the just-added message
    history = conversation_manager.get_history(session_id)
    print(f'Current history after adding user input: {history}')
   
    # Get responses based on speaker
    if speaker == "Operator" or speaker == "Julie":
        responses = vector_store.operator_responses.get(turn["category"], [])
    else:
        responses = vector_store.character_responses[character.lower()].get(turn["category"], [])
    
    context = f"Category: {turn['category']}\nSpeaker: {name}\n\nExample responses:\n" + "\n".join([f"- {response}" for response in responses])
    #print(f'category: {turn["category"]}, session_id: {session_id}, history lines: {(history.count("\n")+1) if history else 0}')
    prompt = turn["prompt"].format(
            name=name,
            persona=persona,
            context=context,
            history=history
        )

    response = clean_response(send_to_ollama(prompt))

    # Determine the response speaker (opposite of input speaker)
    response_speaker = name  # In interactive mode, response always comes from town person
    
    retrieved_info = {
        "full_prompt": turn["prompt"],
        "speaker": town_person.lower()
    }
    
    print(f"Created retrieved_info dictionary for {town_person} with keys: {retrieved_info.keys()}")
    
    # Add the response to conversation history
    conversation_manager.add_message(session_id, response_speaker, response)
    print(f'Added response to history: {response_speaker}: {response}')
    
    # Get updated history count for debugging
    updated_history = conversation_manager.get_history(session_id)
    print(f'Updated history after adding response: {updated_history}')
    #print(f'Total messages in conversation: {updated_history.count("\n")+1 if updated_history else 0}')

    return response, retrieved_info

# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A generator that uses language models to answer questions.")
    #parser.add_argument("-m", "--modelname", required=True, help="Type of generator model to use (gemma2:2b or gpt).")
    parser.add_argument("-persona", "--personafile", required=True, help="File containing persona")
    parser.add_argument("-dialogue", "--dialoguefile", required=True)
    parser.add_argument("-answer", "--answersfile", required=False, help="File to save generated answers.")
    parser.add_argument("-townperson","--townperson", required=True, help="Town person's name")
    #parser.add_argument("-g", "--groundtruthfile", required=True, help="File containing ground truth answers for evaluation.")
    parser.add_argument("--use-mps", action="store_true", help="Enable MPS (Metal Performance Shaders) backend on macOS.")
    args = parser.parse_args()

    # Configure device
    if args.use_mps and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS backend for inference.")
    else:
        device = "cpu"
        print("Using CPU backend for inference.")


    # Generate and save answers
    print("Generating answers...")
    # Retrieve documents based on the question
    persona_file = args.personafile
    dialogue_file = args.dialoguefile
    output_file = args.answersfile

    def read_json_file(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    name = args.townperson
    persona = read_json_file(persona_file)[name]
    dialogue = read_json_file(dialogue_file)[name]


    simulate_dual_role_conversation(persona,name)


    

  
  
    

   
    

  
