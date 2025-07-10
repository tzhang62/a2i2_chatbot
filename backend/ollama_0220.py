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
import sys
from datetime import datetime
import urllib3
import http.client
import warnings

# Disable all HTTP request logging
os.environ['PYTHONWARNINGS'] = 'ignore'
urllib3.disable_warnings()
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('http.client').setLevel(logging.ERROR)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
logging.getLogger('requests.packages.urllib3').setLevel(logging.ERROR)
http.client.HTTPConnection.debuglevel = 0
# Disable all warnings
warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
for module in ['urllib3', 'requests', 'http.client', 'asyncio', 'websockets']:
    logging.getLogger(module).setLevel(logging.ERROR)

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
        self.operator_response_categories = ['greetings', 'progression', 'observations', 'closing', 'emphasize_danger', 'emphasize_value_of_life', 'give_up_persuading']
        self.character_response_categories = ['greetings', 'response_to_operator_greetings', 'progression', 'observations', 'general', 'closing']
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
                        try:
                            if data['character'] == 'operator':
                                # Store operator responses
                                for category in self.operator_response_categories:
                                    if category in data:
                                        if category not in self.operator_responses:
                                            self.operator_responses[category] = []
                                        self.operator_responses[category].extend(data[category])
                        except:
                            import pdb; pdb.set_trace()
                        else:
                            # Store character responses
                            for category in self.character_response_categories:
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
        
    def get_history(self, session_id: str, max_turns: int = 7) -> str:
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
dialogue_file = 'data_for_train/characterlines.jsonl'
if os.path.exists(dialogue_file):
    
    vector_store.add_dialogues(dialogue_file)
   
else:
    logging.error(f"Warning: Dialogue file not found at {dialogue_file}")
    logging.error(f"Current working directory: {os.getcwd()}")


prompt_rag = """System: You are a Fire Department Agent speaking with TownPerson {name} during a fire emergency.

{name}'s background:
{persona}

Relevant conversation examples:
{context}

Current conversation history:
{history}

Based on {name}'s background and the conversation examples, you are the operator to provide an intial greeting for fire rescue.
Format your output as a direct response without any name prefix or additional context."""

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

def simulate_dual_role_conversation(
    persona: str,
    name: str,
    session_id: Optional[str] = None
) -> str:
    """
    Simulate a conversation using LLM while following a specific conversation flow structure.
    """
    if session_id is None:
        session_id = f"{name}_{int(time.time())}"
        
    # Convert name to lowercase for character matching
    character = name.lower()
        
    # Auto mode: generate responses following conversation flow structure
    history = ""
    retrieved_info_list = []
    
    # Initial operator greeting
    greeting_prompt = prompt_rag.format(
        name=name,
        persona=persona,
        context="Example greeting: Hello hi, this is Fire Department dispatcher Tanay. Are you okay?",
        history="",
        speaker="Agent"
    )
    initial_response = clean_response(send_to_ollama(greeting_prompt))
    conversation_manager.add_message(session_id, "Agent", initial_response)
    
    history = f"Agent: {initial_response}\n"
    
    # Add retrieved info for initial greeting
    operator_greetings = vector_store.operator_responses.get('greetings', [])
    retrieved_info_list.append({
        'speaker': 'Agent',
        'category': 'greetings',
        'examples': operator_greetings,
        'context': f"Category: Greetings\nSpeaker: Agent\n\nExample responses:\n" + "\n".join([f"- {greeting}" for greeting in operator_greetings])
    })
    
    # Conversation flow structure with prompts
    if character == "bob":
        conversation_structure = [
            {
                "speaker": name,
            "prompt": """System: You are {name} responding to a Fire Department Agent during an emergency.
            Based on your background: {persona}
            
            Relevant examples:
            {context}
            
            Generate a single-sentence response showing initial resistance to evacuation.
            Keep your response to one brief sentence that reflects your character's background.
            
            Current conversation:
            {history}
            
            Format your output as a direct response without any prefix.""",
            "category": "response_to_operator_greetings"
        },
        {
            "speaker": "Agent",
            "prompt": """System: You are a Fire Department Agent responding to {name}'s reluctance to evacuate.
            Based on these example responses:
            {context}
            
            Generate a single-sentence urgent warning about the fire danger.
            Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
            
            Current conversation:
            {history}
            
            Format your output as a direct response without any prefix.""",
            "category": "emphasize_danger"
        },
        {
            "speaker": name,
            "prompt": """System: You are {name} still showing resistance to evacuation.
            Based on your background: {persona}
            
            Relevant examples:
            {context}
            
            Generate a single-sentence response expressing specific concerns based on your background.
            Keep your response to one brief sentence that reflects your character's background.
            
            Current conversation:
            {history}
            
            Format your output as a direct response without any prefix.""",
            "category": "response_to_operator_greetings"
        },
        {
            "speaker": "Agent",
            "prompt": """System: You are a Fire Department Agent making a final plea about life safety.
            Based on these example responses:
            {context}
            
            Generate a single-sentence response emphasizing life over property.
            Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
            
            Current conversation:
            {history}
            
            Format your output as a direct response without any prefix.""",
            "category": "emphasize_value_of_life"
        },
        {
            "speaker": name,
            "prompt": """System: You are {name} starting to agree to evacuate.
            Based on your background: {persona}
            
            Relevant examples:
            {context}
            
            Generate a single-sentence response showing your agreement to evacuate.
            Keep your response to one brief sentence that reflects your character's background.
            
            Current conversation:
            {history}
            
            Format your output as a direct response without any prefix.""",
            "category": "progression"
        },
        {
            "speaker": "Agent",
            "prompt": """System: You are a Fire Department Agent responding to {name}'s agreement to evacuate.
            Based on these example responses:
            {context}
            
            Generate a single-sentence response about the importance of evacuating.
            Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
            
            Current conversation:
            {history}
            
            Format your output as a direct response without any prefix.""",
            "category": "progression"
        },
        {
            "speaker": name,
            "prompt": """System: You are {name} finally agreeing to evacuate.
            Based on your background: {persona}
            
            Relevant examples:
            {context}
            
            Generate a single-sentence response showing your agreement to evacuate.
            Keep your response a few words.
            
            Current conversation:
            {history}
            
            Format your output as a direct response without any prefix.""",
            "category": "closing"
        }
        ]
        
        
    
    elif character == "niki":
        conversation_structure = [
            {
                "speaker": name,
                "prompt": """System: You are {name} responding to a Fire Department Agent during an emergency.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's greeting.  
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "response_to_operator_greetings"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s response to your greeting.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "emphasize_danger"
            },
            {
                "speaker": name,
                "prompt": """System: You are {name} showing agreement to evacuation.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's message.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "progression"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s agreement to evacuate.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "closing"
            },
            {   
                "speaker": name,
                "prompt": """System: You are {name} finally agreeing to evacuate.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's message.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "closing"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s agreement to evacuate.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "closing"
            }
        ]  
    elif character == "lindsay":
        conversation_structure = [
            {
                "speaker": name,
                "prompt": """System: You are {name} responding to a Fire Department Agent during an emergency.
                Based on your background: {persona}
                
                Relevant examples:
                {context}

                Generate a single-sentence response to the operator's greeting.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "response_to_operator_greetings"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s response to your greeting.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message and emphasize the danger.
                Keep your response to one brief sentence.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "emphasize_danger"
            },
            {
                "speaker": name,
                "prompt": """System: You are {name} showing agreement to evacuation.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's or julie's message.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "progression"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s agreement to evacuate.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}

                Format your output as a direct response without any prefix.""",
                "category": "closing"
            },
            {
                "speaker": name,
                "prompt": """System: You are {name} finally agreeing to evacuate.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's message.   
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "closing"   
            }
        ]  
    elif character == "ross":
        conversation_structure = [
            {
                "speaker": name,
                "prompt": """System: You are {name} responding to a Fire Department Agent during an emergency.
                Based on your background: {persona}

                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's greeting.
                Keep your response to one brief sentence that reflects your character's background. 
                
                Current conversation:   
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "response_to_operator_greetings"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s response to your greeting.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message, mention to send the tWransportation vehicle.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "emphasize_danger"
            },
            {
                "speaker": name,
                "prompt": """System: You are {name} showing agreement to evacuation.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's message.   
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""", 
                "category": "progression"   
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s agreement to evacuate.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message and try to close the conversation.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "closing"
            },
            {
                "speaker": name,
                "prompt": """System: You are {name} finally agreeing to evacuate.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to appreciate the operator's support and close the conversation.   
                Keep your response to one brief sentence.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "closing"
            }
        ]
    else:
        conversation_structure = [
            {
                "speaker": name,
                "prompt": """System: You are {name} responding to a Fire Department Agent during an emergency.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's greeting.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:   
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "response_to_operator_greetings"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s response to your greeting.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "emphasize_danger"
            },
            {
                "speaker": name,
                "prompt": """System: You are {name} showing resistance to evacuation.
                Based on your background: {persona}
                
                Relevant examples:
                {context}

                Generate a single-sentence response to the operator's message.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "resistance"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s resistance to evacuation.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message and emphasize the value of her life.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "emphasize_value_of_life"
            },
            {
                "speaker": name,
                "prompt": """System: You are {name} showing agreement to evacuation.
                Based on your background: {persona} 
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to the operator's message and agree to evacuate.
                Keep your response to one brief sentence that reflects your character's background. 
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "progression"
            },
            {
                "speaker": "Agent",
                "prompt": """System: You are a Fire Department Agent responding to {name}'s agreement to evacuate.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response to previous message and try to close the conversation.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "closing"
            },
            {
                "speaker": name,
                "prompt": """System: You are {name} finally agreeing to evacuate.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response to show agreeing to evacuate and close the conversation.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix.""",
                "category": "closing"
            }
        ]
    
    # Generate responses following the structure
    for turn in conversation_structure:
        # Get relevant examples based on the category
        if turn["speaker"] == name:
            # Get character-specific examples
            responses = vector_store.character_responses[character.lower()].get(turn["category"], [])
            context = f"Category: {turn['category']}\nSpeaker: {name}\n\nExample responses:\n" + "\n".join([f"- {response}" for response in responses])
            retrieved_info_list.append({
                'speaker': name,
                'category': turn["category"],
                'examples': responses,
                'context': context
            })
        else:
            # Get operator examples
            responses = vector_store.operator_responses.get(turn["category"], [])
            context = f"Category: {turn['category']}\nSpeaker: Agent\n\nExample responses:\n" + "\n".join([f"- {response}" for response in responses])
            retrieved_info_list.append({
                'speaker': 'Agent',
                'category': turn["category"],
                'examples': responses,
                'context': context
            })
        
        # Generate response using the original prompt
        prompt = turn["prompt"].format(
            name=name,
            persona=persona,
            context=context,
            history=history
        )
        
        # print(f"\nGenerating response for {turn['speaker']}...")
        response = clean_response(send_to_ollama(prompt))
        # print(f"Generated response: {response}")
        print("-" * 50)
        
        conversation_manager.add_message(session_id, turn["speaker"], response)
        history += f"{turn['speaker']}: {response}\n"
        
    
        
    print("\n=== Final Conversation ===")
    print(history)
    decision_response = decision_making(history, name)
    print(f"Decision: {decision_response}")
    print("\n=== Decision ===")
    if "yes" in decision_response.lower():
        decision = "Evacuate"
        print("Evacuate")
    else:
        decision = "Do not evacuate"
        print("Do not evacuate")

    return history, retrieved_info_list, decision


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


def decision_making(history, name):
    if name =='ross' or name == 'niki':
        num = 1
    else:
        num = 2
    messages = history.split('\n')
    town_person_messages = []
    for message in messages:
        message = message.lower()
        if message.startswith(f'{name}'):
            message = message.split(':')[-1].strip()
            town_person_messages.append(message)
    last_messages = town_person_messages[-num:]
    recent_history = '  '.join(last_messages)
    print(f"recent_history: {recent_history}")
    prompt = f"{name} says:{recent_history}, determine if {name} is leaving/going/being evacuated or not. If {name} is leaving/going/being evacuated, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def emphasize_danger_check(history):
    prompt = f"Based on the previous utterance {history}, determine if this utterance is emphasizing danger. If so, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def emphasize_value_of_life_check(history):
    prompt = f"Based on the previous utterance {history}, determine if this utterance is emphasizing the value of life. If so, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def mentions_fire_check(history):
    prompt = f"Based on the previous utterance {history}, determine if this utterance mentions fire. If so, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def keep_asking_questions_check(history):
    prompt = f"Based on the previous utterance {history}, determine if this utterance is asking about the fire conditions. If so, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def ending_conversation_check(history):
    prompt = f"Based on the previous utterance {history}, determine if the current message is the end of the conversation, for example, if the operator says thanks or goodbye or something similar, it means the conversation is ending. If the conversation is ending, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def ask_about_children_check(history):
    prompt = f"Based on the previous utterance {history}, determine if this utterance asks about children. If so, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def ask_about_parents_check(history):
    prompt = f"Based on the previous utterance {history}, determine if this utterance asks about parents. If so, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def engagement_check(history):
    prompt = f"Based on this utterance {history}, determine if the operator expresses he would like to leave if he is in the situation. If the operator expresses he would like to leave, respond with 'yes', if not, respond with 'no', only respond with 'yes' or 'no'."
    response = clean_response(send_to_ollama(prompt))
    return response

def setup_logging(output_file):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(output_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger





# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A generator that uses language models to answer questions.")
    parser.add_argument("-persona", "--personafile", required=True, help="File containing persona")
    parser.add_argument("-answer", "--answersfile", required=False, help="File to save generated answers.")
    parser.add_argument("-townperson","--townperson", required=True, help="Town person's name")
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
    output_file = args.answersfile

    def read_json_file(filepath):
        with open(filepath, "r") as file:
            data = json.load(file)
        return data
    name = args.townperson
    persona = read_json_file(persona_file)[name]
    if output_file:
        logger = setup_logging(output_file)
        print(f"\n=== Conversation Generation Started at {datetime.now()} ===")
        print(f"Town Person: {name}")
        print(f"Persona: {persona}")
        print("-" * 50)

    # Generate the conversation
    history, retrieved_info, decision = simulate_dual_role_conversation(persona, name)

    if output_file:
        # Save the output to a JSON file
        output_data = {
            "conversation": history,
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
            "town_person": name,
            "persona": persona
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print("\n=== Conversation Generation Completed ===")
        print(f"Output saved to {output_file}")




    

  
  
    

   
    

  
