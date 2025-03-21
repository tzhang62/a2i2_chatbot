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

    def simulate_conversation(self, character, current_context):
        """Generate a response based on character and context."""
        if character not in self.character_responses:
            return "Character not found"

        # Determine appropriate response category based on context
        if "greeting" in current_context.lower():
            category = 'greetings'
        elif "evacuation" in current_context.lower():
            category = 'progression'
        elif "see" in current_context.lower() or "smoke" in current_context.lower():
            category = 'observations'
        elif "goodbye" in current_context.lower():
            category = 'closing'
        elif "operator" in current_context.lower():
            category = 'response_to_operator_greetings'
        else:
            category = 'general'

        # Get both character and operator responses
        char_response = self.get_response(character, category)
        op_response = self.get_operator_response(category)
        return char_response, op_response

    def get_most_similar_response(self, generated_response: str, character: str, category: str) -> str:
        """Find the most similar predefined response to the generated one."""
        if character not in self.character_responses:
            return generated_response
            
        candidates = self.character_responses[character].get(category, [])
        if not candidates:
            return generated_response
            
        # Get embeddings
        generated_embedding = self.encoder.encode(generated_response)
        candidate_embeddings = self.encoder.encode(candidates)
        
        # Calculate similarities
        similarities = np.dot(candidate_embeddings, generated_embedding)
        most_similar_idx = np.argmax(similarities)
        
        return candidates[most_similar_idx]

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
            return ""
            
        history = self.conversations[session_id][-max_turns:]
        return "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in history])

# Initialize global instances
vector_store = DialogueVectorStore()
conversation_manager = ConversationManager()

# Load dialogues if the file exists
dialogue_file = '/Users/tzhang/projects/A2I2/data_for_train/characterlines.jsonl'
if os.path.exists(dialogue_file):
    print(f"Loading dialogues from {dialogue_file}")
    vector_store.add_dialogues(dialogue_file)
else:
    print(f"Warning: Dialogue file not found at {dialogue_file}")

def format_context(retrieved_dialogues: List[Dict], character: str = None) -> str:
    """Format retrieved dialogues into context string."""
    context = "Related examples:\n"
    
    # Filter for character-specific examples if character is provided
    relevant_dialogues = [
        d for d in retrieved_dialogues
        if character is None or d.get('character') == character
    ]
    
    for dialogue in relevant_dialogues:
        context += f"{dialogue['speaker']}: {dialogue['content']}\n"
    return context

prompt_rag = """System: You are a Fire Department Agent speaking with TownPerson {name} during a fire emergency.

{name}'s background:
{persona}

Relevant conversation examples:
{context}

Current conversation history:
{history}

Based on {name}'s background and the conversation examples, generate a natural and contextually appropriate {speaker} response.
Remember to maintain a professional and reassuring tone while addressing the emergency situation.

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
            "category": "progression"
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
            "category": "progression"
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
        
        print(f"\nGenerating response for {turn['speaker']}...")
        response = clean_response(send_to_ollama(prompt))
        print(f"Generated response: {response}")
        print("-" * 50)
        
        conversation_manager.add_message(session_id, turn["speaker"], response)
        history += f"{turn['speaker']}: {response}\n"
    
    print("\n=== Final Conversation ===")
    print(history)
    return history, retrieved_info_list

def generate_response(user_input, context, town_person, persona):
    """Generate a response using the LLM."""
    prompt = """System: You are {name} responding to a Fire Department Agent during an emergency.
    Based on your background: {persona}
    
    Relevant examples:
    {context}
    
    Current conversation:
    {history}
    
    Generate a single-sentence response that reflects your character's background and the current situation.
    Keep your response to one brief sentence.
    
    Format your output as a direct response without any prefix.""".format(
        name=town_person,
        persona=persona,
        context=context,
        history=user_input
    )
    return clean_response(send_to_ollama(prompt))

def simulate_interactive_conversation(town_person, user_input, speaker, persona):
    """Simulate an interactive conversation with the user."""
    try:
        # Get character responses for the town person
        character_responses = vector_store.character_responses.get(town_person, {})
        
        # Get operator responses
        operator_responses = vector_store.operator_responses
        
        # Determine the category based on the conversation flow
        if not user_input:  # Initial message
            category = 'greetings'
            responses = operator_responses.get('greetings', [])
            context = f"Category: {category}\nSpeaker: Operator\n\nExample responses:\n" + "\n".join([f"- {response}" for response in responses])
            prompt = prompt_rag.format(
                name=town_person,
                persona=persona,
                context=context,
                history="",
                speaker="Agent"
            )
            response = clean_response(send_to_ollama(prompt))
            return response, context
            
        # For subsequent messages, determine category based on speaker and context
        if speaker == 'Operator':
            # Operator's conversation flow
            if any(word in user_input.lower() for word in ['hello', 'hi', 'hey']):
                category = 'greetings'
                responses = operator_responses.get('greetings', [])
                prompt = """System: You are a Fire Department Agent responding to {name}'s reluctance to evacuate.
                Based on these example responses:
                {context}
                
                Generate a single-sentence urgent warning about the fire danger.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix."""
            elif any(word in user_input.lower() for word in ['evacuate', 'leave', 'go']):
                category = 'progression'
                responses = operator_responses.get('progression', [])
                prompt = """System: You are a Fire Department Agent making a final plea about life safety.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response emphasizing life over property.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix."""
            elif any(word in user_input.lower() for word in ['goodbye', 'bye', 'thanks']):
                category = 'closing'
                responses = operator_responses.get('closing', [])
                prompt = """System: You are a Fire Department Agent responding to {name}'s agreement to evacuate.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response about the importance of evacuating.
                Keep your response to one brief sentence that matches the professional and authoritative tone of the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix."""
            else:
                category = 'general'
                responses = operator_responses.get('general', [])
                prompt = """System: You are a Fire Department Agent speaking with {name} during a fire emergency.
                Based on these example responses:
                {context}
                
                Generate a single-sentence response that maintains a professional and authoritative tone.
                Keep your response to one brief sentence that matches the examples.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix."""
        else:  # Bob's conversation flow
            if any(word in user_input.lower() for word in ['hello', 'hi', 'hey']):
                category = 'response_to_operator_greetings'
                responses = character_responses.get('response_to_operator_greetings', [])
                prompt = """System: You are {name} responding to a Fire Department Agent during an emergency.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response showing initial resistance to evacuation.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix."""
            elif any(word in user_input.lower() for word in ['evacuate', 'leave', 'go']):
                category = 'progression'
                responses = character_responses.get('progression', [])
                prompt = """System: You are {name} still showing resistance to evacuation.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response expressing specific concerns based on your background.
                Keep your response to one brief sentence that reflects your character's background.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix."""
            elif any(word in user_input.lower() for word in ['goodbye', 'bye', 'thanks']):
                category = 'closing'
                responses = character_responses.get('closing', [])
                prompt = """System: You are {name} finally agreeing to evacuate.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response showing your agreement to evacuate.
                Keep your response a few words.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix."""
            else:
                category = 'general'
                responses = character_responses.get('general', [])
                prompt = """System: You are {name} speaking with a Fire Department Agent during an emergency.
                Based on your background: {persona}
                
                Relevant examples:
                {context}
                
                Generate a single-sentence response that reflects your character's background.
                Keep your response to one brief sentence.
                
                Current conversation:
                {history}
                
                Format your output as a direct response without any prefix."""
        
        # Format context for the response
        context = f"Category: {category}\nSpeaker: {speaker}\n\nExample responses:\n" + "\n".join([f"- {response}" for response in responses])
        
        # Generate response using the appropriate prompt
        response = clean_response(send_to_ollama(prompt.format(
            name=town_person,
            persona=persona,
            context=context,
            history=user_input
        )))
        
        return response, context
        
    except Exception as e:
        print(f"Error in interactive conversation: {str(e)}")
        return "I apologize, but I'm having trouble processing your message.", None

# class OllamaModel(GeneratorModel):
    # def __init__(self, model_name, retriever_index=None, document_file=None, device=None):
    #     """
    #     Initialize the Ollama model with the retriever.
    #     :param model_name: The name of the Ollama model to use.
    #     :param retriever: An instance of the retrieval model to fetch relevant documents.
    #     """
    #     self.model_name = model_name
    #     self.device = device

    #       # Initialize the Ollama model
    #     self.model = OllamaModel(model_name="deepseek-r1:7b")
    #     self.model = self.model.to(device)



    # def query(self, question):
    #     """
    #     Query the Ollama model with retrieved context and a question.
    #     :param question: The user's question.
    #     :param top_k: The number of top retrieved documents to include as context.
    #     :return: Generated answer from the Ollama model.
    #     """
        
    #     prompt = PROMPT.format(question=question)

    #     # Query the Ollama model
    #     response = ollama.chat(model=self.model_name, messages=[{
    #         'role': 'user',
    #         'content': prompt,
    #     }])

    #     # Extract and return the answer
    #     print(response['message']['content'])
    #     return response['message']['content']
    
    # def generate_and_save_answers(self,questions_file, output_file, k=5):
    #     """
    #     Generate concise answers for questions and save them to a file.
    #     :param questions_file: Path to the file containing questions.
    #     :param output_file: Path to the file to save generated answers.
    #     :param k: Number of top results to use for context.
    #     """
    #     with open(questions_file, "r") as qf, open(output_file, "w") as outfile:
    #         i = 0
    #         for line in qf:
    #             if i > 0 and i < 101:
    #                 record = json.loads(line.strip())
    #                 question = record['question']
    #                 answer = OllamaModel.query(question)
    #                 #import pdb; pdb.set_trace()
    #                 print(i,answer)
    #                 outfile.write(str(i) + "#" + answer + "\n")
    #             i+=1
    #     print(f"Generated answers saved to {output_file}")




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


    

  
  
    

   
    

  
