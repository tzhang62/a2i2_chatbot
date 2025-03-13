import ollama
from GeneratorModel import GeneratorModel
import argparse
import torch
import pickle
#from em_retriever import *
import json

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

def send_to_ollama(prompt):
    # Query the Ollama model
    response = ollama.chat(model="llama3.2:latest", messages=[{
        'role': 'user',
        'content': prompt,
    }])
    return response['message']['content'].replace('\n','')

def simulate_dual_role_conversation(persona, name, dialogue, single_turn=False, user_input=None, get_initial_message=False):
    """
    Simulate a conversation between an operator and a town person.
    
    Args:
        persona (str): The persona description of the town person
        name (str): The name of the town person
        dialogue (str): Example dialogue for context
        single_turn (bool): Whether to generate just one response (for interactive mode)
        user_input (str): The user's input message (for interactive mode)
        get_initial_message (bool): Whether to only return the initial operator message
    
    Returns:
        str: Generated conversation or single response
    """
    if get_initial_message:
        # Generate just the initial operator message
        conversation_1 = prompt_1.format(persona=persona, name=name, dialogue=dialogue)
        response = send_to_ollama(conversation_1)
        
        # Clean up the response
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
        
    if single_turn and user_input:
        # For interactive mode, generate a single response
        conversation = prompt_interactive.format(
            persona=persona,
            name=name,
            dialogue=dialogue,
            history=f"{name}: {user_input}"  # Format the user input with the person's name
        )
        response = send_to_ollama(conversation)
        
        # Clean up the response
        response = response.strip()
        # Remove any name prefixes or system messages
        if ":" in response:
            response = response.split(":", 1)[1].strip()
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        if "Agent:" in response:
            response = response.replace("Agent:", "").strip()
        if "Operator:" in response:
            response = response.replace("Operator:", "").strip()
            
        return response
    
    # For auto mode, generate full conversation
    conversation_1 = prompt_1.format(persona=persona, name=name, dialogue=dialogue)
    
    agent_initial = send_to_ollama(conversation_1)
    if "</think>" in agent_initial:
        agent_initial = agent_initial.split("</think>")[1]
    print('operator_initial: ', agent_initial)
    history = ''
    if agent_initial:
        history += agent_initial + "\n"
    else:
        print("No response from Agent.")
        
    # Simulate a back-and-forth conversation for several turns
    for turn in range(3):
        speaker = name
        next_speaker = "Sir"
        conversation_2 = prompt_2.format(
            persona=persona,
            name=name,
            dialogue=dialogue,
            history=history,
            speaker=speaker,
            next_speaker=next_speaker
        )
        town_reply = send_to_ollama(conversation_2)
        if "</think>" in town_reply:
            town_reply = town_reply.split("</think>")[1]
        if town_reply:
            history += town_reply + "\n"
            print('town_reply: ', town_reply)
        else:
            print("No response from TownPerson.")
            break

        speaker = 'Operator'
        next_speaker = name
        conversation_2 = prompt_2.format(
            persona=persona,
            name=name,
            dialogue=dialogue,
            history=history,
            speaker=speaker,
            next_speaker=next_speaker
        )
        agent_reply = send_to_ollama(conversation_2)
        if "</think>" in agent_reply:
            agent_reply = agent_reply.split("</think>")[1]
        if agent_reply:
            history += agent_reply + "\n"
            print("operator_reply: ", agent_reply)
        else:
            print("No response from Agent.")
            break
            
    history = history.replace('Agent:', '').replace('Bob:', '').replace('Operator:', '')
    return history

def simulate_interactive_conversation(persona, name, dialogue, user_input):
    """
    Handle a single turn of conversation in interactive mode.
    
    Args:
        persona (str): The persona description of the town person
        name (str): The name of the town person
        dialogue (str): Example dialogue for context
        user_input (str): The user's input message
    
    Returns:
        str: Generated response from the agent
    """
    # Format the conversation with the user's input
    conversation = prompt_interactive.format(
        persona=persona,
        name=name,
        dialogue=dialogue,
        history=f"{name}: {user_input}"
    )
    
    # Get response from the model
    response = send_to_ollama(conversation)
    
    # Clean up the response
    response = response.strip()
    # Remove any name prefixes or system messages
    if ":" in response:
        response = response.split(":", 1)[1].strip()
    if "</think>" in response:
        response = response.split("</think>")[1].strip()
    if "Agent:" in response:
        response = response.replace("Agent:", "").strip()
    if "Operator:" in response:
        response = response.replace("Operator:", "").strip()
        
    return response

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


    simulate_dual_role_conversation(persona,name,dialogue)


    

  
  
    

   
    

  
