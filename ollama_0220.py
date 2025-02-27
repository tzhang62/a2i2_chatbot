import ollama
from GeneratorModel import GeneratorModel
import argparse
import torch
import pickle
#from em_retriever import *
import json
# Define the prompt template
PROMPT = """System: The following conversation is between a Fire Department Operator and a TownPerson {name} during a fire emergency. 
Here is an introduction of {name}:
{persona}
Use this example as a guide:
{dialogue}
Now, generating a new conversation between the Operator and {name}. Start with Operator's initial message.
Operator: 
"""



class OllamaModel(GeneratorModel):
    def __init__(self, model_name, retriever_index=None, document_file=None, device=None):
        """
        Initialize the Ollama model with the retriever.
        :param model_name: The name of the Ollama model to use.
        :param retriever: An instance of the retrieval model to fetch relevant documents.
        """
        self.model_name = model_name
        self.device = device

          # Initialize the Ollama model
        self.model = OllamaModel(model_name="llama-2-7b")
        self.model = self.model.to(device)



    def query(self, question):
        """
        Query the Ollama model with retrieved context and a question.
        :param question: The user's question.
        :param top_k: The number of top retrieved documents to include as context.
        :return: Generated answer from the Ollama model.
        """
        
        prompt = PROMPT.format(question=question)

        # Query the Ollama model
        response = ollama.chat(model=self.model_name, messages=[{
            'role': 'user',
            'content': prompt,
        }])

        # Extract and return the answer
        print(response['message']['content'])
        return response['message']['content']
    
    def generate_and_save_answers(self,questions_file, output_file, k=5):
        """
        Generate concise answers for questions and save them to a file.
        :param questions_file: Path to the file containing questions.
        :param output_file: Path to the file to save generated answers.
        :param k: Number of top results to use for context.
        """
        with open(questions_file, "r") as qf, open(output_file, "w") as outfile:
            i = 0
            for line in qf:
                if i > 0 and i < 101:
                    record = json.loads(line.strip())
                    question = record['question']
                    answer = OllamaModel.query(question)
                    #import pdb; pdb.set_trace()
                    print(i,answer)
                    outfile.write(str(i) + "#" + answer + "\n")
                i+=1
        print(f"Generated answers saved to {output_file}")




# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A generator that uses language models to answer questions.")
    #parser.add_argument("-m", "--modelname", required=True, help="Type of generator model to use (gemma2:2b or gpt).")
    parser.add_argument("-persona", "--personafile", required=True, help="File containing persona")
    parser.add_argument("-dialogue", "--dialoguefile", required=True)
    parser.add_argument("-answer", "--answersfile", required=False, help="File to save generated answers.")
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
    name = 'bob'
    persona = read_json_file(persona_file)[name]
    dialogue = read_json_file(dialogue_file)[name]

    # Define the prompt template
    prompt_1 = """System: The following conversation is between a Fire Department Agent and a TownPerson {name} who needs to be rescued during a fire emergency. 
    Here is an introduction of {name}:
    {persona}
    Use this example as a guide:
    {dialogue}
    Now, generating a new conversation between the Agent and {name}. Start with Agent's initial message.
    """

    prompt_2 = """System: The following conversation is between a Fire Department Agent and a TownPerson {name} during a fire emergency. 
    Here is an introduction of {name}:
    {persona}
    Use this example as a guide:
    {dialogue}
    Now, generating a new conversation between the Agent and {name}. {history} Generate response of the last utterance. Only generate one response once. Please do not include explanations and keep it short.
    """
    
    def send_to_ollama(prompt):
        # Query the Ollama model
        response = ollama.chat(model="llama3.2:latest", messages=[{
            'role': 'user',
            'content': prompt,
        }])
        return response['message']['content'].replace('\n','')

    def simulate_dual_role_conversation(persona, name, dialogue):
        conversation_1 = prompt_1.format(persona=persona, name=name, dialogue=dialogue)
        
        agent_initial = send_to_ollama(conversation_1)
        print('agent_initial: ', agent_initial)
        history = ''
        if agent_initial:
            history += agent_initial + "\n"
            #print(agent_initial)
        else:
            print("No response from Agent.")
            return
        # Simulate a back-and-forth conversation for several turns.
        for turn in range(3):  # This loop simulates three full exchanges.
            # Append prompt for TownPerson's turn.
           
            conversation_2 = prompt_2.format(persona=persona, name=name, dialogue=dialogue, history = history)
            town_reply = send_to_ollama(conversation_2)
            if town_reply:
                import pdb; pdb.set_trace()
                if 'bob:' not in town_reply:
                    town_reply = 'bob: ' + town_reply
                history += town_reply + "\n"
                print('town_reply: ',town_reply)
            else:
                print("No response from TownPerson.")
                break

            # Append prompt for Agent's next response.
           
            conversation_2 = prompt_2.format(persona=persona, name=name, dialogue=dialogue, history = history)
            agent_reply = send_to_ollama(conversation_2)
            if agent_reply:
                import pdb; pdb.set_trace()
                if 'Agent: ' not in agent_reply:
                    agent_reply = 'Agent: ' + agent_reply
                history += agent_reply + "\n"
                print("agent_reply: ",agent_reply)
            else:
                print("No response from Agent.")
                break

        print("\nFinal Conversation Transcript:\n", history)

    simulate_dual_role_conversation(persona,name,dialogue)


    

  
  
    

   
    

  
