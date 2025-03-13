import ollama
from backend.GeneratorModel import GeneratorModel
import argparse
import torch
import pickle
#from em_retriever import *
import json
# Define the prompt template
PROMPT = """You are a helpful assistant. You will be provided the character's persona and previous conversation (context), assume you are the character and try to respond to Q, please keep your response short. 
Here are some good examples, please only generate the "answers", don't include names, remember you are not the Q.
1."persona: ross -- a Van Driver. context: ross:  Hello? Q:  OK, sir, we need for you to evacuate right now. Q:  The fire line is right just to the south of you. Q:  You need to leave right now. ross:  We're kind of stranded out here. ross:  Our van's busted up. Q:  OK, if you go out on the street and just keep heading north, Q:  there'll be a drone. ross:  Patience, and some of them can't move too quickly. Q:  OK, well, you got to at least try. Q:  So I'm going to have a drone that's Q:  going to direct you, that'll lead you to the emergency Q:  vehicle. Q:  Drone number five, lead residents Q:  to the emergency vehicle. ross:  We'll stay put if they'll help arise. Q:  Well, you need to keep moving now, sir. ross:  Some of them have walkers. Q:  But if you don't start moving, OK, there's Q:  one coming right at you. Q:  Start ready to get to load. Q:  OK. ", "answers": [" Thanks a lot. Really appreciate it."]
2."persona: bob -- a Stubborn Man. context: bob:  Hello? Q:  Are you okay? bob:  Yeah. Q:  Vans on this way shortly. ", "answers": [" Alright."]
3."persona: michelle -- a Stubborn Couple. context: michelle:  Hello? Q:  Hi. michelle:  Can I help you with something? Q:  Yes, ma'am. Q:  You need to evacuate the premises immediately. Q:  The wildfires spreading fast. michelle:  We've gotten through wildfires before. michelle:  We know how to handle our property. Q:  It's spreading fast and the results may be fatal if you don't evacuate immediately. Q:  Please go with drone 2. michelle:  Would you leave if this was your house? Q:  Yes, I would. ", "answers": [" You're right. We'll head out."]
4. "persona: ross -- a Van Driver. context: Q:  Hello? Hello? You're in a mandatory evacuation area. Can you please evacuate immediately? ross:  Our van's busted up. Okay. I'm stranded out here.  Q:  Okay, you can follow the drone or would you like a van to be sent to you?  ross:  We'll need a ride.  We can't make it out on our own.  Q: Okay, we're gonna send an emergency vehicle right now. Julie, please send emergency vehicle to residents at five where drone five is.  ross:  I'm on it.  ross: Thank goodness. We'll stay put until help arrives. Q:  Please get everyone outside and ready to go. ", "answers": [" Sure. "]
Persona and context:
{question}

Provide a answer:
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
    parser.add_argument("-i", "--questionsfile", required=True, help="File containing questions.")
    parser.add_argument("-o", "--answersfile", required=True, help="File to save generated answers.")
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
    questions_file = args.questionsfile
    output_file = args.answersfile
    i = 0
    with open(questions_file, "r") as qf, open(output_file, "w") as outfile:
            for line in qf:
                record = json.loads(line.strip())
                question = record['question']
                answer = record['answers']
                
                # Fetch document texts and ensure they are unique
            
                prompt = PROMPT.format(question=question)

                # Query the Ollama model
                response = ollama.chat(model="llama3.2:latest", messages=[{
                    'role': 'user',
                    'content': prompt,
                }])

                # Extract and return the answer
                print(i, response['message']['content'].replace('\n',''))
                result_entry = {}
                result_entry['index'] = str(i)
                result_entry['question'] = question
                result_entry['response'] = response['message']['content'].replace('\n','')
                result_entry['answers'] = answer
                json_line = json.dumps(result_entry)
                outfile.write(json_line+"\n")
                i+=1
                

   
    

  
