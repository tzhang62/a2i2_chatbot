from abc import ABCMeta, abstractmethod
import pickle


PROMPT = """
<you should define this prompt>
"""

class GeneratorModel(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    @abstractmethod
    def load_model(self):
        # this method will be necessary if you're using Huggingface `generate`,
        # but it is not necessary for Ollama.
        pass 

    @abstractmethod
    def query(self, retrieved_documents, question):
        pass