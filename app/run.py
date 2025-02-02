import os
import pypdf
import chromadb
import urllib3
import accelerate
import sentence_transformers
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelQWEN():

    def __init__(self):
        pass

    def _load_data(self):
        loader = PyPDFLoader('data/ArtigoDSA1.pdf')
        # Load the file PDF
        pages = loader.load()
        
        # Create the chunk text separator
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)

        # Applying the object and extracting the chunks (documents)
        self.docs = splitter.split_documents(pages)

        
    def _load_vectordb(self):
        # Create the vector database
        self.vectordb = Chroma.from_documents(documents = self.docs,
                                              embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"),
                                              persist_directory = "vectordb/chroma/")

    def _load_model(self, llm_model_name):
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name, 
                                                     torch_dtype = "auto", 
                                                     device_map = "auto")

    def _load_tokenizer(self, llm_model_name):
        # Load the tokenizer from the model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)



    def run(self):
        # Set the name of the LLM as it appears in the HF
        llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        self._load_data()
        self._load_vectordb()
        print('Here')
        #self._load_model(llm_model_name)
        #self._load_tokenizer(llm_model_name)
        
        

    def generate(self):
        pass


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    model = ModelQWEN()
    model.run()
    
    