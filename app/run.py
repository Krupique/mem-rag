import os
import argparse
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
        loader = PyPDFLoader('data/article.pdf')
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

    
    def _set_prompt_template(self, question):
        # Extract the context of the question (i.e. perform vector search)
        context = self.vectordb.max_marginal_relevance_search(question, k = 2, fetch_k = 3)

        self.prompt_template = f"""
            You are an expert assistant. You use the context provided as your complementary knowledge base to answer the question.
            context = {context}
            question = {question}
            answer =
            """
        # Create the list of system and user messages
        messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are an expert assistant."},
        {"role": "user", "content": self.prompt_template}
        ]
        return messages
    

    def _set_prompt_tokenization(self, messages):
        # Apply the chat template
        text = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        # Apply tokenization
        model_inputs = self.tokenizer([text], return_tensors = "pt").to(self.model.device)

        return model_inputs


    def generate_answer(self, question):
        msg = self._set_prompt_template(question=question)
        model_inputs = self._set_prompt_tokenization(messages=msg)
        # Generate response with LLM
        generated_ids = self.model.generate(**model_inputs, max_new_tokens = 512)
        # Unpack the answers
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        # Apply the decode to get the generated text
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0]

        return response


    def run(self):
        # Set the name of the LLM as it appears in the HF
        llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self._load_data()
        self._load_vectordb()
        self._load_model(llm_model_name)
        self._load_tokenizer(llm_model_name)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    parser = argparse.ArgumentParser(description="Set environment variables using argparse.")

    parser.add_argument("--question", type=str, default="True", help="How are you?")
    args = parser.parse_args()


    model = ModelQWEN()
    model.run()

    response = model.generate_answer(args.question)

    print(response)
    
    