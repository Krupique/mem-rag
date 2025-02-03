# mem-rag Documentation

## Overview
mem-rag is a Retrieval-Augmented Generation (RAG) system that leverages LangChain, ChromaDB, and an open-source LLM (`Qwen/Qwen2.5-1.5B-Instruct`). The system processes PDF documents, extracts relevant content, and generates responses based on the input question. The project uses Poetry for dependency management and Docker for containerization.

### Benefits of Implementing This Strategy

By integrating a Retrieval-Augmented Generation (RAG) approach with state-of-the-art machine learning techniques, `mem-rag` offers several advantages:

- **Enhanced Information Retrieval**: Instead of relying solely on pretrained model knowledge, `mem-rag` enables real-time retrieval of relevant document data, ensuring up-to-date and contextually accurate responses.
- **Improved Accuracy**: By grounding responses in actual document content, the system minimizes hallucinations and provides more factually correct answers.
- **Scalability**: The modular architecture supports scaling to handle larger document corpora and multiple queries efficiently.
- **Cost Efficiency**: Reduces reliance on large-scale fine-tuning of LLMs by dynamically retrieving and integrating external knowledge.
- **Flexibility**: Supports multiple model configurations and vector database optimizations, making it adaptable to different domains and data types.
- **Seamless Deployment**: With Docker integration, the project can be easily deployed across various environments with minimal setup.

## Project Structure
```
mem-rag/
│── app/
│   └── run.py                 # Main script for running the application
│── data/
│   └── article.pdf            # Dataset used for retrieval
│── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter Notebook for EDA
│── .gitignore                 # Git ignore file
│── Dockerfile                 # Docker configuration file
│── poetry.lock                # Poetry dependency lock file
│── pyproject.toml             # Poetry project configuration
│── README.md                  # Project documentation
```

## Installation

### Prerequisites
- Python 3.12+
- Poetry (Dependency Manager)
- Docker (for containerized execution)

### Setup
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd mem-rag
   ```
2. Install dependencies using Poetry:
   ```sh
   poetry install
   ```
3. Ensure the required dataset (`article.pdf`) is placed in the `data/` directory.

## Usage

### Running Locally
Execute the application by running the following command:
```sh
poetry run python app/run.py --question "What is the summary of the document?"
```

### Running with Docker
Build and run the Docker container:
```sh
# Build the image
docker build -t mem-rag .

# Run the container
docker run --rm mem-rag --question "What is the summary of the document?"
```

## How It Works

### 1. Data Processing
- The system loads the dataset (`article.pdf`) using `PyPDFLoader`.
- The document is split into smaller chunks using `RecursiveCharacterTextSplitter` for better retrieval.

### 2. Vector Database
- The extracted document chunks are embedded using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
- These embeddings are stored in a ChromaDB vector database for efficient similarity search.

### 3. Retrieval and Generation
- When a question is asked, the system searches for the most relevant document chunks using `max_marginal_relevance_search`.
- A custom prompt template is created, including the retrieved context.
- The `Qwen/Qwen2.5-1.5B-Instruct` LLM is used to generate an answer based on the prompt.

## Configuration

### Environment Variables
Set the following environment variables if necessary:
```sh
export TOKENIZERS_PARALLELISM=True
export TF_CPP_MIN_LOG_LEVEL=3
```

### Model Configuration
Modify the `llm_model_name` variable in `run.py` to use a different Hugging Face model.

## Future Enhancements
- Support for multiple document formats (e.g., TXT, DOCX).
- Improved chunking strategies for better context understanding.
- Integration with a web-based interface for easier interaction.

## License
This project is open-source and licensed under the MIT License.

