# Medical QA Bot

This project implements a Medical QA Bot using LangChain, FAISS, and the Llama model connected via LM Studio. The bot retrieves information from a set of medical documents and answers user queries based on the retrieved information.

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the required packages:
    pip install -r requirements.txt
   

## Requirements

Ensure the following packages are included in your `requirements.txt`:
pypdf
langchain
langchain_community
torch
accelerate
bitsandbytes
transformers
sentence-transformers
faiss-cpu
chainlit
lmstudio
PyPDF2
openai

##Setup

Create Vector Database
Place the PDF file Gale Encyclopedia of Medicine Vol. 4 (N-S).pdf in the data/ directory.

Run the ingest.py script to create a FAISS vector store:
python ingest.py

Run the QA Bot
Ensure LM Studio is running and the Llama model is loaded and available at http://localhost:1234/v1.

Run the model_final.py script to start the QA bot: 
chainlit model_final.py

##Usage
Once the server is running, you can interact with the Medical QA Bot by asking medical-related questions. The bot will retrieve relevant information from the Gale Encyclopedia of Medicine Vol. 4 (N-S).pdf document and respond with the best possible answer.

License
This project is licensed under the MIT License.
