#!/bin/bash

# Create Python virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models data faiss_index

# Download Mistral 7B model
echo "Downloading Mistral 7B Q4 GGUF model..."
curl -L https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -o models/mistral-7b-instruct-v0.1.Q4_K_M.gguf

echo "Setup completed! You can now:"
echo "1. Add your PDF documents to the 'data' directory"
echo "2. Run the FastAPI backend: uvicorn app.main:app --reload"
echo "3. Run the Streamlit frontend: streamlit run frontend/app.py" 