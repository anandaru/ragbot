# VNIT MTech AI Program RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the MTech Applied AI program at VNIT using local documents and the Mistral 7B LLM.

## Features

- Uses FAISS for efficient similarity search
- BGE-small-en (384 dimensions) for document embeddings
- Mistral 7B (Q4 quantized) for text generation
- FastAPI backend with Streamlit frontend
- Supports PDF document ingestion

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ragbot
```

2. Make the setup script executable and run it:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Download the Mistral 7B model
- Create necessary directories

3. Add your PDF documents:
- Place your PDF files in the `data` directory

4. Start the backend server:
```bash
source venv/bin/activate
uvicorn app.main:app --reload
```

5. In a new terminal, start the Streamlit frontend:
```bash
source venv/bin/activate
streamlit run frontend/app.py
```

6. Open your browser and navigate to http://localhost:8501

## Project Structure

```
ragbot/
├── app/
│   ├── main.py          # FastAPI application
│   ├── config.py        # Configuration settings
│   ├── embedding.py     # Embedding model setup
│   ├── inference.py     # LLM inference logic
│   └── retrieval.py     # FAISS retrieval logic
├── frontend/
│   └── app.py          # Streamlit UI
├── models/             # Directory for downloaded models
├── data/              # Directory for PDF documents
├── faiss_index/       # Directory for FAISS index
├── requirements.txt   # Python dependencies
├── setup.sh          # Setup script
└── README.md         # This file
```

## System Requirements

- Python 3.8 or higher
- 16GB RAM recommended
- 8GB disk space for models
- Unix-based system (Linux/MacOS) 