# Retrieval-Augmented Generation (RAG) Chatbot for MTech Applied AI at VNIT

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the **MTech Applied AI program at VNIT** using local documents and the **DeepSeek 1.3B** model.

## **✨ Features**

- **FAISS for Efficient Retrieval**: Uses FAISS for fast similarity search.
- **Sentence-Transformers for Embeddings**: `all-MiniLM-L6-v2` model (384 dimensions) for vector embeddings.
- **DeepSeek 1.3B (Quantized) for Text Generation**: Improved LLM response quality.
- **FastAPI Backend & Streamlit Frontend**: Seamless API integration and user-friendly UI.
- **Supports PDF Document Ingestion**: Parses PDFs and indexes them for retrieval.

---

## **🛠 Setup Instructions**

### **1️⃣ Clone the Repository**

```sh
git clone <repository-url>
cd ragbot
```

### **2️⃣ Make the Setup Script Executable & Run**

```sh
chmod +x setup.sh
./setup.sh
```

**This will:** ✅ Create a Python virtual environment ✅ Install all dependencies ✅ Download the DeepSeek 1.3B model ✅ Create necessary directories

### **3️⃣ Add Your PDF Documents**

Place your PDF files in the `data/` directory:

```sh
mkdir -p data
mv /path/to/your/pdfs/*.pdf data/
```

### **4️⃣ Start the Backend Server**

```sh
source venv/bin/activate
uvicorn app.main:app --reload
```

### **5️⃣ Start the Streamlit Frontend** (in a new terminal)

```sh
source venv/bin/activate
streamlit run frontend/app.py
```

Now, open your browser and navigate to [**http://localhost:8501**](http://localhost:8501) to use the chatbot.

---

## **📁 Project Structure**

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
├── data/               # Directory for PDF documents
├── faiss_index/        # Directory for FAISS index
├── requirements.txt    # Python dependencies
├── setup.sh           # Setup script
└── README.md          # This file
```

---

## **💻 System Requirements**

- **Python 3.8 or higher**
- **16GB RAM recommended** (For DeepSeek 1.3B model)
- **8GB disk space for models**
- **Unix-based system (Linux/MacOS)** (Windows WSL recommended)

---



