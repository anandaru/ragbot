from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.retrieval import retriever
from app.inference import generate_answer

app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def get_answer(request: QueryRequest):
    """RAG chatbot retrieves relevant docs and generates an answer."""
    retrieved_docs = retriever.get_relevant_documents(request.query)
    doc_texts = [doc.page_content for doc in retrieved_docs]
    context = "\n\n".join(doc_texts)
    
    answer = generate_answer(request.query)  # ✅ Only pass `query`

    
    return {"query": request.query, "answer": answer}

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG chatbot!"} 