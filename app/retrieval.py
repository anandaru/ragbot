import faiss
import os
import json
import numpy as np
import pdfplumber
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.embedding import embedding_model
from app.config import FAISS_INDEX_PATH, TOP_K_DOCS, DATA_DIR


class FAISSRetriever:
    def __init__(self):
        self.index = None
        self.doc_store = []
        self.doc_store_path = str(FAISS_INDEX_PATH).replace("docs.index", "doc_store.json")
        # Get the actual embedding dimension from the model
        self.dimension = embedding_model.get_embedding_dimension()

        print(f"🔍 Expected FAISS dimension: {self.dimension}")
        self.load_index()  # ✅ Now safe to call, because load_index is defined

    def load_index(self):
        """Load or create the FAISS index and restore `doc_store`."""
        if os.path.exists(str(FAISS_INDEX_PATH)):
            try:
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                print(f"✅ FAISS index loaded successfully with {self.index.ntotal} vectors!")

                # ✅ Load `doc_store` from JSON
                if os.path.exists(self.doc_store_path):
                    with open(self.doc_store_path, "r", encoding="utf-8") as f:
                        doc_data = json.load(f)
                        self.doc_store = [Document(page_content=d["text"], metadata=d["metadata"]) for d in doc_data]
                    print(f"📂 Restored {len(self.doc_store)} documents from `doc_store.json`")

                if self.index.ntotal == 0 or len(self.doc_store) == 0:
                    print("⚠️ FAISS index is empty or `doc_store` is missing. Rebuilding index...")
                    self.create_new_index()
            except Exception as e:
                print(f"⚠️ Error loading FAISS index: {e}")
                self.create_new_index()
        else:
            print("⚠️ No FAISS index found. Creating a new one.")
            self.create_new_index()

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text and tables from PDFs, handling both cases flexibly."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = "\n".join([" | ".join([cell if cell is not None else "" for cell in row]) for row in table])
                        print(f"📄 Extracted table from {pdf_path}:\n{table_text}\n")
                        text += f"\nTABLE DATA:\n{table_text}\n"

        if not text.strip():
            print(f"⚠️ Warning: No text extracted from {pdf_path}. The document might be scanned or empty.")

        return text

    def load_documents(self) -> List[Document]:
        """Load and process documents from the data directory."""
        documents = []
        pdf_files = [f for f in os.listdir(str(DATA_DIR)) if f.endswith('.pdf')]

        print(f"📂 Found {len(pdf_files)} PDF files in {DATA_DIR}")

        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(str(DATA_DIR), pdf_file)
                text = self.extract_text_from_pdf(pdf_path)

                if text.strip():
                    doc = Document(page_content=text, metadata={"source": pdf_file, "type": "pdf"})
                    documents.append(doc)
                    print(f"✅ Extracted content from {pdf_file} (including tables):\n{text[:500]}...\n")
                else:
                    print(f"⚠️ Warning: {pdf_file} is empty after extraction.")

            except Exception as e:
                print(f"⚠️ Error extracting {pdf_file}: {e}")

        print(f"📄 Loaded {len(documents)} document chunks from {len(pdf_files)} PDFs")
        return documents

    def create_new_index(self):
        """Create a new FAISS index and populate it with real documents."""
        self.index = faiss.IndexFlatL2(self.dimension)

        # Load and process real documents
        documents = self.load_documents()
        if not documents:
            print("⚠️ No documents found in the data directory.")
            return

        # ✅ Ensure `doc_store` is updated before FAISS indexing
        self.doc_store = documents  

        # Debugging: Print first 5 documents
        for i, doc in enumerate(documents[:5]):
            print(f"📄 Document {i} ({doc.metadata.get('source', 'Unknown')}): {doc.page_content[:500]}...")

        # Get embeddings for documents
        print("🔄 Generating embeddings for documents...")
        doc_embeddings = []
        for i, doc in enumerate(documents):
            embedding = embedding_model.get_embeddings(doc.page_content)[0]
            doc_embeddings.append(embedding)

        doc_embeddings = np.array(doc_embeddings).astype("float32")

        # Check if the embeddings match FAISS index dimension
        if doc_embeddings.shape[1] != self.dimension:
            print(f"⚠️ Mismatch: FAISS expects {self.dimension}, but got {doc_embeddings.shape[1]}")
            raise ValueError("Embedding dimension mismatch! Check model settings.")

        # Add to FAISS index
        self.index.add(doc_embeddings)

        # Save FAISS index
        os.makedirs(os.path.dirname(str(FAISS_INDEX_PATH)), exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        # ✅ Save `doc_store` to JSON
        with open(self.doc_store_path, "w", encoding="utf-8") as f:
            json.dump([{"text": doc.page_content, "metadata": doc.metadata} for doc in self.doc_store], f)

        print(f"✅ New FAISS index created and saved with {len(documents)} document chunks")
        print(f"📂 Total documents stored in `self.doc_store`: {len(self.doc_store)}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query using FAISS."""
        query_embedding = embedding_model.get_embeddings(query)[0].astype("float32")
        query_embedding = query_embedding.reshape(1, -1)

        print(f"🔍 Query: {query}")
        print(f"🔎 Searching FAISS for {TOP_K_DOCS} relevant documents...")

        # Ensure FAISS index is not empty before searching
        if self.index is None or self.index.ntotal == 0:
            print("⚠️ FAISS index is empty or not initialized! No documents to retrieve.")
            return []

        # Retrieve top documents
        D, I = self.index.search(query_embedding, TOP_K_DOCS)

        relevant_docs = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self.doc_store) and score < 15:
                doc = self.doc_store[idx]
                doc.metadata["similarity_score"] = float(score)
                relevant_docs.append(doc)
                print(f"📄 Retrieved: {doc.metadata.get('source', 'Unknown')} (Similarity: {score:.2f})")
            else:
                print(f"⚠️ Skipping index {idx} with score {score:.2f} (out of range or irrelevant)")

        if not relevant_docs:
            print("⚠️ No relevant documents found in FAISS!")

        return relevant_docs


# Initialize retriever
retriever = FAISSRetriever()
