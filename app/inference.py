from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from app.config import MODEL_NAME, DEVICE
from app.retrieval import retriever  # ✅ Import FAISS retriever

class LLMInference:
    def __init__(self):
        print(f"🔄 Loading LLM model: {MODEL_NAME}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else DEVICE
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✅ LLM model loaded successfully!")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using the LLM model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,  # 🔥 Lowered randomness for accuracy
                top_p=0.9,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

# ✅ Initialize LLM instance globally
llm = LLMInference()

def generate_answer(query: str) -> str:
    """Generate an answer using FAISS-retrieved documents, dynamically selecting context."""

    print(f"🔄 Generating answer for query: {query}")

    # ✅ Retrieve relevant documents using FAISS
    relevant_docs = retriever.get_relevant_documents(query)
    
    if not relevant_docs:
        return "I don't have enough information to answer that question."

    # ✅ Adaptive Context Selection Based on Query Type
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in ["subject", "course", "semester", "credit"]):
        # 🔥 Focus on subjects/courses
        context = "\n\n".join([
            doc.page_content[:1000] for doc in relevant_docs
            if "Semester" in doc.page_content or "Course Name" in doc.page_content
        ])
    elif any(keyword in query_lower for keyword in ["admission", "apply", "procedure", "steps", "registration"]):
        # 🔥 Focus on admission-related details
        context = "\n\n".join([
            doc.page_content[:1000] for doc in relevant_docs
            if "admission" in doc.page_content or "application" in doc.page_content or "registration" in doc.page_content
        ])
    else:
        # 🔥 Use full FAISS results if no specific filter applies
        context = "\n\n".join([doc.page_content[:1000] for doc in relevant_docs])

    if not context.strip():
        return "I don't have enough information in the retrieved documents."

    print(f"📖 Context Passed to LLM:\n{context[:1500]}...")  # Debugging: Print first 1500 chars

    # ✅ Improved Prompt to Ensure Structured Output
    prompt = f"""You are an AI assistant for M.Tech in Applied AI at VNIT.
You **must answer using only the provided context**. Do not generate information outside the context.

**Context:**
{context}

**Question:** {query}

**Instructions:**
- If the query is about **subjects/courses**, list them in an ordered format.
- If the query is about **admission/application steps**, list the steps clearly.
- If the answer is not in the context, say: "I don't have enough information."
- Keep the response **clear and to the point**.

**Answer:**"""

    print("📝 Sending prompt to LLM...")  # Debugging

    # ✅ Lower randomness to prevent hallucinations
    response = llm.generate(prompt, max_tokens=256)
    
    print(f"✅ Response from LLM: {response[:1000]}")  # Limit log size

    return response
