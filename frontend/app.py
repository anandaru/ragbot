import streamlit as st
import requests
import json

# Configure the page
st.set_page_config(
    page_title="VNIT MTech AI Program Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add header
st.title("VNIT MTech AI Program Assistant 🎓")
st.markdown("""
This chatbot can help answer your questions about the MTech Applied AI program at VNIT 
(Visweswaraiah National Institute of Technology).
""")

# Chat interface
with st.container():
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question about the MTech AI program"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    "http://localhost:8001/query",
                    json={"query": prompt}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
                else:
                    error_msg = "Sorry, I encountered an error while processing your request."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.markdown(error_msg)

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses RAG (Retrieval Augmented Generation) to provide accurate 
    information about the MTech Applied AI program at VNIT. It references official 
    program documents to answer your queries.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun() 