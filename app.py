import streamlit as st
import requests
import uuid

# --- CONFIGURATION ---
API_URL = "http://localhost:8001/chat"

st.set_page_config(page_title="Elon Digital Twin", page_icon="🚀")

# Initialize session state for chat history and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- UI LAYOUT ---
st.title("🚀 Elon Musk Digital Twin")
st.caption("High-signal RAG agent powered by LangGraph & Groq")

# Sidebar for session info
with st.sidebar:
    st.header("System Status")
    st.info(f"Thread ID: {st.session_state.thread_id}")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT LOGIC ---
if prompt := st.chat_input("Ask about first principles, Starship, or X..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Processing through first principles..."):
            try:
                payload = {
                    "query": prompt,
                    "thread_id": st.session_state.thread_id
                }
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()

                final_answer = data.get("response", "Error: No response generated.")
                score = data.get("score", 0.0)
                used_web = data.get("needs_web_search", False)

                # Display Answer
                st.markdown(final_answer)
                
                # Metadata / Debug info in expander
                with st.expander("View Logic Metadata"):
                    st.write(f"**Validation Score:** {score}")
                    st.write(f"**Web Search Fallback:** {used_web}")
                    if data.get("logs"):
                        st.write("**Logs:**", data.get("logs"))

                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                error_msg = f"Failed to connect to backend: {str(e)}"
                st.error(error_msg)