# import streamlit as st
# import requests

# # --- CONFIGURATION ---
# API_URL = "http://localhost:8001/chat"

# st.set_page_config(page_title="Elon Digital Twin", page_icon="🚀")

# # 1. PERSISTENCE: Track user_id and message history across turns
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Initialize user_id as None; the server will generate one on the first request
# if "user_id" not in st.session_state:
#     st.session_state.user_id = None

# # --- UI LAYOUT ---
# st.title("🚀 Elon Musk Digital Twin")
# st.caption("High-signal RAG agent with contextual memory")

# # Sidebar for session info
# with st.sidebar:
#     st.header("Session Management")
#     if st.session_state.user_id:
#         st.info(f"Connected as User: \n`{st.session_state.user_id}`")
#     else:
#         st.warning("New session: ID will be assigned after the first message.")
        
#     if st.button("Clear Conversation"):
#         st.session_state.messages = []
#         st.session_state.user_id = None # Resetting will trigger a new UUID on next chat
#         st.rerun()

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # --- CHAT LOGIC ---
# if prompt := st.chat_input("Ask about first principles, Starship, or X..."):
#     # Display user message immediately
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Call FastAPI backend
#     with st.chat_message("assistant"):
#         with st.spinner("Processing through first principles..."):
#             try:
#                 # 2. STATEFUL REQUEST: Send the existing user_id back to the server
#                 payload = {
#                     "query": prompt,
#                     "user_id": st.session_state.user_id 
#                 }
                
#                 response = requests.post(API_URL, json=payload)
#                 response.raise_for_status()
#                 data = response.json()

#                 # Extract data from response
#                 final_answer = data.get("response", "Error: No response generated.")
#                 refined_q = data.get("refined_query", "No refinement needed.")
#                 server_user_id = data.get("user_id")

#                 # 3. CRITICAL FIX: Save the assistant's message to state BEFORE any rerun
#                 st.session_state.messages.append({"role": "assistant", "content": final_answer})

#                 # 4. SYNC ID: If this was the first message, capture the ID and refresh the sidebar
#                 if not st.session_state.user_id:
#                     st.session_state.user_id = server_user_id
#                     st.rerun() # Safe now because the message is already appended to state

#                 # Display Answer
#                 st.markdown(final_answer)
                
#                 # Metadata / Debug info
#                 with st.expander("View Logic Metadata"):
#                     st.write(f"**Contextualized Query:** {refined_q}")
#                     st.write(f"**Validation Score:** {data.get('score', 0.0)}")
#                     if data.get("logs"):
#                         st.write("**System Logs:**", data.get("logs"))

#             except Exception as e:
#                 st.error(f"Failed to connect to backend: {str(e)}")





import streamlit as st
import requests

# --- CONFIGURATION ---
API_URL = "http://localhost:8001/chat"
INIT_URL = "http://localhost:8001/init"

st.set_page_config(page_title="Elon Digital Twin", page_icon="🚀")

# -------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Persistent HTTP session (keeps cookies)
if "http_session" not in st.session_state:
    st.session_state.http_session = requests.Session()

# -------------------------------------------------
# AUTO-LOAD BOT GREETING ON FIRST LOAD
# -------------------------------------------------
if not st.session_state.initialized:
    try:
        response = st.session_state.http_session.get(INIT_URL)
        response.raise_for_status()
        data = response.json()

        greeting = data.get("response", "Welcome.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": greeting
        })

    except Exception:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Welcome. (Backend not reachable)"
        })

    st.session_state.initialized = True


# -------------------------------------------------
# UI LAYOUT
# -------------------------------------------------
st.title("🚀 Elon Musk Digital Twin")
st.caption("High-signal RAG agent with contextual memory")

with st.sidebar:
    st.header("Session Management")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.initialized = False
        st.session_state.http_session = requests.Session()
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------------------------------
# CHAT INPUT LOGIC
# -------------------------------------------------
if prompt := st.chat_input("Ask about first principles, Starship, or X..."):

    # Display user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing through first principles..."):
            try:
                payload = {
                    "query": prompt
                }

                response = st.session_state.http_session.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()

                final_answer = data.get("response", "Error: No response generated.")

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer
                })

                st.markdown(final_answer)

            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")
