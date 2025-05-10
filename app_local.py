import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# --- UI Setup ---
st.set_page_config(page_title="Personalized AI Assistant", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>💬 Personalized AI Assistant</h1>", unsafe_allow_html=True)

# --- API Key ---
api_key = st.secrets["CHATGROQ_API_KEY"]  # move this to st.secrets or env in production

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "chat_model" not in st.session_state and api_key:
    st.session_state.chat_model = ChatGroq(model="gemma2-9b-it", api_key=api_key)

if "conversation" not in st.session_state and api_key:
    st.session_state.conversation = ConversationChain(
        llm=st.session_state.chat_model,
        memory=st.session_state.memory,
        verbose=False
    )


# --- Display chat history first (top to bottom) ---
with st.container():
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])

# --- Input box fixed at bottom ---
user_input = st.chat_input("Type your message...")

# --- Handle user input and store response ---
if user_input:
    response = st.session_state.conversation.run(input=user_input)
    print(response)  # Debugging line to check the response
    with st.chat_message("user"):
        st.markdown(user_input) 
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"user": user_input, "bot": response})

# --- API Key check ---
if not api_key:
    st.warning("Please enter your ChatGroq API Key to proceed.")
