import streamlit as st
import torch
from chatbot import EmpatheticChatbot  # Assuming the original code is in chatbot.py
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page configuration
st.set_page_config(
    page_title="Empathetic Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_chatbot():
    """Initialize the chatbot with proper error handling"""
    try:
        return EmpatheticChatbot(
            model_path='model.pth',
            config_path='config.json'
        )
    except FileNotFoundError as e:
        st.error("Error: Required model files not found. Please ensure all model files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.stop()

# Initialize chatbot
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = initialize_chatbot()

# Custom CSS
st.markdown("""
    <style>
    .user-message {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .emotion-tag {
        font-size: 0.8em;
        color: #666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("Empathetic Chatbot 🤖")
st.markdown("Share your thoughts and feelings, and I'll respond with empathy.")

# Input field
user_input = st.text_input("Type your message here:", key="user_input")

# Send button
if st.button("Send") and user_input:
    # Get chatbot response
    try:
        response = st.session_state.chatbot.get_response(user_input)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "user": user_input,
            "bot": response
        })
        
        # Clear input
        st.session_state.user_input = ""
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.markdown("### Conversation")
    for chat in reversed(st.session_state.chat_history):
        # User message
        st.markdown(f'<div class="user-message">You: {chat["user"]}</div>', 
                   unsafe_allow_html=True)
        
        # Bot response
        emotion = chat["bot"]["emotion"]
        confidence = chat["bot"]["confidence"]
        response = chat["bot"]["response"]
        
        st.markdown(
            f'<div class="bot-message">'
            f'Bot: {response}<br>'
            f'<span class="emotion-tag">Detected emotion: {emotion} '
            f'(Confidence: {confidence:.2f})</span>'
            f'</div>',
            unsafe_allow_html=True
        )

# Clear chat button
if st.session_state.chat_history:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Footer with information
st.markdown("---")
st.markdown("""
    ### About
    This chatbot uses a deep learning model to detect emotions in your messages 
    and respond with empathy. It can recognize various emotional states and 
    provide appropriate responses.
    
    ### Tips
    - Be open about your feelings
    - Write clear and complete sentences
    - The more context you provide, the better the response
""")
