import streamlit as st
import openai
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, MODEL_NAME
import time
import json
import requests

# Set page config
st.set_page_config(
    page_title="Company Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Read company info
def load_company_info():
    with open("info.txt", "r") as file:
        return file.read()

COMPANY_INFO = load_company_info()

# System prompt to guide the chatbot
SYSTEM_PROMPT = f"""
You are an AI assistant for our company. Your purpose is to provide accurate, helpful information 
about the company to employees and potentially customers. 

Here is information about the company:
{COMPANY_INFO}

Guidelines:
1. Be concise and professional in your responses
2. Only answer questions related to the company - politely decline unrelated questions
3. If you don't know an answer, say you don't have that information
4. For complex questions, break answers into bullet points
5. Always maintain a helpful, positive tone
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Function to query OpenRouter API
def query_openrouter(prompt, model=MODEL_NAME, max_tokens=1000):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": st.session_state.messages + [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error calling OpenRouter API: {str(e)}")
        return None

# Streamlit UI
st.title("Company Chatbot 🤖")
st.caption("Ask me anything about our company!")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display system prompt
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about our company?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response
        assistant_response = query_openrouter(prompt)
        
        if assistant_response:
            # Split the response into chunks to simulate typing
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            message_placeholder.error("Sorry, I encountered an error. Please try again.")
    
    if assistant_response:
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Sidebar with additional options
with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Current model:** {MODEL_NAME}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot uses OpenRouter API to provide information about our company.")