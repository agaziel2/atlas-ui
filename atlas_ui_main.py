
import base64
import io
import IPython.display
from PIL import Image
import os

from dotenv import load_dotenv, find_dotenv
import gradio as gr
import requests
from text_generation import Client
requests.adapters.DEFAULT_TIMEOUT = 60

_ = load_dotenv(find_dotenv())  # read local .env file
hf_api_key = os.environ['HF_API_KEY']
llama2_api = os.environ['HF_API_LLAMA2_13B']
headers = {"Authorization": f"Bearer {hf_api_key}"}

def query(payload):
    response = requests.post(llama2_api, headers=headers, json=payload)
    return response.json()

conversation_counter = 0  # Global variable to track conversation state
user_responses = []  # Global variable to store user responses


def respond(message, chat_history, instruction, temperature=0.7):
    global conversation_counter, user_responses
    questions = [
        "What should we work on? 1. Security 2. Risk Management 3. Audit 4. Governance",
        "What can I help with? Options: 1. I want to learn how to do things 2. Suggest a process improvement 3. Suggest a new process",
        "Where do you want me to search? 1. Internal Guidelines and Procedures 2. Industry Best Practice, Both"
    ]
    #  first question
    if conversation_counter < len(questions):
        # Store the user's response to the previous question, if any
        if message:
            user_responses.append(message)
        #  next question
        question = questions[conversation_counter]
        conversation_counter += 1
        chat_history = chat_history + [[question, ""]]
        yield "", chat_history
    else:
        # Store the user's response
        user_responses.append(message)
        # Reset the counter
        conversation_counter = 0
        # Construct the prompt with the user's responses
        user_responses_str = " | ".join(user_responses)
        prompt = f"System:{instruction} | User Responses: {user_responses_str}\nUser: {message}\nAssistant:"
        # Reset user responses for the next interaction
        user_responses = []
        payload = {"inputs": prompt}
        response = query(payload)
        response_text = response.get('generated_text', '')
        
        # Adding the model's response to the chat history
        chat_history = chat_history + [[message, response_text]]
        yield "", chat_history



with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)  # fit the notebook
    msg = gr.Textbox(label="Prompt")
    with gr.Accordion(label="Advanced options", open=False):
        system = gr.Textbox(label="System message", lines=2,
                            value="You are Atlas a senior security professional in the banking industry, with deep knowledge of security risk management and governance.")
        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot])

gr.close_all()
demo.queue().launch(share=True, server_port=16306)
