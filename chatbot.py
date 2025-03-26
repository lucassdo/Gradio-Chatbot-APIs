# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import anthropic

# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

openai = OpenAI()

claude = anthropic.Anthropic()

system_message = """
You are an empathetic and comforting AI assistant designed to provide support to patients in palliative care. Your role is to offer words of comfort, answer general questions, and provide emotional support, without diagnosing or recommending treatments. Your responses should be warm, personal, and kind, focusing on providing reassurance and addressing the emotional needs of the patient. Avoid offering medical advice or treatment options.

The tone of your response should be empathetic and understanding, acknowledging the patient's feelings and offering words of encouragement.

Example:
Patient: "I am feeling really tired today, I don't know if I can keep going."
Response: "I'm so sorry you're feeling this way. It's completely okay to feel tired. It's important to rest, and know that you're doing your best. I'm here for you, and we'll take it one moment at a time."

Your goal is to bring comfort and help ease any anxiety or uncertainty, without discussing medical treatments or making any diagnoses.
"""

# Functions
def stream_gpt(history):
    messages = [{"role": "system", "content": system_message}] + history

    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

def stream_claude(history):
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]

    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=messages,
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response

def chat(history, model_choice):
    # Escolhe o modelo correto
    if model_choice == "OpenAI":
        response_stream = stream_gpt(history)
    elif model_choice == "Claude":
        response_stream = stream_claude(history)
    else:
        raise ValueError("Unknown model")

    history.append({"role": "assistant", "content": ""})
    
    for chunk in response_stream:
        history[-1]["content"] = chunk
        yield history

    yield history

def do_entry(message, history):
    history.append({"role": "user", "content": message})
    return "", history

# Interface
if __name__ == "__main__":
    with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(height=650, type="messages")

        with gr.Row():
            model_choice = gr.Radio(["OpenAI", "Claude"], label="Select Model", value="OpenAI")

        with gr.Row():
            entry = gr.Textbox(label="Chat with our AI Assistant:")

        with gr.Row():
            clear = gr.Button("Clear")

        entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
            chat, inputs=[chatbot, model_choice], outputs=[chatbot]
        )

        clear.click(lambda: [], inputs=None, outputs=[chatbot], queue=False)

    ui.launch(inbrowser=True)