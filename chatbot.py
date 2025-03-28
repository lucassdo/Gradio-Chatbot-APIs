# imports

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from collections import defaultdict
import anthropic

# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
if perplexity_api_key:
    print(f"Perplexity API Key exists and begins {perplexity_api_key[:5]}")
else:
    print("Perplexity API Key not set")

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

web_search_tool = {
    "name": "search_web",
    "description": "Search the web for up-to-date information on general questions, medications, symptoms, or treatments. Use this when unsure of an answer or when a patient asks about medical topics.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The search prompt based on the patient's question.",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": web_search_tool}]

# Functions
def stream_gpt(history):
    messages = [{"role": "system", "content": system_message}] + history

    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        tools=tools,
        stream=True
    )
    
    recovered_pieces = {
        "content": None,
        "role": "assistant",
        "tool_calls": {}
    }
    tool_call = []

    reply = ""
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ""

        if chunk.choices[0].delta.tool_calls:
            tool_call.extend(chunk.choices[0].delta.tool_calls)
            piece = chunk.choices[0].delta.tool_calls[0]
            recovered_pieces["tool_calls"][piece.index] = recovered_pieces["tool_calls"].get(piece.index, {"id": None,  "function": {"arguments":"",  "name": ""},  "type": "function"})
            if piece.id:
                recovered_pieces["tool_calls"][piece.index]["id"] = piece.id
            if piece.function.name:
                recovered_pieces["tool_calls"][piece.index]["function"]["name"] = piece.function.name
            recovered_pieces["tool_calls"][piece.index]["function"]["arguments"] += piece.function.arguments
        else:
            yield reply
    
    if(len(tool_call) > 0):
        recovered_pieces["tool_calls"] = [recovered_pieces["tool_calls"][key] for key in recovered_pieces["tool_calls"]]
        messages.append(recovered_pieces)
        tool_list = tool_list_to_tool_objs(tool_call)
        for tool in tool_list:
            func = get_tool(tool['name'])
            response = func(tool['arguments'])
            tool_response = {
                "role": "tool",
                "content": response.choices[0].message.content,
                "tool_call_id": tool['id']
            }
            messages.append(tool_response)
        response = openai.chat.completions.create(model='gpt-4o-mini', messages=messages, stream=True)
        for chunk in response:
            reply += chunk.choices[0].delta.content or ""
            yield reply

def get_tool(tool_name):
    if tool_name == "search_web":
        return web_search
    else:
        raise ValueError("Unknown tool")

def web_search(message):
    arguments = json.loads(message)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {   
            "role": "user",
            "content": (
                arguments["prompt"]
            ),
        },
    ]

    client = OpenAI(api_key=perplexity_api_key, base_url="https://api.perplexity.ai")

    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )
    return response

def tool_list_to_tool_objs(data):
    result_by_index = defaultdict(lambda: {'id': '', 'arguments': '', 'name': ''})
    
    for item in data:
        if hasattr(item, 'index') and hasattr(item, 'id') and hasattr(item, 'function'):
            index = item.index
            result_by_index[index]['id'] += (item.id or '')
            result_by_index[index]['arguments'] += (item.function.arguments or '')
            result_by_index[index]['name'] += (item.function.name or '')
    
    result_list = []
    for index, data in result_by_index.items():
        result_list.append({'id': data['id'], 'arguments': data['arguments'], 'name': data['name']})
    
    return result_list

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