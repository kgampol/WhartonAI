import gradio as gr
from openai import OpenAI
import os
import time

def predict(message, history, system_prompt, model, max_tokens, temperature, top_p):

    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.environ.get("API_TOKEN"),
    )

    # Start with the system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # Add the conversation history
    messages.extend(history if history else [])

    # Add the current user message
    messages.append({"role": "user", "content": message})

    # Record the start time
    start_time = time.time()

    # Streaming response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=None,
        stream=True
    )

    full_message = ""
    first_chunk_time = None
    last_yield_time = None

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time  # Record time for the first chunk

            full_message += chunk.choices[0].delta.content
            current_time = time.time()
            chunk_time = current_time - start_time  # calculate the time delay of the chunk
            print(f"Message received {chunk_time:.2f} seconds after request: {chunk.choices[0].delta.content}")  

            if last_yield_time is None or (current_time - last_yield_time >= 0.25):
                yield full_message
                last_yield_time = current_time

    # Ensure to yield any remaining message that didn't meet the time threshold
    if full_message:
        total_time = time.time() - start_time
        # Append timing information to the response message
        full_message += f" (First Chunk: {first_chunk_time:.2f}s, Total: {total_time:.2f}s)"
        yield full_message

gr.ChatInterface(
    fn=predict,
    type="messages",
    #save_history=True,
    #editable=True,
    additional_inputs=[
        gr.Textbox("You are a helpful AI assistant.", label="System Prompt"),
        gr.Dropdown(["gpt-4o", "gpt-4o-mini"], label="Model"),
        gr.Slider(800, 4000, value=2000, label="Max Token"),
        gr.Slider(0, 1, value=0.7, label="Temperature"),
        gr.Slider(0, 1, value=0.95, label="Top P"),
    ],
    css="footer{display:none !important}"
).launch()