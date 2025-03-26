"""
WhartonAI - Lecture Notes Assistant
==================================

A Gradio-based web application that provides two main functionalities:
1. An AI chat interface powered by OpenAI's GPT models
2. A PDF processor that can extract text, generate summaries, and create quiz questions from lecture notes

Requirements:
- Python 3.7+
- Dependencies listed in requirements.txt
- OpenAI API key set in .env file

Author: Wharton TEAM
License: MIT
"""

import gradio as gr
from openai import OpenAI
import os
import time
from pdf_processor import PDFProcessor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def predict(message, history, system_prompt, model, max_tokens, temperature, top_p):
    """
    Handles chat interactions with the OpenAI API.
    
    Args:
        message (str): Current user message
        history (list): Previous conversation history
        system_prompt (str): System prompt to guide AI behavior
        model (str): OpenAI model to use
        max_tokens (int): Maximum tokens in response
        temperature (float): Response randomness (0-1)
        top_p (float): Nucleus sampling parameter (0-1)
    
    Yields:
        str: Generated response chunks for streaming
    """
    try:
        # Initialize the OpenAI client
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Start with the system prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Add the conversation history
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})

        # Add the current user message
        messages.append({"role": "user", "content": message})

        # Record the start time for performance monitoring
        start_time = time.time()

        # Get streaming response from OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=None,
            stream=True
        )

        # Variables for tracking response timing and chunks
        full_message = ""
        first_chunk_time = None
        last_yield_time = None

        # Process streaming response chunks
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time

                full_message += chunk.choices[0].delta.content
                current_time = time.time()
                chunk_time = current_time - start_time
                print(f"Message received {chunk_time:.2f} seconds after request: {chunk.choices[0].delta.content}")  

                # Yield updates every 0.25 seconds to avoid overwhelming the UI
                if last_yield_time is None or (current_time - last_yield_time >= 0.25):
                    yield full_message
                    last_yield_time = current_time

        # Add timing information to final message
        if full_message:
            total_time = time.time() - start_time
            full_message += f" (First Chunk: {first_chunk_time:.2f}s, Total: {total_time:.2f}s)"
            yield full_message
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        yield f"Error: {str(e)}"

def process_pdf(pdf_file):
    """
    Processes uploaded PDF files to generate summaries and quiz questions.
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        tuple: (summary, questions) where:
            - summary is a string containing the generated summary
            - questions is a JSON object containing quiz questions
    """
    if pdf_file is None:
        return "Please upload a PDF file.", None
        
    processor = PDFProcessor(api_key=os.getenv("OPENAI_API_KEY"))
    summary, questions = processor.process_pdf(pdf_file)
    
    if summary is None and questions is None:
        return "Error: Could not process the PDF. Please check if the file is valid and contains text.", None
    elif summary is None:
        return "Error: Could not generate summary. The questions have been generated.", questions
    elif questions is None:
        return summary, "Error: Could not generate questions. The summary has been generated."
    else:
        return summary, questions

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# WhartonAI - Lecture Notes Assistant")
    
    # Chat Interface Tab
    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=predict,
            type="messages",
            #save_history=True,  # Uncomment to enable chat history persistence
            #editable=True,      # Uncomment to allow editing messages
            additional_inputs=[
                gr.Textbox("You are a helpful AI assistant.", label="System Prompt"),
                gr.Dropdown(["gpt-3.5-turbo", "gpt-3.5-turbo-16k"], value="gpt-3.5-turbo", label="Model"),
                gr.Slider(800, 4000, value=2000, label="Max Token"),
                gr.Slider(0, 1, value=0.7, label="Temperature"),
                gr.Slider(0, 1, value=0.95, label="Top P"),
            ],
            css="footer{display:none !important}"  # Hide Gradio footer
        )
    
    # PDF Processing Tab
    with gr.Tab("PDF Processing"):
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF Lecture Notes")
            process_btn = gr.Button("Process PDF")
        
        with gr.Row():
            summary_output = gr.Textbox(label="Summary", lines=5)
            questions_output = gr.JSON(label="Questions")
        
        process_btn.click(
            fn=process_pdf,
            inputs=[pdf_input],
            outputs=[summary_output, questions_output]
        )

if __name__ == "__main__":
    # Launch the Gradio interface
    demo.launch()