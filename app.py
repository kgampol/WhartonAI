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
import json
from typing import Dict, List, Optional, Tuple

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

def process_pdf(pdf_file) -> Tuple[Optional[str], Optional[dict]]:
    """
    Process a PDF file and return its text content and generated questions.
    
    Args:
        pdf_file: The uploaded PDF file
        
    Returns:
        Tuple[Optional[str], Optional[dict]]: A tuple containing the extracted text and generated questions
    """
    if pdf_file is None:
        return None, None
    
    processor = PDFProcessor()
    text, success = processor.extract_text(pdf_file.name)
    if not success:
        return None, None
        
    # Generate summary
    summary = processor.generate_summary(text)
    if summary is None:
        return None, None
        
    # Generate questions
    questions = processor.generate_questions(text)
    return summary, questions

def check_answer(question_index: int, selected_answer: int, questions: dict) -> str:
    """
    Check if the selected answer is correct and provide feedback.
    
    Args:
        question_index (int): Index of the current question
        selected_answer (int): Index of the selected answer
        questions (dict): Dictionary containing all questions and answers
        
    Returns:
        str: Feedback message
    """
    if not questions or 'questions' not in questions:
        return "No questions available."
        
    if question_index >= len(questions['questions']):
        return "Invalid question index."
        
    question = questions['questions'][question_index]
    correct_answer = question['correct_answer']
    
    if selected_answer == correct_answer:
        return "Correct! Well done! 🎉"
    else:
        return f"Incorrect. The correct answer was: {question['options'][correct_answer]}"

def create_interface():
    """
    Create the main Gradio interface for the application.
    """
    with gr.Blocks(title="WhartonAI - Lecture Notes Assistant") as interface:
        gr.Markdown("# WhartonAI - Lecture Notes Assistant")
        gr.Markdown("""
        Upload your lecture notes PDF to:
        1. Get an AI-generated summary
        2. Take an interactive quiz based on the content
        """)
        
        # Store questions state at the top level
        questions_state = gr.State(None)
        current_question = gr.State(0)
        
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload PDF")
                process_btn = gr.Button("Process PDF")
                
            with gr.Column():
                summary_output = gr.Textbox(label="Summary", lines=10)
                
        # Quiz interface components
        with gr.Column() as quiz_interface:
            gr.Markdown("## Quiz")
            
            # Display question
            question_text = gr.Markdown(value="", visible=True)
            
            # Radio buttons for answer choices
            answer_choices = gr.Radio(
                choices=[],
                label="Select your answer:",
                interactive=True,
                visible=True
            )
            
            # Submit button
            submit_btn = gr.Button("Submit Answer", visible=True)
            
            # Feedback message
            feedback = gr.Markdown(value="", visible=True)
            
            # Next question button
            next_btn = gr.Button("Next Question", visible=True)
            
            def update_question(index, questions):
                try:
                    print(f"Updating question with index {index} and questions: {questions}")
                    
                    if not questions:
                        print("Questions state is None")
                        return {
                            question_text: gr.update(value="No questions available. Please process a PDF first."),
                            answer_choices: gr.update(choices=[], value=None),
                            feedback: gr.update(value=""),
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True)
                        }
                    
                    if 'questions' not in questions:
                        print("Questions dictionary missing 'questions' key")
                        return {
                            question_text: gr.update(value="Invalid questions format."),
                            answer_choices: gr.update(choices=[], value=None),
                            feedback: gr.update(value=""),
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True)
                        }
                    
                    questions_list = questions['questions']
                    if not questions_list:
                        print("Questions list is empty")
                        return {
                            question_text: gr.update(value="No questions available."),
                            answer_choices: gr.update(choices=[], value=None),
                            feedback: gr.update(value=""),
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True)
                        }
                    
                    if index >= len(questions_list):
                        print("Quiz completed")
                        return {
                            question_text: gr.update(value="Quiz completed!"),
                            answer_choices: gr.update(choices=[], value=None),
                            feedback: gr.update(value=""),
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True)
                        }
                    
                    question = questions_list[index]
                    print(f"Displaying question {index + 1}: {question['question'][:50]}...")
                    
                    return {
                        question_text: gr.update(value=f"Question {index + 1}: {question['question']}"),
                        answer_choices: gr.update(choices=question['options'], value=None),
                        feedback: gr.update(value=""),
                        submit_btn: gr.update(visible=True),
                        next_btn: gr.update(visible=True)
                    }
                except Exception as e:
                    print(f"Error in update_question: {str(e)}")
                    return {
                        question_text: gr.update(value=f"Error displaying question: {str(e)}"),
                        answer_choices: gr.update(choices=[], value=None),
                        feedback: gr.update(value=""),
                        submit_btn: gr.update(visible=True),
                        next_btn: gr.update(visible=True)
                    }
                
            def handle_submit(question_index, selected_answer, questions):
                try:
                    if not questions or 'questions' not in questions:
                        return "No questions available."
                    if selected_answer is None:
                        return "Please select an answer."
                    return check_answer(question_index, selected_answer, questions)
                except Exception as e:
                    print(f"Error in handle_submit: {str(e)}")
                    return f"Error checking answer: {str(e)}"
                
            def handle_next(question_index):
                try:
                    return question_index + 1
                except Exception as e:
                    print(f"Error in handle_next: {str(e)}")
                    return question_index
            
            # Set up event handlers
            current_question.change(
                update_question,
                inputs=[current_question, questions_state],
                outputs=[question_text, answer_choices, feedback, submit_btn, next_btn]
            )
            
            submit_btn.click(
                handle_submit,
                inputs=[current_question, answer_choices, questions_state],
                outputs=[feedback]
            )
            
            next_btn.click(
                handle_next,
                inputs=[current_question],
                outputs=[current_question]
            )
            
        def handle_pdf_processing(pdf_file):
            try:
                print("Processing PDF file...")
                summary, questions = process_pdf(pdf_file)
                
                if summary is None:
                    print("Failed to generate summary")
                    return "Error processing PDF. Please try again.", None, 0
                
                if not questions or 'questions' not in questions or not questions['questions']:
                    print("No questions generated")
                    return summary, None, 0
                
                print(f"Successfully generated {len(questions['questions'])} questions")
                return summary, questions, 0
                
            except Exception as e:
                print(f"Error in handle_pdf_processing: {str(e)}")
                return f"Error processing PDF: {str(e)}", None, 0
            
        # Set up the process button click handler
        process_btn.click(
            handle_pdf_processing,
            inputs=[pdf_input],
            outputs=[
                summary_output,
                questions_state,
                current_question
            ]
        ).then(
            update_question,
            inputs=[current_question, questions_state],
            outputs=[question_text, answer_choices, feedback, submit_btn, next_btn]
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()