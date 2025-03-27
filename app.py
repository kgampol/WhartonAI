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

def check_answer(question_index: int, selected_answer: str, questions: dict) -> str:
    """
    Check if the selected answer is correct for the given question.
    
    Args:
        question_index (int): Index of the current question
        selected_answer (str): The answer selected by the user
        questions (dict): Dictionary containing all questions and answers
        
    Returns:
        str: Feedback message indicating if the answer was correct
    """
    try:
        if not questions or 'questions' not in questions:
            return "Error: No questions available."
            
        if question_index >= len(questions['questions']):
            return "Error: Question index out of range."
            
        question = questions['questions'][question_index]
        correct_answer_index = question['correct_answer']
        options = question['options']
        
        # Find the index of the selected answer in the options list
        selected_index = options.index(selected_answer)
        
        # Compare the indices
        if selected_index == correct_answer_index:
            return "Correct! Well done! 🎉"
        else:
            correct_answer = options[correct_answer_index]
            return f"Incorrect. The correct answer was: {correct_answer}"
            
    except Exception as e:
        print(f"Error checking answer: {str(e)}")
        return f"Error checking answer: {str(e)}"

def handle_regenerate(pdf_file):
    try:
        if pdf_file is None:
            return None, 0, 0, 0, gr.update(value="", visible=False)
        
        print("Regenerating questions...")
        # Show loading message
        loading_msg = "Regenerating questions... Please wait..."
        
        # Process the PDF and generate new questions
        summary, questions = process_pdf(pdf_file)
        
        if not questions or 'questions' not in questions or not questions['questions']:
            print("Failed to regenerate questions")
            return None, 0, 0, 0, gr.update(value="Failed to regenerate questions. Please try again.", visible=True)
        
        print(f"Successfully regenerated {len(questions['questions'])} questions")
        return questions, 0, 0, 0, gr.update(value="", visible=False)
        
    except Exception as e:
        print(f"Error regenerating questions: {str(e)}")
        return None, 0, 0, 0, gr.update(value=f"Error regenerating questions: {str(e)}", visible=True)

def handle_pdf_processing(pdf_file):
    try:
        print("Processing PDF file...")
        # First generate and show the summary
        processor = PDFProcessor()
        text, success = processor.extract_text(pdf_file.name)
        if not success:
            return "Error processing PDF. Please try again.", None, 0, 0, 0, None, "Error processing PDF"
        
        summary = processor.generate_summary(text)
        if summary is None:
            return "Error generating summary. Please try again.", None, 0, 0, 0, None, "Error generating summary"
        
        # Show summary immediately and indicate quiz generation
        quiz_status = "Generating quiz questions... Please wait..."
        
        # Return immediately with summary and status
        return summary, None, 0, 0, 0, pdf_file, quiz_status
        
    except Exception as e:
        print(f"Error in handle_pdf_processing: {str(e)}")
        return f"Error processing PDF: {str(e)}", None, 0, 0, 0, None, f"Error: {str(e)}"

def generate_questions_async(pdf_file):
    try:
        if pdf_file is None:
            return None, 0, 0, 0, "Error: No PDF file provided"
        
        processor = PDFProcessor()
        text, success = processor.extract_text(pdf_file.name)
        if not success:
            return None, 0, 0, 0, "Error processing PDF"
        
        # Generate questions
        questions = processor.generate_questions(text)
        if not questions or 'questions' not in questions or not questions['questions']:
            return None, 0, 0, 0, "Error generating questions"
        
        print(f"Successfully generated {len(questions['questions'])} questions")
        return questions, 0, 0, 0, "Quiz ready! You can now start answering questions."
        
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return None, 0, 0, 0, f"Error: {str(e)}"

def create_interface():
    """
    Create the main Gradio interface for the application.
    """
    with gr.Blocks(title="WhartonAI - Lecture Notes Assistant") as interface:
        gr.Markdown("# WhartonAI - Lecture Notes Assistant")
        gr.Markdown("""
        Welcome to WhartonAI Lecture Notes Assistant!
        
        This tool helps you better understand and retain your lecture materials by:
        1. Creating a concise AI-powered summary of your lecture notes
        2. Generating an interactive quiz with detailed explanations
        3. Providing step-by-step solutions to reinforce your learning
        
        To get started, simply upload your lecture notes PDF using the file selector.
        """)
        
        # Store questions state at the top level
        questions_state = gr.State(None)
        current_question = gr.State(0)
        correct_answers = gr.State(0)
        total_questions = gr.State(0)
        current_pdf = gr.State(None)
        
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload PDF")
                process_btn = gr.Button("Process PDF")
                
            with gr.Column():
                summary_output = gr.Textbox(label="Summary", lines=10)
                quiz_status = gr.Markdown(value="", visible=False)
                
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
            
            # Generate solution button (initially hidden)
            generate_solution_btn = gr.Button("Generate Step-by-Step Solution", visible=False)
            
            # Detailed solution with citations
            solution = gr.Markdown(value="", visible=False)
            
            # Next question button
            next_btn = gr.Button("Next Question", visible=True)
            
            # Performance display
            performance_markdown = gr.Markdown(value="Performance: 0/0 (0%)", visible=True)
            
            # Regenerate questions button
            regenerate_btn = gr.Button("Regenerate Questions", visible=False)
            
            # Loading message for regeneration
            loading_msg = gr.Markdown(value="", visible=False)
            
            def update_performance_display(correct, total):
                if total == 0:
                    return "Performance: 0/0 (0%)"
                percentage = (correct / total) * 100
                return f"Performance: {correct}/{total} ({percentage:.1f}%)"
            
            def update_question(index, questions):
                try:
                    print(f"Updating question with index {index} and questions: {questions}")
                    
                    if not questions or 'questions' not in questions:
                        print("Questions state is None")
                        return {
                            question_text: gr.update(value="No questions available. Please process a PDF first."),
                            answer_choices: gr.update(choices=[], value=None, interactive=True),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True),
                            regenerate_btn: gr.update(visible=False),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False)
                        }
                    
                    if 'questions' not in questions:
                        print("Questions dictionary missing 'questions' key")
                        return {
                            question_text: gr.update(value="Invalid questions format."),
                            answer_choices: gr.update(choices=[], value=None, interactive=True),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True),
                            regenerate_btn: gr.update(visible=False),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False)
                        }
                    
                    questions_list = questions['questions']
                    if not questions_list:
                        print("Questions list is empty")
                        return {
                            question_text: gr.update(value="No questions available."),
                            answer_choices: gr.update(choices=[], value=None, interactive=True),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True),
                            regenerate_btn: gr.update(visible=False),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False)
                        }
                    
                    if index >= len(questions_list):
                        print("Quiz completed")
                        return {
                            question_text: gr.update(value="Quiz completed!"),
                            answer_choices: gr.update(choices=[], value=None, interactive=True),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=False),
                            next_btn: gr.update(visible=False),
                            regenerate_btn: gr.update(visible=True),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False)
                        }
                    
                    question = questions_list[index]
                    print(f"Displaying question {index + 1}: {question['question'][:50]}...")
                    
                    return {
                        question_text: gr.update(value=f"Question {index + 1}: {question['question']}"),
                        answer_choices: gr.update(choices=question['options'], value=None, interactive=True),
                        feedback: gr.update(value=""),
                        solution: gr.update(value="", visible=False),
                        submit_btn: gr.update(visible=True),
                        next_btn: gr.update(visible=True),
                        regenerate_btn: gr.update(visible=False),
                        loading_msg: gr.update(value="", visible=False),
                        generate_solution_btn: gr.update(visible=False)
                    }
                except Exception as e:
                    print(f"Error in update_question: {str(e)}")
                    return {
                        question_text: gr.update(value=f"Error displaying question: {str(e)}"),
                        answer_choices: gr.update(choices=[], value=None, interactive=True),
                        feedback: gr.update(value=""),
                        solution: gr.update(value="", visible=False),
                        submit_btn: gr.update(visible=True),
                        next_btn: gr.update(visible=True),
                        regenerate_btn: gr.update(visible=False),
                        loading_msg: gr.update(value="", visible=False),
                        generate_solution_btn: gr.update(visible=False)
                    }
                
            def handle_submit(question_index, selected_answer, questions, correct, total):
                try:
                    if not questions or 'questions' not in questions:
                        return "No questions available.", correct, total, gr.update(interactive=False), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False)
                    if selected_answer is None:
                        return "Please select an answer.", correct, total, gr.update(interactive=True), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=True)
                    
                    question = questions['questions'][question_index]
                    correct_answer_index = question['correct_answer']
                    options = question['options']
                    
                    # Find the index of the selected answer in the options list
                    selected_index = options.index(selected_answer)
                    
                    # Update performance tracking
                    new_total = total + 1
                    new_correct = correct
                    if selected_index == correct_answer_index:
                        new_correct += 1
                        feedback_msg = "Correct! Well done! 🎉"
                        solution = gr.update(value="", visible=False)
                        generate_solution_btn = gr.update(visible=False)
                    else:
                        correct_answer = options[correct_answer_index]
                        feedback_msg = f"Incorrect. The correct answer was: {correct_answer}"
                        solution = gr.update(value="", visible=False)
                        generate_solution_btn = gr.update(visible=True)
                    
                    # Always disable submit button after first submission
                    return feedback_msg, new_correct, new_total, gr.update(interactive=False), solution, generate_solution_btn, gr.update(visible=False)
                    
                except Exception as e:
                    print(f"Error in handle_submit: {str(e)}")
                    return f"Error checking answer: {str(e)}", correct, total, gr.update(interactive=True), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=True)
                
            def handle_next(question_index):
                try:
                    return question_index + 1
                except Exception as e:
                    print(f"Error in handle_next: {str(e)}")
                    return question_index
            
            def generate_solution(question_index, questions, current_pdf):
                try:
                    # Record the start time for performance monitoring
                    start_time = time.time()
                    
                    if not questions or 'questions' not in questions:
                        return gr.update(value="Error: No questions available.", visible=True)
                    
                    # Show initial timing message
                    yield gr.update(value="Generating solution...", visible=True)
                    
                    question = questions['questions'][question_index]
                    processor = PDFProcessor()
                    text, success = processor.extract_text(current_pdf.name)
                    if not success:
                        yield gr.update(value="Error processing PDF for solution.", visible=True)
                        return
                    
                    # Show intermediate message without timing
                    yield gr.update(value="Generating solution steps...", visible=True)
                    
                    solution_text = processor.generate_solution(question, text)
                    if solution_text:
                        # Calculate final time
                        total_time = time.time() - start_time
                        # Show final solution with timing
                        yield gr.update(value=f"{solution_text}\n\n---\nSolution generated in {total_time:.2f} seconds", visible=True)
                    else:
                        yield gr.update(value="Error generating solution. Please try again.", visible=True)
                    
                except Exception as e:
                    print(f"Error generating solution: {str(e)}")
                    yield gr.update(value=f"Error: {str(e)}", visible=True)
            
            # Set up event handlers
            current_question.change(
                update_question,
                inputs=[current_question, questions_state],
                outputs=[question_text, answer_choices, feedback, solution, submit_btn, next_btn, regenerate_btn, loading_msg, generate_solution_btn]
            )
            
            submit_btn.click(
                handle_submit,
                inputs=[current_question, answer_choices, questions_state, correct_answers, total_questions],
                outputs=[feedback, correct_answers, total_questions, answer_choices, solution, generate_solution_btn, submit_btn]
            ).then(
                update_performance_display,
                inputs=[correct_answers, total_questions],
                outputs=[performance_markdown]
            )
            
            next_btn.click(
                handle_next,
                inputs=[current_question],
                outputs=[current_question]
            )
            
            regenerate_btn.click(
                lambda: gr.update(value="Regenerating questions... Please wait...", visible=True),
                outputs=[loading_msg]
            ).then(
                handle_regenerate,
                inputs=[current_pdf],
                outputs=[questions_state, current_question, correct_answers, total_questions, loading_msg]
            ).then(
                update_performance_display,
                inputs=[correct_answers, total_questions],
                outputs=[performance_markdown]
            )
            
            # Modified event handler for generate_solution_btn to support generator function
            generate_solution_btn.click(
                generate_solution,
                inputs=[current_question, questions_state, current_pdf],
                outputs=[solution],
                queue=True  # Enable queueing for streaming updates
            )
            
        # Set up the process button click handler
        process_btn.click(
            handle_pdf_processing,
            inputs=[pdf_input],
            outputs=[
                summary_output,
                questions_state,
                current_question,
                correct_answers,
                total_questions,
                current_pdf,
                quiz_status
            ]
        ).then(
            generate_questions_async,
            inputs=[current_pdf],
            outputs=[
                questions_state,
                current_question,
                correct_answers,
                total_questions,
                quiz_status
            ]
        ).then(
            update_question,
            inputs=[current_question, questions_state],
            outputs=[question_text, answer_choices, feedback, solution, submit_btn, next_btn, regenerate_btn, loading_msg, generate_solution_btn]
        ).then(
            update_performance_display,
            inputs=[correct_answers, total_questions],
            outputs=[performance_markdown]
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()