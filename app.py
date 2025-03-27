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
        if not pdf_file:
            return gr.update(value=None), gr.update(value=0), gr.update(value=0), gr.update(value=[]), gr.update(value="Please upload a PDF file first."), gr.update(visible=False)
        
        # Record the start time for performance monitoring
        start_time = time.time()
        
        # Show initial loading message
        yield gr.update(value=None), gr.update(value=0), gr.update(value=0), gr.update(value=[]), gr.update(value="Processing PDF file..."), gr.update(visible=False)
        
        # Process PDF and generate questions
        processor = PDFProcessor()
        text, success = processor.extract_text(pdf_file.name)
        if not success:
            yield gr.update(value=None), gr.update(value=0), gr.update(value=0), gr.update(value=[]), gr.update(value="Error processing PDF. Please try again."), gr.update(visible=False)
            return
        
        # Show progress for question generation
        current_time = time.time() - start_time
        yield gr.update(value=None), gr.update(value=0), gr.update(value=0), gr.update(value=[]), gr.update(value=f"Generating questions... ({current_time:.2f}s)"), gr.update(visible=False)
        
        questions = processor.generate_questions(text)
        if not questions or 'questions' not in questions:
            yield gr.update(value=None), gr.update(value=0), gr.update(value=0), gr.update(value=[]), gr.update(value="Error generating questions. Please try again."), gr.update(visible=False)
            return
        
        # Show final success message with timing
        total_time = time.time() - start_time
        print(f"Successfully generated {len(questions['questions'])} questions")
        yield gr.update(value=questions), gr.update(value=0), gr.update(value=0), gr.update(value=[]), gr.update(value=f"Successfully generated {len(questions['questions'])} questions in {total_time:.2f} seconds"), gr.update(visible=False)
        
    except Exception as e:
        print(f"Error in handle_regenerate: {str(e)}")
        yield gr.update(value=None), gr.update(value=0), gr.update(value=0), gr.update(value=[]), gr.update(value=f"Error: {str(e)}"), gr.update(visible=False)

def handle_pdf_processing(pdf_file):
    try:
        print("Processing PDF file...")
        # First generate and show the summary
        processor = PDFProcessor()
        text, success = processor.extract_text(pdf_file.name)
        if not success:
            return "Error processing PDF. Please try again.", None, 0, 0, 0, None, "Error processing PDF", gr.update(visible=False)
        
        summary = processor.generate_summary(text)
        if summary is None:
            return "Error generating summary. Please try again.", None, 0, 0, 0, None, "Error generating summary", gr.update(visible=False)
        
        # Show summary immediately and indicate quiz generation
        quiz_status = "Generating quiz questions... Please wait..."
        
        # Return immediately with summary and status
        return summary, None, 0, 0, 0, pdf_file, quiz_status, gr.update(visible=False)
        
    except Exception as e:
        print(f"Error in handle_pdf_processing: {str(e)}")
        return f"Error processing PDF: {str(e)}", None, 0, 0, 0, None, f"Error: {str(e)}", gr.update(visible=False)

def generate_questions_async(pdf_file):
    try:
        if pdf_file is None:
            yield None, 0, 0, 0, "Error: No PDF file provided", gr.update(visible=False)
            return
        
        processor = PDFProcessor()
        text, success = processor.extract_text(pdf_file.name)
        if not success:
            yield None, 0, 0, 0, "Error processing PDF", gr.update(visible=False)
            return
        
        # Show progress message without timing
        yield None, 0, 0, 0, "Generating questions...", gr.update(visible=False)
        
        # Generate questions
        questions = processor.generate_questions(text)
        if not questions or 'questions' not in questions or not questions['questions']:
            yield None, 0, 0, 0, "Error generating questions", gr.update(visible=False)
            return
        
        # Show final success message without timing
        print(f"Successfully generated {len(questions['questions'])} questions")
        yield questions, 0, 0, 0, f"Quiz ready! Generated {len(questions['questions'])} questions", gr.update(visible=False)
        
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        yield None, 0, 0, 0, f"Error: {str(e)}", gr.update(visible=False)

def generate_performance_review(questions, correct_answers, total_questions, wrong_indices):
    """
    Generate a performance review based on incorrectly answered questions.
    Optimized for lower token usage while maintaining useful feedback.
    
    Args:
        questions (dict): Dictionary containing all questions and their details
        correct_answers (int): Number of correctly answered questions
        total_questions (int): Total number of questions answered
        wrong_indices (list): List of indices for incorrectly answered questions
        
    Returns:
        str: Generated performance review text
    """
    try:
        # Validate input data
        if not questions or 'questions' not in questions or not questions['questions']:
            return "No questions available for review."
        
        # Get up to 3 wrong questions with minimal required information
        wrong_questions = []
        for i in wrong_indices[:3]:  # Limit to 3 questions
            if i < len(questions['questions']):
                question = questions['questions'][i]
                # Extract only essential information
                wrong_questions.append({
                    'question': question['question'],
                    'correct_answer': question['options'][question['correct_answer']],
                    'key_concept': question['key_concepts'][0] if question.get('key_concepts') else None,
                    'citation': question['citations'][0] if question.get('citations') else None
                })
        
        if not wrong_questions:
            return "Congratulations! You got all questions correct! No areas need improvement."
        
        # Create a concise prompt that focuses on key information
        prompt = f"""Review these incorrect answers and provide focused feedback:

Questions:
{json.dumps(wrong_questions, indent=2)}

Provide:
1. Main topics needing improvement
2. Specific slides to review
3. Key study tips

Keep feedback concise and actionable."""
        
        # Call GPT-3.5 API with optimized parameters
        response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise educational advisor. Provide focused, actionable feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating performance review: {str(e)}")
        return f"Error generating review: {str(e)}"

def handle_quiz_completion(questions, correct_answers, total_questions, wrong_indices):
    try:
        print(f"Handling quiz completion with questions: {questions}, correct_answers: {correct_answers}, total_questions: {total_questions}, wrong_indices: {wrong_indices}")
        
        # Only generate review if we've completed all questions
        if not questions or 'questions' not in questions or not questions['questions']:
            print("No questions available for review")
            return gr.update(value="", visible=False), gr.update(visible=False)
            
        # Check if we've completed all questions
        if total_questions != len(questions['questions']):
            print(f"Not all questions completed. Total: {total_questions}, Available: {len(questions['questions'])}")
            return gr.update(value="", visible=False), gr.update(visible=False)
        
        print("Generating performance review...")
        
        # Generate the review
        review = generate_performance_review(questions, correct_answers, total_questions, wrong_indices)
        
        # Format the review with a clear header
        formatted_review = f"""
## Performance Review

{review}

---
*Review generated based on your quiz performance. Use this information to focus your study efforts on areas that need improvement.*
"""
        
        print("Review generated successfully")
        return gr.update(value=formatted_review, visible=True), gr.update(visible=True)
        
    except Exception as e:
        print(f"Error in handle_quiz_completion: {str(e)}")
        return gr.update(value=f"Error generating review: {str(e)}", visible=True), gr.update(visible=True)

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
        wrong_indices = gr.State([])  # Track indices of wrong answers
        
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload PDF")
                process_btn = gr.Button("Process PDF")
                
            with gr.Column():
                summary_output = gr.Textbox(label="Summary", lines=10)
                quiz_status = gr.Markdown(value="", visible=True)
                
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
            
            # Performance review
            performance_review = gr.Markdown(value="", visible=False)
            
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
                            answer_choices: gr.update(choices=[], value=None, interactive=True, visible=True),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=True, interactive=True),
                            next_btn: gr.update(visible=True, interactive=False, value="Next Question"),
                            regenerate_btn: gr.update(visible=False),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False),
                            performance_review: gr.update(value="", visible=False)
                        }
                    
                    if 'questions' not in questions:
                        print("Questions dictionary missing 'questions' key")
                        return {
                            question_text: gr.update(value="Invalid questions format."),
                            answer_choices: gr.update(choices=[], value=None, interactive=True, visible=True),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=True, interactive=True),
                            next_btn: gr.update(visible=True, interactive=False, value="Next Question"),
                            regenerate_btn: gr.update(visible=False),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False),
                            performance_review: gr.update(value="", visible=False)
                        }
                    
                    questions_list = questions['questions']
                    if not questions_list:
                        print("Questions list is empty")
                        return {
                            question_text: gr.update(value="No questions available."),
                            answer_choices: gr.update(choices=[], value=None, interactive=True, visible=True),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=True, interactive=True),
                            next_btn: gr.update(visible=True, interactive=False, value="Next Question"),
                            regenerate_btn: gr.update(visible=False),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False),
                            performance_review: gr.update(value="", visible=False)
                        }
                    
                    # Ensure index is within bounds
                    safe_index = max(0, min(index, len(questions_list) - 1))
                    
                    if safe_index >= len(questions_list):
                        print("Quiz completed")
                        return {
                            question_text: gr.update(value="Quiz completed!"),
                            answer_choices: gr.update(choices=[], value=None, interactive=True, visible=False),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=False),
                            next_btn: gr.update(visible=True, value="End Quiz", interactive=True),
                            regenerate_btn: gr.update(visible=False),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False),
                            performance_review: gr.update(value="", visible=False)
                        }
                    
                    question = questions_list[safe_index]
                    print(f"Displaying question {safe_index + 1}: {question['question'][:50]}...")
                    
                    # Reset all states for the new question
                    return {
                        question_text: gr.update(value=f"Question {safe_index + 1}: {question['question']}"),
                        answer_choices: gr.update(choices=question['options'], value=None, interactive=True, visible=True),
                        feedback: gr.update(value=""),
                        solution: gr.update(value="", visible=False),
                        submit_btn: gr.update(visible=True, interactive=True),
                        next_btn: gr.update(visible=True, interactive=False, value="Next Question"),
                        regenerate_btn: gr.update(visible=False),
                        loading_msg: gr.update(value="", visible=False),
                        generate_solution_btn: gr.update(visible=False),
                        performance_review: gr.update(value="", visible=False)
                    }
                except Exception as e:
                    print(f"Error in update_question: {str(e)}")
                    return {
                        question_text: gr.update(value=f"Error displaying question: {str(e)}"),
                        answer_choices: gr.update(choices=[], value=None, interactive=True, visible=True),
                        feedback: gr.update(value=""),
                        solution: gr.update(value="", visible=False),
                        submit_btn: gr.update(visible=True, interactive=True),
                        next_btn: gr.update(visible=True, interactive=False, value="Next Question"),
                        regenerate_btn: gr.update(visible=False),
                        loading_msg: gr.update(value="", visible=False),
                        generate_solution_btn: gr.update(visible=False),
                        performance_review: gr.update(value="", visible=False)
                    }
                
            def handle_submit(question_index, selected_answer, questions, correct, total, wrong_indices):
                try:
                    if not questions or 'questions' not in questions:
                        return "No questions available.", correct, total, gr.update(interactive=False), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=False), wrong_indices
                    
                    if selected_answer is None:
                        return "Please select an answer.", correct, total, gr.update(interactive=True), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False), wrong_indices
                    
                    # Validate question index
                    if question_index >= len(questions['questions']):
                        return "Error: Question index out of range.", correct, total, gr.update(interactive=True), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False), wrong_indices
                    
                    question = questions['questions'][question_index]
                    if not question or 'options' not in question or 'correct_answer' not in question:
                        return "Error: Invalid question format.", correct, total, gr.update(interactive=True), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False), wrong_indices
                    
                    options = question['options']
                    correct_answer_index = question['correct_answer']
                    
                    # Validate selected answer exists in options
                    if selected_answer not in options:
                        return "Error: Selected answer not found in options.", correct, total, gr.update(interactive=True), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False), wrong_indices
                    
                    # Find the index of the selected answer in the options list
                    selected_index = options.index(selected_answer)
                    
                    # Update performance tracking
                    new_total = total + 1
                    new_correct = correct
                    new_wrong_indices = wrong_indices.copy()  # Create a copy to modify
                    
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
                        # Add the question index to wrong_indices if not already there
                        if question_index not in new_wrong_indices:
                            new_wrong_indices.append(question_index)
                    
                    # Check if this is the last question
                    is_last_question = question_index == len(questions['questions']) - 1
                    
                    # Update UI state based on whether this is the last question
                    if is_last_question:
                        return feedback_msg, new_correct, new_total, gr.update(interactive=False), solution, generate_solution_btn, gr.update(visible=False), gr.update(visible=True, value="End Quiz", interactive=True), new_wrong_indices
                    else:
                        return feedback_msg, new_correct, new_total, gr.update(interactive=False), solution, generate_solution_btn, gr.update(visible=False), gr.update(visible=True, interactive=True), new_wrong_indices
                    
                except Exception as e:
                    print(f"Error in handle_submit: {str(e)}")
                    return f"Error checking answer: {str(e)}", correct, total, gr.update(interactive=True), gr.update(value="", visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False), wrong_indices
                
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
                outputs=[question_text, answer_choices, feedback, solution, submit_btn, next_btn, regenerate_btn, loading_msg, generate_solution_btn, performance_review]
            )
            
            submit_btn.click(
                handle_submit,
                inputs=[current_question, answer_choices, questions_state, correct_answers, total_questions, wrong_indices],
                outputs=[feedback, correct_answers, total_questions, answer_choices, solution, generate_solution_btn, submit_btn, next_btn, wrong_indices]
            ).then(
                update_performance_display,
                inputs=[correct_answers, total_questions],
                outputs=[performance_markdown]
            ).then(
                lambda: gr.update(visible=False),
                outputs=[regenerate_btn]
            )
            
            next_btn.click(
                handle_next,
                inputs=[current_question],
                outputs=[current_question]
            ).then(
                lambda q, c, t, w: handle_quiz_completion(q, c, t, w),
                inputs=[questions_state, correct_answers, total_questions, wrong_indices],
                outputs=[performance_review, regenerate_btn],
                queue=True
            ).then(
                lambda: gr.update(visible=False),
                outputs=[next_btn]
            )
            
            regenerate_btn.click(
                lambda: gr.update(value="Starting question generation...", visible=True),
                outputs=[loading_msg]
            ).then(
                handle_regenerate,
                inputs=[current_pdf],
                outputs=[questions_state, current_question, correct_answers, wrong_indices, loading_msg, regenerate_btn],
                queue=True  # Enable queueing for streaming updates
            ).then(
                lambda _: 0,  # Explicitly set current_question to 0
                inputs=[questions_state],
                outputs=[current_question]
            ).then(
                update_question,
                inputs=[current_question, questions_state],
                outputs=[question_text, answer_choices, feedback, solution, submit_btn, next_btn, regenerate_btn, loading_msg, generate_solution_btn, performance_review]
            ).then(
                update_performance_display,
                inputs=[correct_answers, total_questions],
                outputs=[performance_markdown]
            ).then(
                lambda: gr.update(value="", visible=False),
                outputs=[loading_msg]
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
                quiz_status,
                regenerate_btn
            ]
        ).then(
            generate_questions_async,
            inputs=[current_pdf],
            outputs=[
                questions_state,
                current_question,
                correct_answers,
                total_questions,
                quiz_status,
                regenerate_btn
            ],
            queue=True  # Enable queueing for streaming updates
        ).then(
            update_question,
            inputs=[current_question, questions_state],
            outputs=[question_text, answer_choices, feedback, solution, submit_btn, next_btn, regenerate_btn, loading_msg, generate_solution_btn, performance_review]
        ).then(
            update_performance_display,
            inputs=[correct_answers, total_questions],
            outputs=[performance_markdown]
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()