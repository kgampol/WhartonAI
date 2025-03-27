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
            return gr.update(value="No PDF file uploaded."), gr.update(value=0), gr.update(value=0), gr.update(value="Please upload a PDF file first.")
        
        # Record the start time for performance monitoring
        start_time = time.time()
        
        # Show initial loading message
        yield gr.update(value="Processing PDF file...", visible=True)
        
        # Process PDF and generate questions
        processor = PDFProcessor()
        text, success = processor.extract_text(pdf_file.name)
        if not success:
            yield gr.update(value="Error processing PDF."), gr.update(value=0), gr.update(value=0), gr.update(value="Error processing PDF. Please try again.")
            return
        
        # Show progress for question generation
        current_time = time.time() - start_time
        yield gr.update(value=f"Generating questions... ({current_time:.2f}s)", visible=True)
        
        questions = processor.generate_questions(text)
        if not questions or 'questions' not in questions:
            yield gr.update(value="Error generating questions."), gr.update(value=0), gr.update(value=0), gr.update(value="Error generating questions. Please try again.")
            return
        
        # Show final success message with timing
        total_time = time.time() - start_time
        yield gr.update(value=questions), gr.update(value=0), gr.update(value=0), gr.update(value=f"Successfully generated questions in {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in handle_regenerate: {str(e)}")
        yield gr.update(value="Error occurred."), gr.update(value=0), gr.update(value=0), gr.update(value=f"Error: {str(e)}")

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
            yield None, 0, 0, 0, "Error: No PDF file provided"
            return
        
        processor = PDFProcessor()
        text, success = processor.extract_text(pdf_file.name)
        if not success:
            yield None, 0, 0, 0, "Error processing PDF"
            return
        
        # Show progress message without timing
        yield None, 0, 0, 0, "Generating questions..."
        
        # Generate questions
        questions = processor.generate_questions(text)
        if not questions or 'questions' not in questions or not questions['questions']:
            yield None, 0, 0, 0, "Error generating questions"
            return
        
        # Show final success message without timing
        print(f"Successfully generated {len(questions['questions'])} questions")
        yield questions, 0, 0, 0, f"Quiz ready! Generated {len(questions['questions'])} questions"
        
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        yield None, 0, 0, 0, f"Error: {str(e)}"

def generate_performance_review(questions, correct_answers, total_questions, wrong_indices):
    try:
        if not questions or 'questions' not in questions or not questions['questions']:
            return "No questions available for review."
        
        # Get questions that were answered incorrectly using wrong_indices
        wrong_questions = []
        for i in wrong_indices:
            if i < len(questions['questions']):
                question = questions['questions'][i]
                wrong_questions.append({
                    'question': question['question'],
                    'key_concepts': question['key_concepts'],
                    'citations': question['citations']
                })
        
        if not wrong_questions:
            return "Congratulations! You got all questions correct! No areas need improvement."
        
        # Prepare the prompt for GPT
        prompt = f"""Based on the following questions that were answered incorrectly, provide a detailed performance review:
        
        Questions answered incorrectly:
        {json.dumps(wrong_questions, indent=2)}
        
        Please provide:
        1. A summary of topics that need improvement
        2. Specific lecture slides to review
        3. Key concepts to focus on
        4. Study recommendations
        
        Format the response in a clear, structured way with sections and bullet points."""
        
        # Call GPT-3.5 API
        response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational advisor helping students identify areas for improvement in their studies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating performance review: {str(e)}")
        return f"Error generating performance review: {str(e)}"

def handle_quiz_completion(questions, correct_answers, total_questions, wrong_indices):
    try:
        print(f"Handling quiz completion with questions: {questions}, correct_answers: {correct_answers}, total_questions: {total_questions}, wrong_indices: {wrong_indices}")
        
        # Only generate review if we've completed all questions
        if not questions or 'questions' not in questions or not questions['questions']:
            print("No questions available for review")
            return gr.update(value="", visible=False)
            
        # Check if we've completed all questions
        if total_questions != len(questions['questions']):
            print(f"Not all questions completed. Total: {total_questions}, Available: {len(questions['questions'])}")
            return gr.update(value="", visible=False)
        
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
        return gr.update(value=formatted_review, visible=True)
        
    except Exception as e:
        print(f"Error in handle_quiz_completion: {str(e)}")
        return gr.update(value=f"Error generating review: {str(e)}", visible=True)

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
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True, interactive=False),
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
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True, interactive=False),
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
                            submit_btn: gr.update(visible=True),
                            next_btn: gr.update(visible=True, interactive=False),
                            regenerate_btn: gr.update(visible=False),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False),
                            performance_review: gr.update(value="", visible=False)
                        }
                    
                    if index >= len(questions_list):
                        print("Quiz completed")
                        return {
                            question_text: gr.update(value="Quiz completed!"),
                            answer_choices: gr.update(choices=[], value=None, interactive=True, visible=False),
                            feedback: gr.update(value=""),
                            solution: gr.update(value="", visible=False),
                            submit_btn: gr.update(visible=False),
                            next_btn: gr.update(visible=False),
                            regenerate_btn: gr.update(visible=True),
                            loading_msg: gr.update(value="", visible=False),
                            generate_solution_btn: gr.update(visible=False),
                            performance_review: gr.update(value="Generating performance review...", visible=True)
                        }
                    
                    question = questions_list[index]
                    print(f"Displaying question {index + 1}: {question['question'][:50]}...")
                    
                    return {
                        question_text: gr.update(value=f"Question {index + 1}: {question['question']}"),
                        answer_choices: gr.update(choices=question['options'], value=None, interactive=True, visible=True),
                        feedback: gr.update(value=""),
                        solution: gr.update(value="", visible=False),
                        submit_btn: gr.update(visible=True),
                        next_btn: gr.update(visible=True, interactive=False),
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
                        submit_btn: gr.update(visible=True),
                        next_btn: gr.update(visible=True, interactive=False),
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
                    
                    question = questions['questions'][question_index]
                    correct_answer_index = question['correct_answer']
                    options = question['options']
                    
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
                    
                    # Always disable submit button after first submission and enable next button
                    return feedback_msg, new_correct, new_total, gr.update(interactive=False), solution, generate_solution_btn, gr.update(visible=False), gr.update(interactive=True), new_wrong_indices
                    
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
            )
            
            next_btn.click(
                handle_next,
                inputs=[current_question],
                outputs=[current_question]
            ).then(
                lambda q, c, t, w: handle_quiz_completion(q, c, t, w),
                inputs=[questions_state, correct_answers, total_questions, wrong_indices],
                outputs=[performance_review],
                queue=True
            )
            
            regenerate_btn.click(
                lambda: gr.update(value="Starting question generation...", visible=True),
                outputs=[loading_msg]
            ).then(
                handle_regenerate,
                inputs=[current_pdf],
                outputs=[questions_state, current_question, correct_answers, total_questions, loading_msg],
                queue=True  # Enable queueing for streaming updates
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