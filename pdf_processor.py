"""
PDF Processing Module
====================

This module provides functionality for processing PDF files, extracting text content,
and generating AI-powered summaries and quiz questions using OpenAI's GPT models.

Dependencies:
- pypdf: For PDF text extraction
- openai: For AI text generation
- python-dotenv: For loading environment variables
- time: For implementing retry delays

Classes:
--------
PDFProcessor
    Main class that handles all PDF processing functionality.
"""

from pypdf import PdfReader
from openai import OpenAI
import json
import os
import time
from dotenv import load_dotenv
from typing import Optional, Tuple, Any, Callable, Dict, List

# Load environment variables from .env file
load_dotenv()

class PDFProcessor:
    """
    A class to process PDF files and generate AI-powered content from them.

    This class handles PDF text extraction and uses OpenAI's GPT models to generate
    summaries and quiz questions from the extracted content.

    Attributes:
        client (OpenAI): OpenAI API client instance

    Args:
        api_key (str, optional): OpenAI API key. If not provided, loads from environment variables.
    """

    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
    def extract_text(self, pdf_file) -> Tuple[str, bool]:
        """
        Extracts text content from a PDF file.

        Processes the PDF page by page, handling potential errors for individual pages
        without failing the entire extraction.

        Args:
            pdf_file: File object or path to PDF file

        Returns:
            tuple: (str, bool) where:
                - str is the extracted text (empty string if extraction fails)
                - bool indicates if the extraction was successful

        Raises:
            No exceptions are raised; errors are handled internally
        """
        try:
            reader = PdfReader(pdf_file)
            
            # Check if PDF is empty
            if len(reader.pages) == 0:
                return "", False
                
            text = ""
            has_content = False
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                        has_content = True
                    else:
                        print(f"Warning: Page {page_num} is empty")
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {str(e)}")
                    continue
            
            if not has_content:
                return "", False
                
            return text.strip(), True
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return "", False
    
    def safe_openai_call(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """
        Safely make an OpenAI API call with retries and error handling.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            model (str): OpenAI model to use
            temperature (float): Temperature for response generation
            max_tokens (int): Maximum tokens in response
            
        Returns:
            Optional[str]: The API response content or None if the call fails
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response and response.choices:
                    return response.choices[0].message.content
                    
            except Exception as e:
                print(f"OpenAI API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
        
        return None

    def generate_summary(self, text: str) -> Optional[str]:
        """
        Generates an AI-powered summary of the provided text.

        Uses OpenAI's GPT-3.5-turbo model to create a concise summary of academic content.
        Truncates large text inputs to avoid token limit errors.

        Args:
            text (str): The text content to summarize

        Returns:
            Optional[str]: Generated summary text, or None if generation fails

        Raises:
            No exceptions are raised; errors are handled internally
        """
        try:
            # Truncate text if it's too long (approximately 3000 characters to stay within token limits)
            if len(text) > 3000:
                text = text[:3000] + "..."

            response = self.safe_openai_call(
                [
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of academic content. Include all important details and key points, equations, and diagrams needed to understand the content and answer questions."},
                    {"role": "user", "content": f"Please summarize the following lecture notes:\n\n{text}"}
                ],
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
            return response
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return None

    def generate_questions(self, text: str, num_questions: int = 5) -> Optional[Dict]:
        """
        Generate multiple-choice questions from the text.
        
        Args:
            text (str): The text to generate questions from
            num_questions (int): Number of questions to generate
            
        Returns:
            Optional[Dict]: Dictionary containing questions and answers, or None if generation fails
        """
        try:
            # Truncate text if too long
            if len(text) > 3000:
                text = text[:3000] + "..."
            
            # System prompt for question generation
            system_prompt = '''You are an expert at creating multiple-choice questions.
            Follow these rules:
            1. Each question must have exactly 4 options
            2. The correct answer must be one of the options
            3. Questions should be clear and unambiguous
            4. Options should be distinct and plausible
            5. The correct_answer field must be the 0-based index of the correct option
            6. Each question must test understanding, not just memorization
            7. Avoid obvious patterns in correct answer positions
            8. Make sure all fields are properly formatted strings
            9. The JSON structure must be exactly as shown in the example
            
            Example format:
            {
                "questions": [
                    {
                        "question": "What is the capital of France?",
                        "options": [
                            "London",
                            "Paris",
                            "Berlin",
                            "Madrid"
                        ],
                        "correct_answer": 1
                    }
                ]
            }'''
            
            # User prompt for question generation
            user_prompt = f'''Generate {num_questions} multiple-choice questions based on this text:
            {text}
            
            Return the questions in valid JSON format exactly as shown in the example.
            Make sure each question has exactly 4 options and a valid correct_answer index.'''
            
            # Generate questions using OpenAI
            response = self.safe_openai_call(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
            
            if not response:
                print("Failed to generate questions")
                return None
            
            # Parse the response as JSON
            try:
                questions_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {str(e)}")
                return None
            
            # Validate the structure
            if not isinstance(questions_data, dict) or 'questions' not in questions_data:
                print("Invalid JSON structure: missing 'questions' key")
                return None
            
            questions_list = questions_data['questions']
            if not isinstance(questions_list, list):
                print("Invalid JSON structure: 'questions' must be a list")
                return None
            
            # Validate each question
            valid_questions = []
            for i, q in enumerate(questions_list):
                try:
                    # Check required fields
                    if not all(k in q for k in ['question', 'options', 'correct_answer']):
                        print(f"Missing required fields in question {i}")
                        continue
                    
                    # Validate question text
                    if not isinstance(q['question'], str) or not q['question'].strip():
                        print(f"Invalid question text at index {i}")
                        continue
                    
                    # Validate options
                    if not isinstance(q['options'], list) or len(q['options']) != 4:
                        print(f"Invalid options at index {i}: must be a list of exactly 4 items")
                        continue
                    
                    if not all(isinstance(opt, str) and opt.strip() for opt in q['options']):
                        print(f"Invalid option format at index {i}: all options must be non-empty strings")
                        continue
                    
                    # Validate correct answer
                    if not isinstance(q['correct_answer'], int) or q['correct_answer'] < 0 or q['correct_answer'] >= 4:
                        print(f"Invalid correct_answer at index {i}: must be an integer between 0 and 3")
                        continue
                    
                    # Add valid question to list
                    valid_questions.append({
                        'question': q['question'].strip(),
                        'options': [opt.strip() for opt in q['options']],
                        'correct_answer': q['correct_answer']
                    })
                    
                except Exception as e:
                    print(f"Error validating question {i}: {str(e)}")
                    continue
            
            if not valid_questions:
                print("No valid questions generated")
                return None
            
            print(f"Successfully generated {len(valid_questions)} valid questions")
            return {'questions': valid_questions}
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return None

    def process_pdf(self, pdf_file) -> Tuple[Optional[str], Optional[dict]]:
        """
        Main method to process a PDF file and generate both summary and questions.

        This method coordinates the complete PDF processing workflow:
        1. Extracts text from the PDF
        2. Generates a summary of the content
        3. Generates quiz questions from the content

        Args:
            pdf_file: File object or path to PDF file

        Returns:
            tuple: (Optional[str], Optional[dict]) where:
                - First element is the generated summary (or None if failed)
                - Second element is the generated questions (or None if failed)

        Raises:
            No exceptions are raised; errors are handled internally
        """
        try:
            # Extract text from PDF
            text, is_valid = self.extract_text(pdf_file)
            
            if not is_valid:
                print("Failed to extract text from PDF")
                return None, None
            
            print(f"Successfully extracted text from PDF: {len(text)} characters")
            
            # Generate summary
            summary = self.generate_summary(text)
            if not summary:
                print("Failed to generate summary")
                return None, None
            
            print("Successfully generated summary")
            
            # Generate questions
            questions = self.generate_questions(text)
            if not questions:
                print("Failed to generate questions")
                return summary, None
            
            print(f"Successfully generated {len(questions['questions'])} questions")
            return summary, questions
            
        except Exception as e:
            print(f"Error in process_pdf: {str(e)}")
            return None, None 