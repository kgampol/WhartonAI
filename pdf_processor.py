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
import re

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
                    {"role": "system", "content": "You are a top-performing student taking clear, thorough lecture notes for study purposes. Your goal is to create easy-to-understand, organized, and information-rich notes that summarize the material in full detail. Capture all key concepts, definitions, equations, examples, and diagrams (described if not visible). Use bullet points, headings, or numbering to make the notes easy to follow. Prioritize clarity, completeness, and study-readiness."},
                    {"role": "user", "content": f"Summarize the following lecture notes as if you're a student taking detailed notes for future studying. Include all important points, concepts, formulas, examples, and visual elements (describe diagrams if present). Make the notes as clear, complete, and structured as possible:\n\n{text}"}
                ],
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
            return response
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return None

    def _split_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """
        Split text into chunks of approximately equal size.
        
        Args:
            text (str): The text to split
            chunk_size (int): Target size for each chunk
            
        Returns:
            List[str]: List of text chunks
        """
        # Split text into sentences
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, start a new chunk
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks

    def generate_questions(self, text: str) -> Optional[Dict]:
        """
        Generate multiple-choice questions from the text using GPT-3.5 Turbo.
        
        Args:
            text (str): The text content to generate questions from
            
        Returns:
            Optional[Dict]: Dictionary containing generated questions or None if generation fails
        """
        try:
            # Split text into chunks for better context management
            chunks = self._split_text(text)
            all_questions = []
            
            for chunk in chunks:
                # Generate questions for this chunk
                prompt = f"""
                Based on the following lecture content, generate 2 multiple-choice questions.
                Return ONLY a JSON array of questions, with no additional text or explanation.
                
                Each question should be a JSON object with these exact fields:
                {{
                    "question": "The question text",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "correct_answer": 0,  // Index of correct answer (0-3)
                    "key_concepts": ["Concept 1", "Concept 2"],
                    "citations": ["Citation 1", "Citation 2"]
                }}
                
                Lecture content:
                {chunk}
                
                Requirements:
                1. Return ONLY the JSON array, no other text
                2. Each question must have exactly 4 options
                3. correct_answer must be an integer 0-3
                4. Test understanding of key concepts
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a JSON-only response assistant. Return only valid JSON arrays containing question objects."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                try:
                    # Get the response text
                    questions_text = response.choices[0].message.content.strip()
                    
                    # Try to parse the entire response as JSON first
                    try:
                        chunk_questions = json.loads(questions_text)
                    except json.JSONDecodeError:
                        # If that fails, try to extract JSON array using regex
                        questions_match = re.search(r'\[.*\]', questions_text, re.DOTALL)
                        if questions_match:
                            questions_json = questions_match.group(0)
                            chunk_questions = json.loads(questions_json)
                        else:
                            print(f"Could not find valid JSON array in response: {questions_text[:100]}...")
                            continue
                    
                    # Validate each question
                    valid_questions = []
                    for q in chunk_questions:
                        if not isinstance(q, dict):
                            continue
                            
                        # Check required fields
                        required_fields = ['question', 'options', 'correct_answer', 'key_concepts', 'citations']
                        if not all(field in q for field in required_fields):
                            continue
                            
                        # Validate options
                        if not isinstance(q['options'], list) or len(q['options']) != 4:
                            continue
                            
                        # Validate correct_answer
                        if not isinstance(q['correct_answer'], int) or q['correct_answer'] < 0 or q['correct_answer'] >= 4:
                            continue
                            
                        valid_questions.append(q)
                    
                    all_questions.extend(valid_questions)
                    
                except Exception as e:
                    print(f"Error parsing questions from response: {str(e)}")
                    print(f"Response text: {questions_text[:200]}...")
                    continue
            
            if not all_questions:
                print("No valid questions generated")
                return None
                
            print(f"Successfully generated {len(all_questions)} valid questions")
            return {"questions": all_questions}
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return None

    def generate_solution(self, question: Dict, lecture_text: str) -> Optional[str]:
        """
        Generate a detailed step-by-step solution for a question using GPT-4 Turbo.
        
        Args:
            question (Dict): The question object containing question text, options, and correct answer
            lecture_text (str): The relevant lecture text for context
            
        Returns:
            Optional[str]: Generated solution text or None if generation fails
        """
        try:
            prompt = f"""
            Generate a detailed step-by-step solution for this question based on the lecture content.
            Include specific citations from the lecture text.
            
            Question: {question['question']}
            Options: {question['options']}
            Correct Answer: {question['options'][question['correct_answer']]}
            Key Concepts: {question['key_concepts']}
            
            Lecture Content:
            {lecture_text}
            
            Provide a solution that:
            1. Explains why the correct answer is right
            2. Shows step-by-step reasoning
            3. References specific parts of the lecture
            4. Explains why other options are incorrect
            5. Uses clear, concise language
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at explaining complex concepts step by step."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating solution: {str(e)}")
            return None

    def generate_targeted_questions(self, text: str, weak_concepts: List[str]) -> Optional[Dict]:
        """
        Generate questions specifically targeting weak concepts.
        
        Args:
            text (str): The lecture content
            weak_concepts (List[str]): List of concepts that need more practice
            
        Returns:
            Optional[Dict]: Dictionary containing generated questions or None if generation fails
        """
        try:
            prompt = f"""
            Based on the lecture content below, create 3-5 multiple-choice questions that target these weak concepts:
            {', '.join(weak_concepts)}

            Return ONLY a valid JSON array of questions.
            Each question should have:
            - question
            - options (4)
            - correct_answer (index)
            - key_concepts
            - citations

            Lecture content:
            {text[:3000]}
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an educational assistant who only outputs valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            questions_text = response.choices[0].message.content.strip()
            
            # Try to parse the response as JSON
            try:
                questions = json.loads(questions_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON array using regex
                questions_match = re.search(r'\[.*\]', questions_text, re.DOTALL)
                if questions_match:
                    questions_json = questions_match.group(0)
                    questions = json.loads(questions_json)
                else:
                    print(f"Could not find valid JSON array in response: {questions_text[:100]}...")
                    return None
            
            # Validate each question
            valid_questions = []
            for q in questions:
                if not isinstance(q, dict):
                    continue
                    
                # Check required fields
                required_fields = ['question', 'options', 'correct_answer', 'key_concepts', 'citations']
                if not all(field in q for field in required_fields):
                    continue
                    
                # Validate options
                if not isinstance(q['options'], list) or len(q['options']) != 4:
                    continue
                    
                # Validate correct_answer
                if not isinstance(q['correct_answer'], int) or q['correct_answer'] < 0 or q['correct_answer'] >= 4:
                    continue
                    
                valid_questions.append(q)
            
            if not valid_questions:
                print("No valid questions generated")
                return None
                
            print(f"Successfully generated {len(valid_questions)} targeted questions")
            return {"questions": valid_questions}
            
        except Exception as e:
            print(f"Error in targeted question generation: {str(e)}")
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