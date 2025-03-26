"""
PDF Processing Module
====================

This module provides functionality for processing PDF files, extracting text content,
and generating AI-powered summaries and quiz questions using OpenAI's GPT models.

Dependencies:
- pypdf: For PDF text extraction
- openai: For AI text generation
- python-dotenv: For loading environment variables

Classes:
--------
PDFProcessor
    Main class that handles all PDF processing functionality.
"""

from pypdf import PdfReader
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from typing import Optional, Tuple

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
    
    def generate_summary(self, text: str) -> Optional[str]:
        """
        Generates an AI-powered summary of the provided text.

        Uses OpenAI's GPT-3.5-turbo model to create a concise summary of academic content.

        Args:
            text (str): The text content to summarize

        Returns:
            Optional[str]: Generated summary text, or None if generation fails

        Raises:
            No exceptions are raised; errors are handled internally
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of academic content."},
                    {"role": "user", "content": f"Please summarize the following lecture notes:\n\n{text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return None

    def generate_questions(self, text: str, num_questions: int = 5) -> Optional[dict]:
        """
        Generates AI-powered multiple-choice questions from the provided text.

        Uses OpenAI's GPT-3.5-turbo model to create quiz questions based on the content.

        Args:
            text (str): The text content to generate questions from
            num_questions (int, optional): Number of questions to generate. Defaults to 5.

        Returns:
            Optional[dict]: JSON object containing questions, options, and correct answers,
                          or None if generation fails. Format:
                          {
                              'questions': [
                                  {
                                      'question': str,
                                      'options': List[str],
                                      'correct_answer': int
                                  },
                                  ...
                              ]
                          }

        Raises:
            No exceptions are raised; errors are handled internally
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Create multiple-choice questions based on the provided content. Format the response as JSON with the following structure: {'questions': [{'question': '...', 'options': ['...', '...', '...', '...'], 'correct_answer': 0}]}"},
                    {"role": "user", "content": f"Generate {num_questions} multiple-choice questions with 4 options each based on:\n\n{text}"}
                ]
            )
            return json.loads(response.choices[0].message.content)
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
        # Extract text from PDF
        text, is_valid = self.extract_text(pdf_file)
        
        if not is_valid:
            return None, None
        
        # Generate summary
        summary = self.generate_summary(text)
        
        # Generate questions
        questions = self.generate_questions(text)
        
        return summary, questions 