# WhartonAI
WhartonAI-Thon project
# WhartonAI - Lecture Notes Assistant

An AI-powered tool that helps you better understand and retain your lecture materials by creating summaries, generating quizzes, and providing detailed performance reviews.

## Features

### 1. PDF Processing
- Upload your lecture notes in PDF format
- Automatic text extraction and processing
- Support for various PDF layouts and formats

### 2. AI-Powered Summary
- Concise summary of lecture content
- Key points and main concepts highlighted
- Easy to understand format
- Citations to original lecture sections

### 3. Interactive Quiz
- Multiple-choice questions generated from lecture content
- Questions test understanding of key concepts
- Immediate feedback on answers
- Step-by-step solutions for incorrect answers
- Progress tracking throughout the quiz

### 4. Performance Review
- Detailed analysis of quiz performance
- Personalized feedback on areas for improvement
- Topic-specific study recommendations
- Actionable learning strategies
- List of questions to review with:
  - Correct answers
  - Key concepts
  - Relevant lecture sections

### 5. Study Tools
- Question regeneration for additional practice
- Performance tracking across attempts
- Citation-based learning
- Concept-focused review

## How to Use

1. **Upload Your Notes**
   - Click the file upload button
   - Select your lecture notes PDF
   - Wait for processing to complete

2. **Review the Summary**
   - Read the AI-generated summary
   - Note key concepts and main points
   - Use citations to reference original content

3. **Take the Quiz**
   - Answer multiple-choice questions
   - Get immediate feedback
   - View solutions for incorrect answers
   - Track your progress

4. **Review Your Performance**
   - Get detailed feedback on your performance
   - Identify areas for improvement
   - Access personalized study recommendations
   - Review questions you got wrong

5. **Practice More**
   - Regenerate questions for additional practice
   - Focus on specific topics
   - Track improvement over time

## Technical Details

- Built with Python and Gradio
- Uses OpenAI's GPT models for content generation
- PDF processing with PyPDF2
- Efficient text chunking for better context management
- Streaming responses for real-time feedback

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kgampol/WhartonAI.git
cd WhartonAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

4. Run the application:
```bash
python app.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT-3.5-turbo API
- Gradio team for the excellent UI framework
- All contributors who have helped improve this project

---
Last updated: March 2024
