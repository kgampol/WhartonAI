# WhartonAI
WhartonAI-Thon project
# WhartonAI - Lecture Notes Assistant

A powerful AI-powered application that helps students process and understand lecture notes through intelligent summarization and interactive quizzes.

## Features

### 1. PDF Processing
- Upload and process PDF lecture notes
- Intelligent text extraction and processing
- Support for mathematical equations and diagrams
- Handles large documents with smart truncation

### 2. AI-Powered Summaries
- Generates concise, comprehensive summaries
- Includes important details, key points, and equations
- Maintains academic context and terminology
- Optimized for student understanding

### 3. Interactive Quiz System
- Automatically generates multiple-choice questions
- Questions test understanding, not just memorization
- Immediate feedback on answers
- Progress tracking through questions
- Clear explanations for correct/incorrect answers

### 4. User-Friendly Interface
- Clean, intuitive Gradio-based UI
- Easy PDF upload and processing
- Real-time feedback and updates
- Mobile-responsive design

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kgampol/WhartonAI.git
cd WhartonAI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the provided URL (typically http://127.0.0.1:7860)

3. Upload your PDF lecture notes

4. Click "Process PDF" to generate:
   - A comprehensive summary
   - Interactive quiz questions

5. Take the quiz:
   - Select your answers
   - Get immediate feedback
   - Navigate through questions
   - See your progress

## Technical Details

### Dependencies
- Python 3.8+
- OpenAI API (GPT-3.5-turbo)
- Gradio for UI
- PyPDF2 for PDF processing
- python-dotenv for environment management

### Architecture
- Modular design with separate components for:
  - PDF processing
  - Text extraction
  - Summary generation
  - Question generation
  - Quiz interface

### Error Handling
- Robust error handling for:
  - PDF processing issues
  - API rate limits
  - Invalid inputs
  - Network connectivity
  - State management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT-3.5-turbo API
- Gradio team for the excellent UI framework
- All contributors who have helped improve this project

---
Last updated: March 2024
