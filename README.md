# Chat with Your PDF Files using Google Gemini Pro

This project is a Streamlit application that allows users to chat with their PDF files using Google Gemini Pro. The application leverages Google Generative AI Embeddings and ChromaDB for efficient document retrieval and question answering.

## Features

- **PDF File Upload**: Upload multiple PDF files and extract text.
- **Text Preprocessing**: Clean and preprocess extracted text.
- **Text Chunking**: Segment text into manageable chunks for embedding.
- **Google Generative AI Embeddings**: Embed text using Google Generative AI Embeddings.
- **ChromaDB Integration**: Store and retrieve embedded text using ChromaDB.
- **MultiQuery Retriever**: Generate multiple query variations for better document retrieval.
- **Parent Document Retriever**: Retrieve relevant parent documents.
- **Contextual Compression Retriever**: Compress and retrieve contextually relevant documents.

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- SpaCy
- Google Generative AI
- ChromaDB
- dotenv

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chat-with-pdf.git
    cd chat-with-pdf
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure SpaCy model is downloaded:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Configuration

1. Set up your Google API Key in a `.env` file:
    ```env
    GOOGLE_API_KEY=your-google-api-key
    ```

2. Configure ChromaDB connection in `app.py`:
    ```python
    configuration = {
        "client": "PersistentClient",
        "path": "/tmp/.chroma"
    }
    ```

## Running the Application

Run the Streamlit application:
```bash
streamlit run app.py
