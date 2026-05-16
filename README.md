# AI Chatbot Assistant

A Conversational Retrieval-Augmented Generation (RAG) system built with Python, LangChain, and Google Gemini. This chatbot is designed to let you easily query and interact with the knowledge contained in your local PDF documents.

## Features

- **Document Ingestion**: Automatically loads, splits, and processes PDF files from a local directory (`/data`).
- **Vector Storage**: Uses ChromaDB for efficient embedding storage and similarity search.
- **Context-Aware Responses**: Leverages conversational memory to maintain context throughout the conversation, automatically rephrasing follow-up questions.
- **Source Attribution**: Provides exact source citations (document name and page number) for the generated answers.
- **Google Gemini Integration**: Uses `gemini-2.5-flash-lite` for high-quality, fast, and intelligent responses.

## Requirements

Ensure you have Python installed. The project dependencies are defined in `requirements.txt`. Key dependencies include:
- `langchain`, `langchain-community`, `langchain-chroma`
- `langchain-google-genai`
- `sentence-transformers`, `chromadb`
- `pymupdf` (for PDF processing)
- `python-dotenv`
- `streamlit`

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Prepare Your Data**
   Place your PDF files inside the `/data` directory.

4. **Ingest Documents**
   Run the ingestion script to process the documents and create the local ChromaDB vector store:
   ```bash
   python ingest.py
   ```

## Usage

You can run the assistant in two ways: via a Web GUI or a Command-Line Interface.

### Web GUI (Streamlit)

To launch the interactive web application with a modern UI, run:
```bash
streamlit run app.py
```
This will open the "ML/DL Tutor" in your default web browser, allowing you to easily type questions, view source attributions, and clear chat history visually.

### Command-Line Interface

To start the chatbot assistant via the terminal, run:
```bash
python chatbot.py
```

**CLI Commands**:
- **Ask a question**: Simply type your prompt and press Enter.
- **Clear Memory**: Type `c` or `clear` to reset the conversation history.
- **Quit**: Type `q`, `quit`, or `exit` to stop the chatbot.

## Architecture & Workflow

1. **Load**: PDFs are loaded using `PyMuPDFLoader`.
2. **Chunk**: Documents are split into smaller chunks for optimal retrieval.
3. **Embed**: Text chunks are embedded and saved to a local Chroma vector database.
4. **Retrieve**: User queries (contextualized using conversation history) are used to perform a similarity search to fetch the most relevant chunks.
5. **Generate**: The retrieved context, history, and query are combined into a prompt and sent to Google Gemini to generate the final, accurate response.
