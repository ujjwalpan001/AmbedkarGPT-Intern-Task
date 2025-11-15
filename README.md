# AmbedkarGPT - Q&A System

A question-answering system built for Dr. B.R. Ambedkar's speeches using Retrieval-Augmented Generation (RAG). The system answers questions based only on the provided text, preventing the LLM from making things up.

## What It Does

Instead of relying on the language model's general knowledge, this system retrieves relevant chunks from the actual speech before generating answers. This grounds responses in the source material and lets you verify answers against the original text.

## Tech Stack

- **Python 3.8+** - Core programming language
- **LangChain with LCEL** - For building the RAG pipeline
- **Ollama (Mistral model)** - Local LLM running on your machine
- **ChromaDB** - Vector database for storing embeddings
- **sentence-transformers (all-MiniLM-L6-v2)** - For converting text to embeddings
- **PyTorch** - Backend for running the embedding model

## How It Works

The system uses a straightforward RAG approach:

1. **Document Processing**: The speech text gets split into chunks of 500 characters with 50 characters overlap. This overlap prevents important information from being cut off at boundaries.

2. **Embedding Creation**: Each chunk is converted into a 384-dimensional vector using the sentence-transformer model. These vectors capture the semantic meaning of the text.

3. **Storage**: All vectors are stored in ChromaDB, which uses SQLite under the hood for persistence. This means you only need to process the document once.

4. **Query Processing**: When you ask a question:
   - Your question gets converted to a vector
   - The system finds the 4 most similar chunks from the database
   - These chunks are sent to the Mistral LLM along with your question
   - The LLM generates an answer based only on those chunks

5. **Constrained Responses**: The prompt explicitly tells the model to only use the provided context. If the answer isn't in the retrieved chunks, it says so instead of making things up.

## Why These Choices?

**Chunk size (500 chars)**: I tested different sizes. Smaller chunks gave more precise retrieval but sometimes lacked context. Larger chunks had the opposite problem. 500 worked well for this speech.

**Retrieval count (k=4)**: Retrieves enough context for comprehensive answers without overwhelming the LLM's context window or slowing down responses.

**Local LLM**: Using Ollama means everything runs on your machine. No API costs, no privacy concerns, and it works offline.

**ChromaDB**: Lightweight, easy to set up, and handles persistence automatically. No need for a separate database server.

## Setup

### Prerequisites

You need Python 3.8 or higher and Ollama installed on your system.

**Install Ollama:**

Download from [ollama.ai](https://ollama.ai/) and install it. Then pull the Mistral model:

```bash
ollama pull mistral
```

The Ollama service should be running on localhost:11434.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AmbedkarGPT-Intern-Task
   ```

2. **Create a virtual environment:**
   
   On Linux/Mac:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Make sure speech.txt is in the project folder** - this contains Dr. Ambedkar's speech text.

## Running the Application

Start the application:

```bash
python main.py
```

**First run takes about 2-3 minutes** because it needs to:
- Load the embedding model
- Process and chunk the speech text
- Generate embeddings for all chunks
- Create the ChromaDB vector database

**Subsequent runs start in 10-20 seconds** since it just loads the existing database.

### Using the System

Once it's ready, you'll see:

```
System ready. Enter your questions below.
Type 'exit', 'quit', or 'q' to terminate

> 
```

Type your question and hit Enter. The system will show you the answer along with source excerpts from the speech.

**Example:**

```
> What is the real remedy?

Answer: According to the text, the real remedy is to destroy the caste system...

Source excerpts:
  1. ...the real remedy is to destroy the caste system...
  2. ...nothing else will serve as an effective solution...
```

If you ask something that's not in the speech, it will tell you:

```
> What are Dr. Ambedkar's views on modern technology?

Answer: I cannot find this information in the provided text.
```

To exit, type `exit`, `quit`, or `q`, or just press Ctrl+C.

## Technical Implementation

### Project Structure

```
AmbedkarGPT-Intern-Task/
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── speech.txt          # Source document (Dr. Ambedkar's speech)
└── README.md           # This file
```

After the first run, a `chroma_db_prod/` folder gets created automatically. This contains the vector database that persists between runs.

### Code Overview

The main.py file has several key functions:

- `initialize_embeddings()` - Sets up the sentence-transformer model
- `verify_ollama_connection()` - Checks if Ollama server is running
- `process_document()` - Loads and splits the speech text into chunks
- `get_chroma_client()` - Creates the ChromaDB client with telemetry disabled
- `initialize_vector_store()` - Creates or loads the vector database
- `build_qa_pipeline()` - Sets up the retrieval chain using LCEL
- `main()` - Runs everything and handles the interactive Q&A loop

All configuration settings are in the `AppConfig` class at the top of main.py.

### Configuration

If you want to tune the system, you can modify these parameters in the `AppConfig` class:

- `CHUNK_SIZE` (default: 500) - Size of each text chunk
- `CHUNK_OVERLAP` (default: 50) - Overlap between chunks
- `RETRIEVAL_COUNT` (default: 4) - Number of chunks to retrieve per query

You can also change the LLM model by modifying the model name in the `build_qa_pipeline()` function. Just make sure you've pulled that model with Ollama first.
## Known Issues

ChromaDB's telemetry can be noisy in the console. I added a custom stderr filter to suppress those warnings while keeping actual errors visible. This doesn't affect functionality, just keeps the output clean.

## What Could Be Better

Given more time, I'd add:

- **Hybrid search** - combining semantic similarity with keyword-based BM25 for better retrieval
- **Query caching** - storing answers to common questions
- **Web interface** - a simple Flask or FastAPI UI instead of command line
- **Reranking** - using a cross-encoder to improve which chunks get sent to the LLM
- **Better evaluation** - automated testing with ground truth Q&A pairs to measure accuracy    
