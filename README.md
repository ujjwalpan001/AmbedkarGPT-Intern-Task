```markdown
# AmbedkarGPT - Retrieval-Augmented Generation Q&A System

## Project Overview

This is a production-ready question-answering system built for the Kalpit Pvt Ltd AI Intern assignment. The system implements a complete Retrieval-Augmented Generation (RAG) pipeline to answer questions exclusively from Dr. B.R. Ambedkar's speech text. The architecture is designed to run entirely offline without requiring API keys or external services, making it suitable for environments with strict data privacy requirements.

## Technical Architecture

### Core Technologies

**Programming Language:** Python 3.8+

**Framework:** LangChain with LCEL (LangChain Expression Language) for modern chain composition

**Large Language Model:** Ollama running the Mistral model locally

**Vector Database:** ChromaDB with persistent storage

**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 from HuggingFace

**Additional Libraries:** PyTorch for model inference, Pydantic for data validation

### System Design Decisions

The implementation follows several key design principles that were carefully chosen to ensure reliability, performance, and maintainability:

#### 1. Retrieval-Augmented Generation (RAG) Approach

Rather than relying solely on the language model's pre-trained knowledge, this system implements RAG to ground all responses in the actual source text. This approach provides several advantages:

- Prevents hallucination by constraining responses to retrieved context
- Enables citation of source material for verification
- Allows dynamic knowledge base updates without model retraining
- Reduces computational requirements compared to fine-tuning

#### 2. Document Processing Strategy

The text processing pipeline uses a RecursiveCharacterTextSplitter with carefully tuned parameters:

- **Chunk Size:** 500 characters - This size balances context preservation with retrieval precision. Smaller chunks would lose context, while larger chunks would reduce retrieval accuracy.

- **Chunk Overlap:** 50 characters - Ensures important information at chunk boundaries is not lost, improving answer completeness.

- **Separator Hierarchy:** [". ", "\n\n", "\n", " ", ""] - This ordering prioritizes natural text boundaries like sentences and paragraphs, maintaining semantic coherence within chunks.

#### 3. Embedding and Retrieval Configuration

The embedding model (all-MiniLM-L6-v2) was selected for its balance of performance and resource efficiency:

- Produces 384-dimensional vectors
- Normalized embeddings for consistent similarity scoring
- CPU-optimized for deployment flexibility
- Proven performance on semantic similarity tasks

Retrieval uses similarity search with k=4, retrieving the four most relevant chunks for each query. This number was chosen to provide sufficient context without overwhelming the language model's context window.

#### 4. Vector Store Implementation

ChromaDB was chosen for its lightweight design and persistent storage capabilities:

- Automatic persistence to disk eliminates the need for manual serialization
- SQLite-backed storage ensures ACID compliance
- Efficient similarity search with HNSW indexing
- Minimal memory footprint suitable for resource-constrained environments

The implementation manually instantiates the ChromaDB PersistentClient with explicit settings to ensure telemetry is disabled and configurations are deterministic across runs.

#### 5. Prompt Engineering

The QA chain uses a carefully crafted prompt template that enforces strict adherence to source material:

```
You are a helpful assistant answering questions only from Dr. B.R. Ambedkar's speech.
Use ONLY the context provided below.
If the answer is not found in the context, you MUST reply:
"I cannot find this information in the provided text."
```

This instruction pattern:
- Explicitly restricts the model to provided context
- Defines expected behavior for out-of-scope queries
- Reduces hallucination risk through clear constraints
- Provides consistent behavior across different query types

#### 6. Error Handling and Robustness

The system implements comprehensive error handling at every stage:

- **Connection Validation:** Checks Ollama server availability before initialization
- **File Existence Verification:** Validates source file presence before processing
- **Graceful Degradation:** Catches and reports errors without crashing
- **Clean Shutdown:** Properly closes resources on exit or interruption

A custom stderr filtering mechanism suppresses non-critical telemetry warnings while preserving important error messages, ensuring clean output for production use.

#### 7. Code Organization and Maintainability

The codebase follows software engineering best practices:

- **Configuration Class:** Centralizes all settings in a single Config class for easy modification
- **DRY Principle:** Reusable functions for embedding model creation and common operations
- **Separation of Concerns:** Distinct functions for loading, splitting, storing, and retrieving
- **Type Safety:** Proper exception handling and error propagation
- **Documentation:** Clear docstrings explaining function purposes and behaviors

## Installation and Setup

### Prerequisites

Before beginning installation, ensure your system meets these requirements:

1. **Python Version:** Python 3.8 or higher installed on your system. You can verify this by running:
   ```bash
   python --version
   ```

2. **Ollama Installation:** Download and install Ollama from [ollama.ai](https://ollama.ai/). Ollama provides a local inference server for running large language models.

3. **Mistral Model:** After installing Ollama, pull the Mistral model:
   ```bash
   ollama pull mistral
   ```
   This downloads the model weights locally. The first download may take several minutes depending on your internet connection.

4. **Verify Ollama is Running:** Start the Ollama service. It typically runs on localhost:11434. You can test connectivity:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Project Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd AmbedkarGPT-Intern-Task
   ```

2. **Create Virtual Environment:**
   
   Creating a virtual environment isolates project dependencies from system Python packages:
   
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

3. **Install Dependencies:**
   
   The requirements.txt file contains all necessary packages with pinned versions for reproducibility:
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs:
   - LangChain core libraries for RAG pipeline construction
   - ChromaDB for vector storage
   - Sentence-transformers for text embeddings
   - Ollama Python client for LLM interaction
   - PyTorch for model inference
   - Supporting utilities for text processing

4. **Verify Installation:**
   
   Ensure the speech.txt file containing Dr. Ambedkar's text is present in the project root directory. This file should have been provided with the assignment materials.

## Running the Application

### First Run

When you run the application for the first time:

```bash
python main.py
```

The system will perform the following initialization sequence:

1. **Environment Validation:** Checks Python version compatibility and imports required libraries

2. **Ollama Connection Test:** Verifies that the Ollama server is accessible and responds to requests

3. **Embedding Model Loading:** Initializes the sentence-transformer model for text vectorization

4. **Document Processing:** 
   - Reads speech.txt file
   - Splits text into semantic chunks
   - Generates embeddings for each chunk

5. **Vector Store Creation:** 
   - Creates the chroma_db_prod directory
   - Stores chunk embeddings in ChromaDB
   - Persists database to disk

6. **QA Chain Initialization:** Sets up the retrieval and generation pipeline

This first run may take 2-3 minutes depending on your hardware as models are loaded and embeddings are computed.

### Subsequent Runs

On subsequent executions, the system detects the existing vector database and loads it directly, bypassing the document processing stage. This reduces startup time to approximately 10-20 seconds.

### Asking Questions

Once initialized, you will see the prompt:

```
System ready! Ask questions below.
Commands: 'exit', 'quit', 'q'

> 
```

Enter your question and press Enter. The system will:
1. Vectorize your query
2. Retrieve relevant text chunks from the database
3. Send chunks and query to the LLM
4. Display the generated answer
5. Show source snippets for verification

### Example Interactions

**Question within scope:**
```
> What is the real remedy?

A: According to the text, the real remedy is to destroy the caste system...

Sources:
  1. ...the real remedy is to destroy the caste system. Nothing else will serve...
```

**Question outside scope:**
```
> What are Dr. Ambedkar's views on modern technology?

A: I cannot find this information in the provided text.
```

### Exiting the Application

To terminate the program, type any of: `exit`, `quit`, or `q`

Alternatively, press Ctrl+C for immediate termination.

## Technical Implementation Details

### File Structure

```
AmbedkarGPT-Intern-Task/
├── main.py                  # Main application code
├── requirements.txt         # Python dependencies
├── speech.txt              # Source document
├── README.md               # This file
└── chroma_db_prod/         # Generated vector database (after first run)
    ├── chroma.sqlite3      # SQLite database file
    └── [collection_id]/    # Vector storage
```

### Key Components Explanation

**FilteredStderr Class:**
Implements a custom standard error stream wrapper that filters out ChromaDB telemetry warnings. This provides clean output while preserving important error messages.

**Config Class:**
Centralizes all configuration parameters including file paths, model names, chunking parameters, and retrieval settings. Modifying system behavior only requires editing this class.

**get_embedding_model() Function:**
Factory function for creating the HuggingFace embedding model with consistent configuration. Implements the DRY (Don't Repeat Yourself) principle by centralizing model instantiation.

**check_ollama_connection() Function:**
Performs a health check on the Ollama server before processing begins. Prevents cryptic errors by validating connectivity upfront.

**load_and_split() Function:**
Handles document loading with UTF-8 encoding and implements the chunking strategy. Returns None on failure to enable proper error handling.

**create_vector_store() Function:**
Manually instantiates ChromaDB PersistentClient with explicit settings to ensure telemetry is disabled and persistence is configured correctly. Creates the vector store and populates it with document embeddings.

**load_existing_store() Function:**
Reconnects to an existing ChromaDB database using the same client configuration as creation. Enables fast restarts by skipping document processing.

**setup_qa_chain() Function:**
Constructs the LCEL chain by combining the retriever, prompt template, and LLM. Uses create_retrieval_chain for proper document passing and create_stuff_documents_chain for context formatting.

**main() Function:**
Orchestrates the entire application flow with proper error handling at each stage. Implements the interactive loop for user queries.

### Configuration Parameters

You can modify these parameters in the Config class to tune system behavior:

- **CHUNK_SIZE:** Increase for more context per chunk, decrease for more precise retrieval
- **CHUNK_OVERLAP:** Increase to reduce boundary effects, decrease to reduce redundancy
- **SEARCH_K:** Number of chunks retrieved per query (higher = more context but slower)
- **LLM_MODEL:** Change to use different Ollama models (llama2, codellama, etc.)
- **EMBEDDING_MODEL:** Switch to different sentence-transformers models for different language support


## Design Rationale Summary

This implementation prioritizes:

1. **Accuracy over speed:** Multiple retrieval chunks and explicit prompt constraints ensure correct answers
2. **Reproducibility:** Pinned dependency versions and fixed random seeds ensure consistent behavior
3. **Privacy:** Complete offline operation with no external API calls or telemetry
4. **Maintainability:** Clean code structure with clear separation of concerns
5. **Production readiness:** Comprehensive error handling and graceful degradation
6. **Resource efficiency:** Optimized for CPU-only environments without requiring GPU acceleration

## Potential Improvements

Given additional time, the following enhancements could be implemented:

1. **Hybrid Search:** Combine semantic similarity with keyword-based BM25 scoring for improved retrieval
2. **Query Expansion:** Automatically generate related queries to capture more relevant context
3. **Reranking:** Add a cross-encoder reranking stage to improve retrieval precision
4. **Caching:** Implement query result caching for frequently asked questions
5. **Logging:** Add structured logging for debugging and monitoring
6. **Web Interface:** Build a Flask or FastAPI web UI for easier accessibility
7. **Batch Processing:** Support processing multiple questions from a file
8. **Evaluation Metrics:** Implement automated testing with ground truth Q&A pairs

## Conclusion

This project demonstrates a complete end-to-end implementation of a RAG-based question answering system following industry best practices. The architecture balances accuracy, performance, and maintainability while remaining accessible for deployment in resource-constrained environments. All design decisions are driven by practical considerations and documented for transparency.    