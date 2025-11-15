# AmbedkarGPT - Retrieval-Augmented Generation Q&A System

```
═══════════════════════════════════════════════════════════════════
    Intelligent Question-Answering System using RAG Architecture
    Built with LangChain, ChromaDB, and Local LLM (Ollama)
═══════════════════════════════════════════════════════════════════
```

## Project Overview



The system demonstrates a practical implementation of RAG technology, combining vector search with local language models to provide accurate, context-grounded answers. Unlike traditional chatbots that might hallucinate information, this system strictly answers based only on the provided source text.

## Technical Architecture

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.8+ | Core programming environment |
| **Framework** | LangChain with LCEL | RAG pipeline orchestration |
| **LLM** | Ollama (Mistral) | Local language model inference |
| **Vector Store** | ChromaDB | Persistent embedding storage |
| **Embeddings** | all-MiniLM-L6-v2 | Semantic text vectorization |
| **ML Backend** | PyTorch | Model inference engine |
| **Validation** | Pydantic | Data schema validation |

### System Design Decisions

The implementation follows several key design principles that were carefully chosen to ensure reliability, performance, and maintainability:

#### 1. Retrieval-Augmented Generation (RAG) Approach

Rather than relying solely on the language model's pre-trained knowledge, this system implements RAG to ground all responses in the actual source text.

```
User Query → Vectorize → Retrieve Top-K Chunks → LLM Generation → Answer + Sources
                ↓              ↑
           Embedding      Vector Database
             Model          (ChromaDB)
```

This approach provides several advantages:

- Prevents hallucination by constraining responses to retrieved context
- Enables citation of source material for verification
- Allows dynamic knowledge base updates without model retraining
- Reduces computational requirements compared to fine-tuning

#### 2. Document Processing Strategy

The text processing pipeline uses a RecursiveCharacterTextSplitter with carefully tuned parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Chunk Size** | 500 chars | Balances context preservation with retrieval precision |
| **Chunk Overlap** | 50 chars | Prevents information loss at chunk boundaries |
| **Separators** | `[". ", "\n\n", "\n", " "]` | Prioritizes natural text boundaries for semantic coherence |

#### 3. Embedding and Retrieval Configuration

The embedding model (all-MiniLM-L6-v2) was selected for its balance of performance and resource efficiency:

| Specification | Value | Benefit |
|--------------|-------|---------|
| **Vector Dimensions** | 384 | Optimal balance of accuracy and speed |
| **Normalization** | L2-normalized | Consistent similarity scoring |
| **Hardware** | CPU-optimized | Flexible deployment without GPU |
| **Retrieval Count** | k=4 | Sufficient context without overwhelming LLM |

The retrieval count of 4 was chosen to provide sufficient context for comprehensive answers without overwhelming the language model's context window.

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

```
Required: Python 3.8+ | Ollama | Mistral Model
```

Before beginning installation, ensure your system meets these requirements:

**1. Python Version**

Python 3.8 or higher must be installed on your system. Verify with:

```bash
python --version
```

**2. Ollama Installation**

Download and install Ollama from [ollama.ai](https://ollama.ai/). Ollama provides a local inference server for running large language models.

**3. Mistral Model**

After installing Ollama, pull the Mistral model:

```bash
ollama pull mistral
```

This downloads the model weights locally. The first download may take several minutes depending on your internet connection.

**4. Verify Ollama Service**

Start the Ollama service (runs on localhost:11434). Test connectivity:

```bash
curl http://localhost:11434/api/tags
```

### Project Installation

```
Steps: Clone → Virtual Env → Install Dependencies → Verify Files
```

**1. Clone the Repository:**
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

```
[1] Environment Validation
    └─ Check Python version & imports

[2] Ollama Connection Test
    └─ Verify server accessibility

[3] Embedding Model Loading
    └─ Initialize sentence-transformer

[4] Document Processing
    ├─ Read speech.txt
    ├─ Split into chunks (500 chars, 50 overlap)
    └─ Generate embeddings

[5] Vector Store Creation
    ├─ Create chroma_db_prod/
    ├─ Store embeddings in ChromaDB
    └─ Persist to disk

[6] QA Chain Initialization
    └─ Setup retrieval pipeline

Duration: ~2-3 minutes (hardware dependent)
```

### Subsequent Runs

| Run Type | Startup Time | Operations |
|----------|-------------|------------|
| First Run | 2-3 minutes | Full initialization + embedding generation |
| Subsequent | 10-20 seconds | Load existing vector database |

On subsequent executions, the system detects the existing vector database and loads it directly, bypassing the document processing stage.

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

**Example 1: Question within scope**

```
> What is the real remedy?

Answer: According to the text, the real remedy is to destroy the caste 
system and promote social equality...

Source excerpts:
  1. ...the real remedy is to destroy the caste system...
  2. ...nothing else will serve as an effective solution...
```

**Example 2: Question outside scope**

```
> What are Dr. Ambedkar's views on modern technology?

Answer: I cannot find this information in the provided text.
```

**Exit Commands:** Type `exit`, `quit`, or `q` to terminate | Press `Ctrl+C` for immediate shutdown

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

### Key Components

| Component | Purpose | Key Feature |
|-----------|---------|-------------|
| **StderrFilter** | Error stream wrapper | Filters ChromaDB telemetry warnings |
| **AppConfig** | Configuration hub | Centralizes all system parameters |
| **initialize_embeddings()** | Embedding factory | Creates HuggingFace model instance |
| **verify_ollama_connection()** | Health check | Validates Ollama server availability |
| **process_document()** | Document processor | Loads and chunks text with UTF-8 encoding |
| **get_chroma_client()** | Database factory | Creates ChromaDB client with telemetry disabled |
| **initialize_vector_store()** | Vector DB manager | Creates or loads existing vector database |
| **build_qa_pipeline()** | LCEL chain builder | Constructs retrieval → prompt → LLM pipeline |

**Component Interactions:**

```
[main] → [verify_ollama_connection] → [initialize_embeddings]
   ↓
[process_document] → [get_chroma_client] → [initialize_vector_store]
   ↓
[build_qa_pipeline] → Interactive Q&A Loop
```

**main() Function:**
Orchestrates the entire application flow with proper error handling at each stage. Implements the interactive loop for user queries.

### Configuration Tuning

**Available Parameters in AppConfig:**

| Parameter | Default | Effect of Increase | Effect of Decrease |
|-----------|---------|-------------------|-------------------|
| `CHUNK_SIZE` | 500 | More context per chunk | More precise retrieval |
| `CHUNK_OVERLAP` | 50 | Reduces boundary effects | Reduces redundancy |
| `RETRIEVAL_COUNT` | 4 | More context (slower) | Faster but less context |

**Model Selection:**

```python
# Change LLM model (in build_qa_pipeline function)
llm = ChatOllama(model="llama2")  # or "codellama", "mixtral", etc.

# Change embedding model (in initialize_embeddings function)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```


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
