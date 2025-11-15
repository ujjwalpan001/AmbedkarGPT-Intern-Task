
import os
import sys
import warnings
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from chromadb.config import Settings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class StderrFilter:
    # Filters ChromaDB telemetry warnings from stderr output
    def __init__(self, original_stream):
        self.stream = original_stream
    
    def write(self, message):
        if 'telemetry' not in message.lower() and 'capture()' not in message:
            self.stream.write(message)
    
    def flush(self):
        self.stream.flush()
    
    def fileno(self):
        return self.stream.fileno()


# Suppress warnings and telemetry before importing libraries
sys.stderr = StderrFilter(sys.stderr)
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

try:
    from chromadb.telemetry.posthog import Posthog
    Posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass


class AppConfig:
    # Application configuration and constants
    SOURCE_FILE = "speech.txt"
    DB_PATH = "./chroma_db_prod"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_NAME = "mistral"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    RETRIEVAL_COUNT = 4
    COLLECTION_NAME = "ambedkar_speech"


class Colors:
    # ANSI color codes for terminal output
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def initialize_embeddings():
    # Initialize and return the HuggingFace embedding model
    return HuggingFaceEmbeddings(
        model_name=AppConfig.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def verify_ollama_connection(language_model):
    # Test connection to Ollama server
    print(f"{Colors.YELLOW}Verifying Ollama connection ({AppConfig.LLM_NAME})...{Colors.RESET}")
    try:
        language_model.invoke("test")
        print(f"{Colors.GREEN}Connected to Ollama server{Colors.RESET}")
        return True
    except Exception as error:
        print(f"\n{Colors.RED}{Colors.BOLD}Error: Cannot reach Ollama server{Colors.RESET}")
        print(f"{Colors.YELLOW}Ensure Ollama is running and model '{AppConfig.LLM_NAME}' is available{Colors.RESET}")
        print(f"Error details: {error}")
        return False


def process_document(file_path):
    # Load document from file and split into chunks
    if not os.path.exists(file_path):
        print(f"{Colors.RED}File not found: {file_path}{Colors.RESET}")
        return None
    
    try:
        document_loader = TextLoader(file_path, encoding='utf-8')
        documents = document_loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
            separators=[". ", "\n\n", "\n", " ", ""]
        )
        document_chunks = text_splitter.split_documents(documents)
        
        print(f"{Colors.GREEN}Processed {len(document_chunks)} text chunks{Colors.RESET}")
        return document_chunks
    except Exception as error:
        print(f"{Colors.RED}Document processing failed: {error}{Colors.RESET}")
        return None


def get_chroma_client():
    # Create and return ChromaDB persistent client with telemetry disabled
    db_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
    return chromadb.PersistentClient(path=AppConfig.DB_PATH, settings=db_settings)


def initialize_vector_store(document_chunks, embedding_model, is_new=True):
    # Create or load vector store with document embeddings
    action = "Creating" if is_new else "Loading"
    print(f"{Colors.YELLOW}{action} vector store...{Colors.RESET}")
    
    db_client = get_chroma_client()
    vector_db = Chroma(
        client=db_client,
        embedding_function=embedding_model,
        collection_name=AppConfig.COLLECTION_NAME
    )
    
    if is_new and document_chunks:
        vector_db.add_documents(document_chunks)
        print(f"{Colors.GREEN}Vector store created and persisted{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}Vector store loaded from disk{Colors.RESET}")
    
    return vector_db


def build_qa_pipeline(vector_store, language_model):
    # Construct the retrieval-augmented generation chain
    print(f"{Colors.YELLOW}Building QA pipeline...{Colors.RESET}")
    
    system_prompt = """You are a helpful assistant answering questions only from Dr. B.R. Ambedkar's speech.
Use ONLY the context provided below.
If the answer is not found in the context, you MUST reply:
"I cannot find this information in the provided text."

<context>
{context}
</context>

Question: {input}

Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(system_prompt)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": AppConfig.RETRIEVAL_COUNT}
    )
    
    combine_docs_chain = create_stuff_documents_chain(language_model, prompt_template)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    print(f"{Colors.GREEN}QA pipeline ready{Colors.RESET}")
    return rag_chain


def main():
    # Main application entry point
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}")
    print("              AmbedkarGPT - Speech Q&A System")
    print(f"{'='*70}{Colors.RESET}\n")
    
    if sys.version_info < (3, 8):
        print(f"{Colors.RED}Python 3.8+ required{Colors.RESET}")
        sys.exit(1)
    
    # Initialize language model and verify connectivity
    llm = Ollama(model=AppConfig.LLM_NAME, temperature=0.1)
    if not verify_ollama_connection(llm):
        sys.exit(1)
    
    # Setup embedding model
    embeddings = initialize_embeddings()
    
    # Initialize or load vector database
    try:
        if os.path.exists(AppConfig.DB_PATH):
            vector_store = initialize_vector_store(None, embeddings, is_new=False)
        else:
            chunks = process_document(AppConfig.SOURCE_FILE)
            if not chunks:
                sys.exit(1)
            vector_store = initialize_vector_store(chunks, embeddings, is_new=True)
    except Exception as error:
        print(f"{Colors.RED}Vector store initialization failed: {error}{Colors.RESET}")
        sys.exit(1)
    
    # Build QA chain
    qa_pipeline = build_qa_pipeline(vector_store, llm)
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}System ready. Enter your questions below.{Colors.RESET}")
    print(f"{Colors.YELLOW}Type 'exit', 'quit', or 'q' to terminate{Colors.RESET}\n")
    
    # Interactive query loop
    try:
        while True:
            query = input(f"{Colors.BLUE}{Colors.BOLD}> {Colors.RESET}").strip()
            
            if query.lower() in ['q', 'quit', 'exit']:
                print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
                break
            
            if not query:
                continue
            
            response = qa_pipeline.invoke({"input": query})
            answer_text = response['answer']
            context_docs = response['context']
            
            print(f"\n{Colors.GREEN}{Colors.BOLD}Answer: {answer_text}{Colors.RESET}")
            
            if context_docs:
                print(f"\n{Colors.YELLOW}Source excerpts:{Colors.RESET}")
                for idx, document in enumerate(context_docs, 1):
                    excerpt = document.page_content.replace("\n", " ")[:150]
                    print(f"  {idx}. {excerpt}...")
            
            print(f"\n{'-'*70}\n")
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
    except Exception as error:
        print(f"\n{Colors.RED}Unexpected error: {error}{Colors.RESET}")
    finally:
        print(f"{Colors.BLUE}Shutdown complete{Colors.RESET}")


if __name__ == "__main__":
    main()