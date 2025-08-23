import streamlit as st
import os
import psutil
import tempfile
import time
import logging
import hashlib
import shutil
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gc

# Enhanced document loaders with better structure preservation
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)

# CRITICAL UPGRADE: Enhanced PDF processing with better layout handling
try:
    import fitz  # PyMuPDF for better PDF processing
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available, falling back to PyPDFLoader")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL UPGRADE: Enhanced PDF loader with table and structure detection
class EnhancedPDFLoader:
    """Enhanced PDF loader that preserves structure, tables, and metadata."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.use_pymupdf = PYMUPDF_AVAILABLE
    
    def load(self) -> List[Document]:
        """Load PDF with enhanced structure preservation."""
        if self.use_pymupdf:
            return self._load_with_pymupdf()
        else:
            # Fallback to standard PyPDFLoader
            loader = PyPDFLoader(self.file_path)
            return loader.load()
    
    def _load_with_pymupdf(self) -> List[Document]:
        """Load PDF using PyMuPDF for better text extraction and structure detection."""
        documents = []
        doc = None
        
        try:
            doc = fitz.open(self.file_path)
            total_pages = len(doc)
            
            # Process all pages first, then close document
            page_data = []
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                # Extract text with layout preservation
                text = page.get_text()
                
                # CRITICAL FIX: Extract tables immediately while page is accessible
                tables = self._extract_tables_from_page(page, page_num)
                
                page_data.append({
                    'text': text,
                    'tables': tables,
                    'page_num': page_num
                })
            
            # Now process the extracted data after closing the document
            doc.close()
            doc = None  # Ensure we don't try to use it again
            
            # Create documents from extracted data
            for page_info in page_data:
                page_num = page_info['page_num']
                text = page_info['text']
                tables = page_info['tables']
                
                # Enhanced metadata with structure info
                metadata = {
                    'source': os.path.basename(self.file_path),
                    'page': page_num + 1,
                    'total_pages': total_pages,
                    'has_tables': len(tables) > 0,
                    'table_count': len(tables),
                    'extraction_method': 'pymupdf'
                }
                
                # Create main text document
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata=metadata.copy()
                    ))
                
                # CRITICAL: Create separate documents for tables with enhanced context
                for i, table_data in enumerate(tables):
                    table_metadata = metadata.copy()
                    table_metadata.update({
                        'content_type': 'table',
                        'table_index': i,
                        'table_context': f"Table {i+1} from page {page_num + 1}"
                    })
                    
                    documents.append(Document(
                        page_content=table_data,
                        metadata=table_metadata
                    ))
            
            logger.info(f"Enhanced PDF processing: {len(documents)} documents extracted from {total_pages} pages")
            
        except Exception as e:
            logger.error(f"PyMuPDF processing failed: {e}, falling back to PyPDFLoader")
            # Ensure document is closed on error
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
            # Fallback to standard loader
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
        
        return documents
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[str]:
        """Extract tables from a PDF page."""
        tables = []
        try:
            # Check if page is still valid before accessing
            if page is None:
                return tables
                
            # Find tables using PyMuPDF's table detection
            table_list = page.find_tables()
            
            for table_idx, table in enumerate(table_list):
                try:
                    # Extract table data with error handling
                    table_data = table.extract()
                    
                    if table_data and len(table_data) > 0:
                        # Convert table to readable format
                        table_text = self._format_table_data(table_data, page_num, table_idx)
                        if table_text:
                            tables.append(table_text)
                except Exception as table_error:
                    logger.warning(f"Failed to extract table {table_idx} from page {page_num}: {table_error}")
                    continue
        
        except Exception as e:
            logger.warning(f"Table extraction failed for page {page_num}: {e}")
        
        return tables
    
    def _format_table_data(self, table_data: List[List], page_num: int, table_idx: int) -> str:
        """Format extracted table data into readable text."""
        if not table_data:
            return ""
        
        try:
            # Filter out empty rows and cells
            cleaned_data = []
            for row in table_data:
                if row and any(cell and str(cell).strip() for cell in row):
                    cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                    cleaned_data.append(cleaned_row)
            
            if not cleaned_data:
                return ""
            
            # Create DataFrame for better formatting
            try:
                df = pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0] if len(cleaned_data) > 1 else None)
                # Clean up the data
                df = df.fillna('')
                
                # Format as text with context
                table_text = f"TABLE {table_idx + 1} (Page {page_num + 1}):\n"
                table_text += df.to_string(index=False, na_rep='')
                table_text += f"\n[End of Table {table_idx + 1}]\n"
                
                return table_text
            except Exception:
                # Fallback to simple formatting if DataFrame creation fails
                table_text = f"TABLE {table_idx + 1} (Page {page_num + 1}):\n"
                for row in cleaned_data:
                    table_text += " | ".join(str(cell) for cell in row if cell) + "\n"
                table_text += f"[End of Table {table_idx + 1}]\n"
                return table_text
        
        except Exception as e:
            logger.warning(f"Table formatting failed: {e}")
            # Final fallback
            table_text = f"TABLE {table_idx + 1} (Page {page_num + 1}):\n"
            try:
                for row in table_data:
                    if row:
                        table_text += " | ".join(str(cell) for cell in row if cell) + "\n"
            except:
                table_text += "[Table data could not be formatted]\n"
            table_text += f"[End of Table {table_idx + 1}]\n"
            return table_text
        
        
# CRITICAL UPGRADE: Hierarchical text splitter that preserves document structure
class HierarchicalTextSplitter:
    """Enhanced text splitter that preserves document structure and context."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Different splitting strategies for different content types
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            add_start_index=True
        )
        
        # Specialized splitter for technical content
        self.technical_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size + 200,  # Slightly larger for technical content
            chunk_overlap=chunk_overlap + 50,
            length_function=len,
            separators=[
                "\n\n### ",  # Markdown sections
                "\n## ",     # Markdown subsections
                "\n# ",      # Markdown headers
                "\n\n",      # Paragraphs
                "\n",        # Lines
                ". ",        # Sentences
                " "          # Words
            ],
            add_start_index=True
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents while preserving structure and context."""
        split_docs = []
        
        for doc in documents:
            # Determine content type and use appropriate splitter
            content_type = doc.metadata.get('content_type', 'text')
            
            if content_type == 'table':
                # Tables need special handling - don't split them
                split_docs.append(doc)
            elif self._is_technical_content(doc.page_content):
                # Use technical splitter for complex content
                chunks = self.technical_splitter.split_documents([doc])
                split_docs.extend(self._enhance_chunk_metadata(chunks, doc))
            else:
                # Use standard splitter for regular text
                chunks = self.text_splitter.split_documents([doc])
                split_docs.extend(self._enhance_chunk_metadata(chunks, doc))
        
        return split_docs
    
    def _is_technical_content(self, text: str) -> bool:
        """Detect if content is technical/scientific in nature."""
        technical_indicators = [
            'table', 'figure', 'equation', 'section', '§', 
            'cfr', 'ieee', 'asme', 'regulatory', 'compliance',
            'safety', 'nuclear', 'reactor', 'mw', 'temperature',
            'pressure', 'specification', 'requirement'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in technical_indicators)
    
    def _enhance_chunk_metadata(self, chunks: List[Document], original_doc: Document) -> List[Document]:
        """Enhance chunk metadata with hierarchical information."""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Preserve all original metadata
            chunk.metadata.update(original_doc.metadata)
            
            # Add chunk-specific metadata
            chunk.metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_id': f"{original_doc.metadata.get('source', 'unknown')}_{i}",
                'parent_doc_id': id(original_doc)
            })
            
            # CRITICAL: Add content classification
            chunk.metadata['content_classification'] = self._classify_content(chunk.page_content)
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _classify_content(self, text: str) -> str:
        """Classify content type for better retrieval."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['table', 'column', 'row', '|']):
            return 'tabular'
        elif any(word in text_lower for word in ['figure', 'diagram', 'chart', 'graph']):
            return 'visual_reference'
        elif any(word in text_lower for word in ['procedure', 'step', 'instruction']):
            return 'procedural'
        elif any(word in text_lower for word in ['regulation', 'requirement', 'shall', 'must']):
            return 'regulatory'
        elif any(word in text_lower for word in ['abstract', 'summary', 'conclusion']):
            return 'summary'
        else:
            return 'general'

# Rest of the original functions with enhancements...

def get_temp_directory():
    """Create platform-independent temporary directory."""
    return tempfile.mkdtemp(prefix="rag_temp_")

def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory safely."""
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Could not clean up temp directory {temp_dir}: {e}")

def get_file_hash(uploaded_file) -> str:
    """Generate a unique hash for an uploaded file based on name and content."""
    try:
        content = uploaded_file.getvalue()
        file_data = f"{uploaded_file.name}_{len(content)}_{hashlib.md5(content).hexdigest()}"
        return hashlib.sha256(file_data.encode()).hexdigest()[:16]
    except Exception as e:
        logger.warning(f"Failed to generate hash for {uploaded_file.name}: {e}")
        return f"{uploaded_file.name}_{int(time.time())}"

def validate_model_config(model_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate model configuration parameters."""
    errors = []
    provider = model_config.get("provider")
    
    if not provider:
        errors.append("No provider specified")
        return False, errors
    
    if provider == "Ollama (Local)":
        if not all(k in model_config for k in ["llm_model", "embedding_model", "base_url"]):
            errors.append("Ollama configuration is incomplete.")
    elif provider in ["OpenAI", "Google", "Custom (OpenAI-Compatible)"]:
        if not model_config.get("api_key") or len(model_config.get("api_key", "").strip()) < 10:
            errors.append("API key is missing or invalid.")
    
    return len(errors) == 0, errors

@st.cache_resource(show_spinner="Initializing AI models...")
def get_models(model_config: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """Initialize LLM and embedding models based on configuration."""
    is_valid, errors = validate_model_config(model_config)
    if not is_valid:
        for error in errors:
            st.error(f"Model Configuration Error: {error}")
        return None, None
    
    provider = model_config.get("provider")
    
    try:
        if provider == "Ollama (Local)":
            llm = OllamaLLM(model=model_config["llm_model"], base_url=model_config["base_url"])
            embeddings = OllamaEmbeddings(model=model_config["embedding_model"], base_url=model_config["base_url"])
        elif provider == "OpenAI":
            llm = ChatOpenAI(model=model_config["llm_model"], api_key=model_config["api_key"])
            embeddings = OpenAIEmbeddings(model=model_config["embedding_model"], api_key=model_config["api_key"])
        elif provider == "Google":
            llm = ChatGoogleGenerativeAI(model=model_config["llm_model"], google_api_key=model_config["api_key"])
            embeddings = GoogleGenerativeAIEmbeddings(model=model_config["embedding_model"], google_api_key=model_config["api_key"])
        elif provider == "Custom (OpenAI-Compatible)":
            llm = ChatOpenAI(model=model_config["llm_model"], api_key=model_config["api_key"], base_url=model_config["base_url"])
            embeddings = OpenAIEmbeddings(model=model_config["embedding_model"], api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        logger.info(f"Successfully initialized models for provider: {provider}")
        return llm, embeddings
    
    except Exception as e:
        st.error(f"Error initializing models from {provider}: {str(e)}")
        return None, None

def get_appropriate_loader(file_path: str, file_name: str):
    """ENHANCED: Get the most appropriate document loader with better structure preservation."""
    file_ext = os.path.splitext(file_name.lower())[1]
    
    try:
        if file_ext == '.pdf':
            # CRITICAL UPGRADE: Use enhanced PDF loader for better structure extraction
            return EnhancedPDFLoader(file_path)
        elif file_ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        elif file_ext == '.csv':
            return CSVLoader(file_path, encoding='utf-8')
        elif file_ext in ['.xlsx', '.xls']:
            return UnstructuredExcelLoader(file_path)
        elif file_ext in ['.docx', '.doc']:
            return UnstructuredWordDocumentLoader(file_path)
        elif file_ext in ['.md', '.markdown']:
            return UnstructuredMarkdownLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)
    except ImportError as e:
        logger.warning(f"Failed to import loader for {file_ext}: {e}")
        return TextLoader(file_path, encoding='utf-8')

def get_optimized_text_splitter() -> HierarchicalTextSplitter:
    """ENHANCED: Get hierarchical text splitter that preserves document structure."""
    return HierarchicalTextSplitter(
        chunk_size=1200,  # Slightly larger for better context
        chunk_overlap=250  # More overlap for technical content
    )

def batch_iterable(iterable, batch_size=2):
    """Split iterable into batches to manage memory usage."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def process_single_file(uploaded_file, temp_dir: str, text_splitter) -> Tuple[List[Document], Dict[str, Any]]:
    """ENHANCED: Process a single file with better structure preservation."""
    file_details = {
        'chunks': [],
        'processing_time': 0,
        'loader_type': 'unknown',
        'success': False,
        'error': None,
        'tables_extracted': 0,
        'content_types': {}
    }
    
    try:
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:
            logger.warning(f"High memory usage ({memory_percent:.1f}%) before processing {uploaded_file.name}")
            gc.collect()
        
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        start_time = time.time()
        
        # Get enhanced loader
        loader = get_appropriate_loader(temp_file_path, uploaded_file.name)
        file_details['loader_type'] = type(loader).__name__
        
        # Load documents with enhanced processing
        try:
            documents = loader.load()
        except Exception as load_error:
            logger.warning(f"Primary loader failed for {uploaded_file.name}, trying fallback: {load_error}")
            loader = UnstructuredFileLoader(temp_file_path)
            documents = loader.load()
            file_details['loader_type'] = 'UnstructuredFileLoader (fallback)'
        
        # CRITICAL: Count tables and content types
        tables_count = sum(1 for doc in documents if doc.metadata.get('content_type') == 'table')
        file_details['tables_extracted'] = tables_count
        
        # Enhance metadata for all documents BEFORE splitting
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            
            doc.metadata.update({
                'source': uploaded_file.name,
                'file_size': len(uploaded_file.getvalue()),
                'file_type': os.path.splitext(uploaded_file.name)[1],
                'document_index': i,
                'total_documents': len(documents),
                'processing_enhanced': True  # Flag for enhanced processing
            })
            
            # Enhanced page/section handling
            if uploaded_file.name.lower().endswith('.pdf'):
                if 'page' not in doc.metadata and i is not None:
                    doc.metadata['page'] = i + 1
                elif 'page' in doc.metadata:
                    try:
                        doc.metadata['page'] = int(doc.metadata['page']) + 1 if int(doc.metadata['page']) == i else int(doc.metadata['page'])
                    except (ValueError, TypeError):
                        doc.metadata['page'] = i + 1
            elif 'page' not in doc.metadata and len(documents) > 1:
                doc.metadata['page'] = f"Section {i + 1}"
            elif 'page' not in doc.metadata:
                doc.metadata['page'] = 1
        
        # Split documents using enhanced splitter
        chunked_docs = text_splitter.split_documents(documents)
        
        # CRITICAL: Analyze content types
        content_types = {}
        for chunk in chunked_docs:
            content_type = chunk.metadata.get('content_classification', 'general')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        processing_time = time.time() - start_time
        
        file_details.update({
            'chunks': chunked_docs,
            'processing_time': processing_time,
            'success': True,
            'original_docs': len(documents),
            'final_chunks': len(chunked_docs),
            'content_types': content_types
        })
        
        logger.info(f"Enhanced processing {uploaded_file.name}: {len(documents)} docs → {len(chunked_docs)} chunks, {tables_count} tables in {processing_time:.2f}s")
        
        return chunked_docs, file_details
        
    except Exception as e:
        error_msg = f"Error processing {uploaded_file.name}: {str(e)}"
        logger.error(error_msg)
        file_details.update({
            'error': error_msg,
            'processing_time': time.time() - start_time if 'start_time' in locals() else 0
        })
        return [], file_details

# Rest of the functions remain the same...
def process_files_in_batches(uploaded_files, embeddings, batch_size=2):
    """Process files in small batches to prevent memory crashes."""
    processing_details = {}
    vector_store = st.session_state.get("vector_store")
    text_splitter = get_optimized_text_splitter()
    
    total_batches = (len(uploaded_files) + batch_size - 1) // batch_size
    
    for batch_num, batch_files in enumerate(batch_iterable(uploaded_files, batch_size), 1):
        st.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files) with enhanced structure detection...")
        
        temp_dir = get_temp_directory()
        batch_docs = []
        
        try:
            for uploaded_file in batch_files:
                docs, file_details = process_single_file(uploaded_file, temp_dir, text_splitter)
                
                if docs:
                    batch_docs.extend(docs)
                    processing_details[uploaded_file.name] = file_details
                else:
                    processing_details[uploaded_file.name] = file_details.get('error', 'Processing failed')
            
            if batch_docs:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch_docs, embeddings)
                    st.session_state.vector_store = vector_store
                else:
                    try:
                        vector_store.add_documents(batch_docs)
                    except Exception as e:
                        logger.warning(f"Could not add to existing vector store: {e}. Creating new one.")
                        vector_store = FAISS.from_documents(batch_docs, embeddings)
                        st.session_state.vector_store = vector_store
            
            gc.collect()
            time.sleep(0.5)
        
        finally:
            cleanup_temp_directory(temp_dir)
    
    return vector_store, {}, processing_details

def validate_uploaded_files(uploaded_files) -> Tuple[bool, List[str]]:
    """Validate uploaded files for processing."""
    warnings = []
    max_file_size = 100 * 1024 * 1024  # 100MB
    
    supported_extensions = {'.pdf', '.txt', '.csv', '.xlsx', '.xls', '.docx', '.doc', '.md', '.markdown'}
    
    for file in uploaded_files:
        if file.size > max_file_size:
            warnings.append(f"File '{file.name}' exceeds 100MB limit.")
        
        file_ext = os.path.splitext(file.name.lower())[1]
        if file_ext not in supported_extensions:
            warnings.append(f"File '{file.name}' has unsupported format '{file_ext}'. Supported: {', '.join(supported_extensions)}")
    
    return len(warnings) == 0, warnings

def get_processed_files_info() -> Dict[str, Any]:
    """Get information about already processed files from session state."""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}
    return st.session_state.processed_files

def update_processed_files_info(file_hash: str, file_name: str, doc_count: int):
    """Update the processed files information."""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}
    
    st.session_state.processed_files[file_hash] = {
        "name": file_name,
        "doc_count": doc_count,
        "processed_time": time.time()
    }

def remove_documents_from_vector_store(vector_store, removed_file_names: List[str], embeddings) -> Optional[Any]:
    """Remove documents from vector store by source file names."""
    try:
        if not vector_store or not removed_file_names:
            return vector_store
        
        remaining_docs = []
        if hasattr(vector_store, 'docstore') and hasattr(vector_store.docstore, '_dict'):
            for doc_id, doc in vector_store.docstore._dict.items():
                if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                    if doc.metadata['source'] not in removed_file_names:
                        remaining_docs.append(doc)
        
        if remaining_docs:
            logger.info(f"Rebuilding vector store: keeping {len(remaining_docs)} documents")
            new_vector_store = FAISS.from_documents(documents=remaining_docs, embedding=embeddings)
            return new_vector_store
        else:
            logger.info("No documents remain after removal")
            return None
    
    except Exception as e:
        logger.error(f"Error removing documents from vector store: {e}")
        return vector_store

def process_uploaded_files_incremental(
    uploaded_files,
    embeddings,
    model_config: Dict[str, Any] = None
) -> Tuple[Optional[Any], Dict[str, Any], Dict[str, Any]]:
    """ENHANCED: Main function with advanced document understanding capabilities."""
    if not uploaded_files or embeddings is None:
        return None, {}, {}

    try:
        is_valid, warnings = validate_uploaded_files(uploaded_files)
        if not is_valid:
            for warning in warnings:
                st.warning(warning)
            st.error("File validation failed.")
            return None, {}, {}

        current_file_hashes = {get_file_hash(f): f for f in uploaded_files}
        processed_files_info = get_processed_files_info()
        
        files_to_process = []
        for file_hash, uploaded_file in current_file_hashes.items():
            if file_hash not in processed_files_info:
                files_to_process.append((file_hash, uploaded_file))

        if not files_to_process and st.session_state.get("vector_store"):
            logger.info("No new files to process, returning existing vector store")
            return st.session_state.vector_store, {}, st.session_state.get("processing_details", {})

        try:
            files_only = [uploaded_file for _, uploaded_file in files_to_process]
            
            vector_store, _, processing_details = process_files_in_batches(
                files_only, embeddings, batch_size=2
            )

            for file_hash, uploaded_file in files_to_process:
                file_name = uploaded_file.name
                if file_name in processing_details:
                    file_detail = processing_details[file_name]
                    if isinstance(file_detail, dict) and file_detail.get('success'):
                        chunks_count = len(file_detail.get('chunks', []))
                        update_processed_files_info(file_hash, file_name, chunks_count)
                    else:
                        update_processed_files_info(file_hash, file_name, 0)

            logger.info(f"Enhanced processing completed: {len(files_to_process)} new files with advanced document understanding")
            return vector_store, {}, processing_details

        except Exception as e:
            logger.error(f"Enhanced batch processing failed: {e}")
            st.error(f"Document processing failed: {str(e)}")
            return None, {}, {"error": f"Enhanced processing failed: {str(e)}"}

    except Exception as e:
        logger.error(f"Critical error in enhanced processing: {e}")
        st.error(f"Critical processing error: {str(e)}")
        return None, {}, {"error": f"Critical processing error: {str(e)}"}