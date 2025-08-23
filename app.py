import streamlit as st
import nest_asyncio
import os
import time
import gc  # ADDED: For memory management
import psutil  # ADDED: For memory monitoring
import shutil  # ADDED: For cleanup

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import logging
from typing import Optional, Dict, Any
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import ValidationError
import traceback

from utils import get_models, process_uploaded_files_incremental, get_file_hash, remove_documents_from_vector_store
from agents import setup_multi_agent_system, FinalAnswer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply the asyncio patch
nest_asyncio.apply()

st.set_page_config(
    page_title="Nuclear AI Chatbot",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CRITICAL: Memory safety functions
def get_memory_info():
    """Get current memory usage information."""
    try:
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
    except:
        return {'percent': 0, 'available_gb': 0, 'used_gb': 0, 'total_gb': 8}

def check_memory_safety(uploaded_files, max_total_mb=150):
    """Prevent memory crashes by limiting total upload size."""
    if not uploaded_files:
        return True, None
    
    total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
    memory_info = get_memory_info()
    available_gb = memory_info['available_gb']
    
    if total_size_mb > max_total_mb:
        return False, f"⚠️ Total upload size ({total_size_mb:.1f}MB) exceeds safe limit ({max_total_mb}MB). Please upload fewer/smaller files to prevent crashes."
    
    if available_gb < 1.0:
        return False, f"⚠️ Insufficient memory ({available_gb:.1f}GB available). Close other applications or upload smaller files."
    
    return True, None

def clear_uploaded_files_memory():
    """Clear uploaded file memory after processing."""
    try:
        # Clear various file-related session state keys
        keys_to_clear = ['uploaded_files', 'document_uploader', 'file_uploader']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Force garbage collection
        gc.collect()
        logger.info("Cleared uploaded file memory")
    except Exception as e:
        logger.warning(f"Could not clear file memory: {e}")

def display_memory_status():
    """Enhanced memory status with warnings."""
    memory_info = get_memory_info()
    memory_color = "🔴" if memory_info['percent'] > 80 else "🟡" if memory_info['percent'] > 60 else "🟢"
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Memory Status")
    st.sidebar.write(f"{memory_color} RAM Usage: {memory_info['percent']:.1f}%")
    st.sidebar.write(f"Available: {memory_info['available_gb']:.1f} GB / {memory_info['total_gb']:.1f} GB")
    
    # Enhanced warnings
    if memory_info['percent'] > 85:
        st.sidebar.error("🚨 Critical memory usage! App may crash. Clear cache or restart.")
    elif memory_info['percent'] > 75:
        st.sidebar.warning("⚠️ High memory usage! Consider clearing cache or processing fewer files.")
    elif memory_info['percent'] > 60:
        st.sidebar.info("💾 Memory usage is elevated but safe.")

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes session state variables if they don't exist with enhanced error handling."""
    try:
        defaults = {
            "agent_executor": None,
            "messages": [],
            "memory": ConversationBufferWindowMemory(
                k=5,  # Keep small for memory efficiency
                memory_key="chat_history",
                return_messages=True
            ),
            "pending_approval": None,
            "processing_query": False,
            "processing_details": {},
            "processed_files": {},  # Track processed files
            "vector_store": None  # Store vector store reference
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        logger.info("Session state initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        st.error(f"Failed to initialize application state: {str(e)}")

def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """Validates user query for safety and format."""
    if not query or not query.strip():
        return False, "Please enter a question."
    if len(query.strip()) < 3:
        return False, "Please enter a more detailed question."
    if len(query) > 2000:
        return False, "Question is too long. Please keep it under 2000 characters."
    return True, None

# --- UI Rendering ---
st.title("⚛️ Nuclear Industry Agentic RAG Chatbot")
st.markdown("An AI assistant with specialized knowledge of nuclear regulations, safety standards, and data analysis.")

# ADDED: Performance status display
memory_info = get_memory_info()
if memory_info['percent'] > 60:
    st.info(f"💾 Current RAM usage: {memory_info['percent']:.1f}% - Memory-optimized document processing")

with st.expander("⚠️ Important Safety Notice", expanded=False):
    st.warning("""
    **IMPORTANT**: This AI assistant is designed to help with nuclear industry information and analysis, but:
    
    - **Always verify critical information** with official sources and qualified professionals
    - **Do not rely on this tool alone** for safety-critical decisions
    - **Consult nuclear safety experts** for operational guidance
    - **Follow all applicable regulations** and safety protocols
    """)

initialize_session_state()

# --- Callback Functions ---
def handle_approve():
    if st.session_state.pending_approval:
        ai_message = AIMessage(
            content=st.session_state.pending_approval["answer"],
            additional_kwargs={"sources": st.session_state.pending_approval["sources"]}
        )
        st.session_state.messages.append(ai_message)
        st.session_state.memory.chat_memory.add_message(ai_message)
        st.session_state.pending_approval = None

def handle_reject():
    st.session_state.pending_approval = None
    rejection_message = AIMessage(content="*The previous response was rejected by the user.*")
    st.session_state.messages.append(rejection_message)
    st.session_state.memory.chat_memory.add_message(rejection_message)

def clear_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.pending_approval = None
    st.session_state.processing_query = False
    # ADDED: Force garbage collection when clearing
    gc.collect()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔧 Configuration")
    
    provider = st.selectbox(
        "Choose Model Provider",
        ["Ollama (Local)", "OpenAI", "Google", "Custom (OpenAI-Compatible)"]
    )
    
    model_config = {"provider": provider}
    
    if provider == "Ollama (Local)":
        st.info("💡 Make sure Ollama is running locally before proceeding.")
        model_config["llm_model"] = st.text_input("Ollama Chat Model", value="gemma:2b")
        model_config["embedding_model"] = st.text_input("Ollama Embedding Model", value="nomic-embed-text")
        model_config["base_url"] = st.text_input("Ollama Base URL", value="http://localhost:11434")
    elif provider == "Custom (OpenAI-Compatible)":
        model_config["api_key"] = st.text_input("API Key", type="password")
        model_config["base_url"] = st.text_input("Base URL", placeholder="https://api.example.com/v1")
        model_config["llm_model"] = st.text_input("Chat Model Name")
        model_config["embedding_model"] = st.text_input("Embedding Model Name", value="")
    else:  # OpenAI and Google
        model_config["api_key"] = st.text_input(f"{provider} API Key", type="password")
        if provider == "OpenAI":
            model_config["llm_model"] = st.text_input("Chat Model", value="gpt-4o")
            model_config["embedding_model"] = st.text_input("Embedding Model", value="text-embedding-3-small")
        elif provider == "Google":
            model_config["llm_model"] = st.text_input("Chat Model", value="gemini-1.5-flash-latest")
            model_config["embedding_model"] = st.text_input("Embedding Model", value="models/text-embedding-004")
    
    st.markdown("---")
    st.subheader("🔧 Supporting Services")
    model_config["cohere_api_key"] = st.text_input("Cohere API Key", type="password")
    model_config["cohere_rerank_model"] = st.text_input("Cohere Rerank Model", value="rerank-english-v3.0")
    model_config["tavily_api_key"] = st.text_input("Tavily API Key", type="password")
    
    st.markdown("---")
    st.header("📄 Memory-Efficient Document Processing")
    
    # UPDATED: Processing optimization info
    st.info("🚀 **Memory-Optimized Processing**\n- Lightweight document loaders\n- Batch processing for stability\n- Support for PDF, DOCX, Excel, CSV, TXT, MD\n- Automatic text chunking\n- Efficient memory management")
    
    uploaded_files = st.file_uploader(
        "Upload documents for analysis",
        type=["pdf", "txt", "csv", "xlsx", "xls", "docx", "doc", "md"],
        accept_multiple_files=True
    )
    
    # Handle file removal and cleanup
    current_file_hashes = {get_file_hash(f): f.name for f in uploaded_files} if uploaded_files else {}
    processed_files_info = st.session_state.get("processed_files", {})
    
    # Detect removed files
    removed_files = []
    for file_hash, file_info in processed_files_info.items():
        if file_hash not in current_file_hashes:
            removed_files.append((file_hash, file_info["name"]))
    
    # Clean up removed files with selective vector store update
    if removed_files:
        removed_file_names = []
        for file_hash, file_name in removed_files:
            if file_hash in st.session_state.processed_files:
                del st.session_state.processed_files[file_hash]
            if file_name in st.session_state.get("processing_details", {}):
                del st.session_state.processing_details[file_name]
            removed_file_names.append(file_name)
        
        # FIXED: Selective removal from vector store instead of clearing everything
        if st.session_state.get("vector_store") and removed_file_names:
            # Use the utility function for clean document removal
            updated_vector_store = remove_documents_from_vector_store(
                st.session_state.vector_store,
                removed_file_names,
                embeddings if 'embeddings' in locals() else None
            )
            
            # OPTIMIZED: Only re-initialize agent if vector store actually changed
            if updated_vector_store is not None:
                st.session_state.vector_store = updated_vector_store
                
                # Only update agent if we have necessary components
                if st.session_state.get("agent_executor") and 'llm' in locals() and llm:
                    st.session_state.agent_executor = setup_multi_agent_system(
                        llm, model_config, updated_vector_store
                    )
                    logger.info("Agent executor updated after selective file removal")
            else:
                # No documents remain after removal
                st.session_state.vector_store = None
                st.session_state.agent_executor = None
                logger.info("No documents remain, cleared vector store and agent")
        else:
            # If no vector store exists, just clear everything
            st.session_state.vector_store = None
            st.session_state.agent_executor = None
        
        # Show removal message
        if len(removed_files) == 1:
            st.info(f"🗑️ Removed: {removed_files[0][1]}")
        else:
            st.info(f"🗑️ Removed {len(removed_files)} documents")
        
        # ADDED: Force garbage collection after file removal
        gc.collect()
    
    # FIXED: Display uploaded documents as dropdown
    if uploaded_files:
        with st.expander("📄 Uploaded Documents", expanded=True):
            for uploaded_file in uploaded_files:
                file_hash = get_file_hash(uploaded_file)
                file_info = processed_files_info.get(file_hash, {})
                
                # FIXED: Check individual file processing status
                if file_hash in processed_files_info:
                    status_icon = "✅"
                    status_text = "Processed"
                    chunk_count = file_info.get("doc_count", 0)
                    chunk_info = f" ({chunk_count} chunks)"
                else:
                    status_icon = "⏳"
                    status_text = "Pending"
                    chunk_info = ""
                
                # FIXED: Removed file type badges and simplified display
                st.write(f"{status_icon} **{uploaded_file.name}**{chunk_info}")
                st.caption(f"{uploaded_file.size / 1024:.1f} KB • {status_text}")
        
        # FIXED: Show processing progress if files are pending
        pending_count = len([f for f in uploaded_files if get_file_hash(f) not in processed_files_info])
        if pending_count > 0:
            st.info(f"⏳ {pending_count} file(s) pending processing...")
    else:
        st.info("📄 No documents uploaded yet")
        if st.session_state.get("processed_files"):
            st.session_state.processed_files = {}
            st.session_state.processing_details = {}
            st.session_state.vector_store = None
            st.session_state.agent_executor = None
            # ADDED: Garbage collection when clearing all
            gc.collect()
    
    # ADDED: Display memory status
    display_memory_status()
    
    st.markdown("---")
    st.subheader("🔧 Actions")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_conversation()
        st.rerun()
    
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        st.session_state.processed_files = {}
        st.session_state.processing_details = {}
        st.session_state.vector_store = None
        st.session_state.agent_executor = None
        clear_conversation()
        # ADDED: Force memory cleanup
        clear_uploaded_files_memory()
        gc.collect()
        st.rerun()
    
    if st.button("🔄 Reset All", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # ADDED: Aggressive memory cleanup
        clear_uploaded_files_memory()
        gc.collect()
        st.rerun()

# --- Main Application Logic ---
keys_ready = (
    (provider == "Ollama (Local)" and model_config.get("cohere_api_key")) or
    (model_config.get("api_key") and model_config.get("cohere_api_key"))
)

if not keys_ready:
    st.info("🔧 Please configure API keys to begin.", icon="⚠️")
else:
    llm, embeddings = get_models(model_config)
    
    if llm and embeddings and uploaded_files:
        # MEMORY CHECK: Monitor memory before processing
        memory_info = get_memory_info()
        if memory_info['percent'] > 85:
            st.warning("⚠️ High memory usage detected! Consider clearing cache or processing fewer files at once.")
        
        # FIXED: Process files individually and update status immediately
        files_to_process = []
        current_file_hashes = {get_file_hash(f): f for f in uploaded_files}
        processed_files_info = st.session_state.get("processed_files", {})
        
        # Check which files need processing
        for uploaded_file in uploaded_files:
            file_hash = get_file_hash(uploaded_file)
            if file_hash not in processed_files_info:
                files_to_process.append(uploaded_file)
        
        # Process new files
        if files_to_process:
            # CRITICAL: Add safety check before processing
            can_process, error_msg = check_memory_safety(files_to_process, max_total_mb=150)
            if not can_process:
                st.error(error_msg)
                st.info("💡 **Tip**: Upload files one at a time or use smaller files to avoid crashes.")
            else:
                with st.spinner(f"Processing {len(files_to_process)} document(s) with memory-optimized loaders..."):
                    try:
                        vector_store, _, processing_details = process_uploaded_files_incremental(
                            files_to_process, embeddings, model_config
                        )
                        
                        # FIXED: Update processing details and status immediately for each processed file
                        if processing_details:
                            st.session_state.processing_details.update(processing_details)
                            
                            # Update processed files info for each file individually
                            for file_name, details in processing_details.items():
                                if isinstance(details, dict) and details.get('success'):
                                    file_hash = None
                                    # Find the corresponding file hash
                                    for uploaded_file in files_to_process:
                                        if uploaded_file.name == file_name:
                                            file_hash = get_file_hash(uploaded_file)
                                            break
                                    
                                    if file_hash:
                                        chunk_count = len(details.get("chunks", []))
                                        processing_time = details.get("processing_time", 0)
                                        st.session_state.processed_files[file_hash] = {
                                            "name": file_name,
                                            "doc_count": chunk_count,
                                            "processing_time": processing_time,
                                            "processed_time": time.time()
                                        }
                        
                        # FIXED: Always update vector store and recreate agent when new files are processed
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            
                            # OPTIMIZED: Only recreate agent if we successfully processed files
                            st.session_state.agent_executor = setup_multi_agent_system(
                                llm, model_config, vector_store
                            )
                            logger.info("Agent executor updated with new vector store")
                        
                        # CRITICAL: Clear uploaded file memory after successful processing
                        clear_uploaded_files_memory()
                        
                        # ADDED: Cleanup after processing
                        gc.collect()
                        # Force rerun to update sidebar status
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
                        logger.error(f"File processing error: {str(e)}")
        
        # Ensure agent executor exists with current vector store if we have processed files
        elif st.session_state.get("processed_files") and not st.session_state.get("agent_executor"):
            # This handles the case where files were processed but agent wasn't created
            if st.session_state.get("vector_store"):
                st.session_state.agent_executor = setup_multi_agent_system(
                    llm, model_config, st.session_state.vector_store
                )
                logger.info("Agent executor created with existing vector store")
        
        # Show success message when all files are processed
        if st.session_state.get("processed_files"):
            processed_count = len(st.session_state.processed_files)
            total_chunks = sum(info.get("doc_count", 0) for info in st.session_state.processed_files.values())
            avg_time = sum(info.get("processing_time", 0) for info in st.session_state.processed_files.values()) / processed_count if processed_count > 0 else 0
            st.success(f"✅ Knowledge base ready with {processed_count} files ({total_chunks} chunks) - Avg processing: {avg_time:.1f}s/file", icon="🚀")
        
        # Display processing details with memory-efficient loaders info
        if st.session_state.get("processing_details"):
            st.markdown("---")
            with st.expander("📄 View Document Processing Details"):
                details = st.session_state.processing_details
                if not details:
                    st.info("No documents have been processed yet.")
                
                for filename, file_details in details.items():
                    if isinstance(file_details, dict) and file_details.get('success'):
                        chunks = file_details.get("chunks", [])
                        processing_time = file_details.get("processing_time", 0)
                        loader_type = file_details.get("loader_type", "Unknown")
                        original_docs = file_details.get("original_docs", 0)
                        final_chunks = file_details.get("final_chunks", len(chunks))
                        
                        with st.expander(f"📄 **{filename}** ({final_chunks} chunks, {processing_time:.1f}s)"):
                            st.info(f"✨ **Processing Details:**")
                            st.write(f"• **Loader Used:** {loader_type}")
                            st.write(f"• **Original Documents:** {original_docs}")
                            st.write(f"• **Final Chunks:** {final_chunks}")
                            st.write(f"• **Processing Time:** {processing_time:.2f}s")
                            st.write(f"• **Memory Optimizations:** Batch processing, efficient chunking, automatic cleanup")
                            
                            st.markdown("---")
                            st.subheader("Document Chunks")
                            
                            # MEMORY OPTIMIZATION: Limit displayed chunks to first 10 for large documents
                            max_display_chunks = min(10, len(chunks))
                            if len(chunks) > max_display_chunks:
                                st.info(f"Showing first {max_display_chunks} of {len(chunks)} chunks (memory optimization)")
                            
                            for i, doc in enumerate(chunks[:max_display_chunks]):
                                key_suffix = f"chunk_{hash(filename)}_{i}_{id(doc)}"
                                
                                # Build comprehensive chunk header
                                chunk_header = f"**Chunk {i+1}**"
                                
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    metadata_parts = []
                                    
                                    # Page information (most important)
                                    if doc.metadata.get("page") is not None:
                                        page_info = doc.metadata.get("page")
                                        if isinstance(page_info, (int, str)) and str(page_info).isdigit():
                                            metadata_parts.append(f"Page: {page_info}")
                                        else:
                                            metadata_parts.append(f"Page: {page_info}")
                                    
                                    # Document and chunk tracking
                                    if doc.metadata.get("document_index") is not None:
                                        doc_idx = doc.metadata.get("document_index")
                                        metadata_parts.append(f"Doc: {doc_idx + 1}")
                                    
                                    if doc.metadata.get("chunk_index") is not None:
                                        chunk_idx = doc.metadata.get("chunk_index")
                                        total_chunks = doc.metadata.get("total_chunks_in_doc", "?")
                                        metadata_parts.append(f"Chunk: {chunk_idx + 1}/{total_chunks}")
                                    
                                    # File type
                                    if doc.metadata.get("file_type"):
                                        file_type = doc.metadata.get("file_type").upper().replace(".", "")
                                        metadata_parts.append(f"[{file_type}]")
                                    
                                    if metadata_parts:
                                        chunk_header += f" ({' • '.join(metadata_parts)})"
                                
                                st.markdown(chunk_header)
                                
                                # Show detailed metadata in an expander for debugging
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    with st.expander(f"🔍 Metadata Details", expanded=False):
                                        st.json(doc.metadata)
                                
                                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                st.text_area(f"Chunk {i+1} Content", content, height=150, key=key_suffix, label_visibility="collapsed")
                                st.markdown("---")
                    else:
                        error_msg = file_details if isinstance(file_details, str) else file_details.get('error', 'Processing failed')
                        st.warning(f"📄 **{filename}**: {error_msg}")
    
    elif uploaded_files and (not llm or not embeddings):
        st.warning("⚠️ Please check your model configuration.")
    elif not uploaded_files:
        st.info("📄 Please upload documents to begin analysis.")

# --- Chat Interface ---
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message.type):
        st.markdown(message.content)
        
        if message.type == "ai" and message.additional_kwargs.get("sources"):
            with st.expander(f"📚 View Sources ({len(message.additional_kwargs['sources'])} found)"):
                for src_idx, source in enumerate(message.additional_kwargs["sources"]):
                    source_name = source.get("source", "N/A")
                    page_info = source.get("page", "N/A")
                    
                    # Enhanced source display with better formatting
                    if str(page_info).isdigit():
                        source_header = f"📄 **{source_name}** (Page {page_info})"
                    elif page_info != "N/A":
                        source_header = f"📄 **{source_name}** ({page_info})"
                    else:
                        source_header = f"📄 **{source_name}**"
                    
                    st.markdown(source_header)
                    
                    key_suffix = f"msg_{msg_idx}_src_{src_idx}_{hash(str(source))}"
                    content = source.get("content", "")
                    
                    # Truncate very long content for better UI
                    if len(content) > 500:
                        content = content[:500] + "...\n\n[Content truncated - see full document for complete text]"
                    
                    st.text_area("Content Snippet", content, height=100, key=key_suffix, label_visibility="collapsed")
                    st.markdown("---")

if st.session_state.pending_approval:
    with st.container(border=True):
        st.markdown("### 🤖 AI Response (Awaiting Your Review)")
        st.markdown(st.session_state.pending_approval["answer"])
        
        with st.expander(f"📚 Sources ({len(st.session_state.pending_approval['sources'])} found)"):
            for src_idx, source in enumerate(st.session_state.pending_approval["sources"]):
                source_name = source.get("source", "N/A")
                page_info = source.get("page", "N/A")
                
                # Enhanced source display with better formatting
                if str(page_info).isdigit():
                    source_header = f"📄 **{source_name}** (Page {page_info})"
                elif page_info != "N/A":
                    source_header = f"📄 **{source_name}** ({page_info})"
                else:
                    source_header = f"📄 **{source_name}**"
                
                st.markdown(source_header)
                
                key_suffix = f"pending_src_{src_idx}_{hash(str(source))}"
                content = source.get("content", "")
                
                # Truncate very long content for better UI
                if len(content) > 500:
                    content = content[:500] + "...\n\n[Content truncated - see full document for complete text]"
                
                st.text_area("Content Snippet", content, height=100, key=key_suffix, label_visibility="collapsed")
                st.markdown("---")
        
        col1, col2, _ = st.columns([1, 1, 4])
        if col1.button("✅ Approve", on_click=handle_approve, use_container_width=True):
            st.rerun()
        if col2.button("❌ Reject", on_click=handle_reject, use_container_width=True):
            st.rerun()

if not st.session_state.pending_approval:
    if query := st.chat_input("Ask a question about your documents..."):
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            st.error(error_msg)
        else:
            user_message = HumanMessage(content=query)
            st.session_state.messages.append(user_message)
            st.session_state.memory.chat_memory.add_message(user_message)
            
            with st.chat_message("human"):
                st.markdown(query)
            
            if st.session_state.agent_executor:
                with st.chat_message("ai"):
                    with st.spinner("🤖 AI agent is analyzing documents with memory-efficient processing..."):
                        try:
                            # MEMORY CHECK: Monitor memory during query processing
                            memory_before = get_memory_info()['percent']
                            
                            response = st.session_state.agent_executor.invoke({"input": query})
                            output = response.get("output")
                            
                            # ADDED: Memory monitoring after processing
                            memory_after = get_memory_info()['percent']
                            if memory_after > memory_before + 10:
                                logger.warning(f"Memory usage increased by {memory_after - memory_before:.1f}% during query processing")
                            
                            logger.info(f"Agent output: {output}")
                            
                            if output and output.strip() != "":
                                # Create OutputFixingParser (this should match what's in agents.py)
                                base_parser = PydanticOutputParser(pydantic_object=FinalAnswer)
                                parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
                                
                                # Simple parsing - OutputFixingParser handles all the complexity
                                try:
                                    parsed_output = parser.parse(output)
                                    st.session_state.pending_approval = {
                                        "answer": parsed_output.answer,
                                        "sources": [s.model_dump() for s in parsed_output.sources]
                                    }
                                    st.rerun()
                                
                                except Exception as e:
                                    logger.error(f"OutputFixingParser failed: {str(e)}")
                                    st.error("Failed to parse the AI response even with error correction. Please try rephrasing your question.")
                            else:
                                st.error("The AI agent returned an empty response. Please try again.")
                        
                        except Exception as e:
                            logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
                            st.error("An error occurred while processing your question. Please try rephrasing or check the logs.")
                        
                        finally:
                            # ADDED: Cleanup after each query
                            gc.collect()
            else:
                st.warning("⚠️ Please upload documents to activate the AI agent.")