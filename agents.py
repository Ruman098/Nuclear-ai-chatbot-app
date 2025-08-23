import os
import streamlit as st
import logging
from langchain.chains import RetrievalQA
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser  # ADD THIS IMPORT
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Optional, Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Pydantic models for structured output
class Source(BaseModel):
    source: Optional[str] = Field(description="The source document, file name, or URL of the information.")
    page: Optional[str] = Field(description="The page number or section of the source.")
    content: Optional[str] = Field(description="The specific text snippet extracted from the source.")

class FinalAnswer(BaseModel):
    answer: str = Field(description="The final, synthesized answer to the user's query.")
    sources: List[Source] = Field(description="A list of all sources used to construct the answer.")

    @field_validator('sources')
    def require_sources_for_answer(cls, v, info):
        """Validate sources based on answer type and content."""
        answer = info.data.get('answer', '')
        
        # Extended conversational patterns
        conversational_patterns = [
            "hello", "hi", "how are you", "thank you", "thanks", "good morning",
            "good afternoon", "good evening", "bye", "goodbye", "nice to meet",
            "i am", "i'm a", "my name is", "assistance", "help you", "ready to help"
        ]
        
        answer_lower = answer.lower()
        is_conversational = any(pattern in answer_lower for pattern in conversational_patterns)
        
        # Check for uncertainty expressions - expanded list
        uncertainty_phrases = [
            "don't know", "could not find", "cannot find", "not sure", "unclear",
            "don't have", "have no access", "no access", "not provided", "limited to",
            "no information", "no specific knowledge", "not available", "i don't have",
            "i have no", "i cannot", "unable to", "no data", "not found"
        ]
        has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)
        
        # Only require sources for substantive, factual answers
        if not is_conversational and not has_uncertainty and len(answer.strip()) > 20 and not v:
            raise ValueError("A non-conversational answer was generated, but no sources were provided.")
        
        return v

def load_prompt_template(file_path: str = "prompt_template.txt") -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        st.error(f"Critical error: Could not load prompt template file at {file_path}")
        return None


def create_rag_output(result: Dict[str, Any]) -> Dict[str, Any]:
    answer = result.get("result", "Could not find an answer.")
    source_documents = result.get("source_documents", [])
    
    source_list = []
    for doc in source_documents:
        source_list.append({
            "source": doc.metadata.get('source', 'N/A'),
            "page": str(doc.metadata.get('page', 'N/A')),
            "content": doc.page_content
        })
            
    return {"answer": answer, "sources": source_list}


def setup_multi_agent_system(
    llm, 
    model_config: Dict[str, Any],
    vector_store=None
) -> Optional[AgentExecutor]:
    """Setup multi-agent system with robust OutputFixingParser."""
    tavily_api_key = model_config.get("tavily_api_key")
    cohere_api_key = model_config.get("cohere_api_key")
    cohere_rerank_model = model_config.get("cohere_rerank_model", "rerank-english-v3.0")

    tools = []

    if vector_store:
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 25, "fetch_k": 50})
        reranker = CohereRerank(cohere_api_key=cohere_api_key, model=cohere_rerank_model, top_n=10, user_agent="langchain")
        compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
        
        QA_PROMPT = PromptTemplate.from_template(
            "Use the following pieces of context to answer the question. You MUST use only the information from the context provided. If you don't know the answer, just say that you don't know.\n"
            "CONTEXT: {context}\nQUESTION: {question}\nANSWER:"
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm, 
            retriever=compression_retriever, 
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        @tool
        def document_qa_tool(query: str) -> Dict[str, Any]:
            """Use this tool FIRST to answer questions using the provided documents. This is your primary source of information."""
            result = qa_chain.invoke({"query": query.strip()})
            return create_rag_output(result)
        tools.append(document_qa_tool)

    if tavily_api_key:
        tavily_search = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)
        @tool
        def web_search_tool(query: str) -> str:
            """Use as a fallback for general knowledge or if documents don't answer the question."""
            return tavily_search.invoke(query.strip())
        tools.append(web_search_tool)

    prompt_template_string = load_prompt_template()
    if not prompt_template_string: return None
    
    try:
        # MODIFIED: Create base parser and wrap with OutputFixingParser
        base_parser = PydanticOutputParser(pydantic_object=FinalAnswer)
        parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
        
        # Get format instructions from the base parser
        format_instructions = base_parser.get_format_instructions()
        
        # Clean the format instructions to prevent template parsing issues
        # Replace newlines and normalize whitespace to prevent them being interpreted as template variables
        clean_format_instructions = format_instructions.replace('\n', ' ').replace('\r', ' ')
        # Remove extra spaces
        clean_format_instructions = ' '.join(clean_format_instructions.split())
        
        logger.info(f"Using OutputFixingParser with base Pydantic parser")
        logger.info(f"Cleaned format instructions: {clean_format_instructions}")
        
        # Create the prompt with proper variable mapping
        prompt = PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            template=prompt_template_string,
            partial_variables={
                "format_instructions": clean_format_instructions,
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
            }
        )
        
        agent = create_react_agent(llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            memory=st.session_state.get('memory'),
            handle_parsing_errors=True,
            max_iterations=20,
            max_execution_time=300,
            #early_stopping_method="generate"
        )
        return agent_executor
        
    except Exception as e:
        st.error(f"Failed to initialize AI agent: {str(e)}")
        logger.error(f"Agent initialization error: {str(e)}\n{traceback.format_exc()}")
        return None