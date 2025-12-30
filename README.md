# ☢️ Nuclear Domain Assistant AI Chatbot

A **domain-restricted AI chatbot** built with **Streamlit, LangChain, and Google Gemini**, designed to answer **nuclear engineering and nuclear science–related questions** using **uploaded documents** and **controlled web search**, while strictly avoiding operational or procedural guidance.

🚀 **Live Demo (Streamlit Cloud):**  
👉 **https://nuclear-ai-chatbot-app.streamlit.app/**

---

## 📌 Overview

**Nuclear Domain Assistant** is a safety-aware AI application that enables users to:

- Upload technical documents (PDFs, reports, spreadsheets, etc.)
- Ask nuclear-domain questions grounded in those documents
- Receive **concise, citation-backed answers**
- Safely fall back to web search when internal knowledge is insufficient

The system enforces **strict domain boundaries**, ensuring compliance with safety and ethical constraints by **refusing operational or step-by-step nuclear procedures**.

---

## ✨ Key Features

- 🔍 **Document-Based Question Answering (RAG)**
- 📚 **Multi-format document support** (PDF, DOCX, XLSX, CSV, TXT, MD)
- 🧠 **Gemini 2.5 Flash LLM integration**
- 🧩 **FAISS vector search with dynamic retrieval**
- 🌐 **Controlled web search fallback (Tavily)**
- 🛡️ **Safety-first nuclear domain guardrails**
- 📎 **Source citation with page-level references**
- ⚡ **Streamlit UI with chat approval workflow**
- ☁️ **Deployed on Streamlit Cloud**

---

## 🏗️ Project Structure
```bash
├── app.py                 # Streamlit UI & app orchestration
├── agents.py              # ReAct agent, tool logic, safety fallbacks
├── utils.py               # Document ingestion, chunking, embeddings
├── prompt_template.txt    # System prompt & safety rules
├── requirements.txt       # Project dependencies
```
---

## 🧠 How It Works

1. **Document Upload**
   - Users upload nuclear-related documents via the sidebar
   - Files are validated, parsed, and chunked

2. **Vector Indexing**
   - Documents are embedded using Gemini embeddings
   - FAISS enables efficient similarity search

3. **Agent Reasoning**
   - A ReAct-based agent determines whether to:
     - Query uploaded documents first
     - Use web search only if necessary

4. **Safety Enforcement**
   - Operational, emergency, or step-by-step requests are rejected
   - Responses remain descriptive, analytical, and non-procedural

5. **Answer Approval**
   - Generated responses require user approval before appearing in chat history

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM:** Google Gemini 2.5 Flash  
- **Framework:** LangChain (ReAct Agent)  
- **Vector Store:** FAISS  
- **Embeddings:** Google Generative AI Embeddings  
- **Document Parsing:** PyMuPDF, Unstructured  
- **Web Search:** Tavily  
- **Deployment:** Streamlit Cloud  

---

## 📦 Installation (Local Setup)

### Clone the Repository
```bash
git clone https://github.com/Ruman098/Nuclear-ai-chatbot-app.git
cd nuclear-ai-chatbot-app
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Application
```bash
streamlit run app.py
```

---

## Future Improvements

- Role-based access control
- Per-document confidence scoring
- PDF highlight-based citations
- Multi-model support
- Usage analytics dashboard





