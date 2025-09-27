# AskYourDoc

**Upload PDFs or DOCs and interact with them: get summaries and ask questions using AI with LangChain.**

---

## Description
**AskYourDoc** is a Python project that allows you to:
- **Summarize documents:** Generate concise summaries of PDFs and Word files.
- **Ask questions:** Get accurate answers about the content of your documents.

Powered by **LangChain** and **FAISS**, it provides intelligent document retrieval and natural language understanding, making it easy to extract insights from any document.

---

## Features
- Upload PDF or DOC/DOCX files.
- Automatic document splitting into manageable chunks.
- AI-powered summarization.
- Interactive question-answering based on document content.
- Uses **HuggingFace embeddings** and **Flan-T5 model** for natural language processing.

---

## How It Works

1. **Upload Document:** Upload a PDF, DOC/DOCX, or TXT file to the system.  
2. **Load Document:** The file is read and converted into text.  
3. **Split Document:** Text is split into smaller chunks for better AI processing.  
4. **Generate Embeddings:** Each chunk is converted into vector embeddings using HuggingFace.  
5. **Create Vector Store:** Chunks are stored in FAISS for fast retrieval.  
6. **Summarize Document:** AI generates a concise summary of the entire document.  
7. **Ask Questions:** Users can ask specific questions and AI provides accurate answers from the document content.  

---

## Installation

```bash
pip install langchain langchain-community faiss-cpu pypdf python-docx sentence-transformers transformers torch
```
