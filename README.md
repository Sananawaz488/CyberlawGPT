⚖️ Cyber Law GPT

Cyber Law GPT is a Retrieval-Augmented Generation (RAG) application that answers questions from a Pakistani cyber-law PDF using:

Python

Streamlit

FAISS

Sentence Transformers

PyMuPDF

Groq

Features

Pakistani cyber-law document as the knowledge base

Automatic PDF text extraction

Automatic embeddings during setup

FAISS semantic search

Groq-powered grounded answers

Beginner / Intermediate / Expert technical level

Short / Medium / Detailed responses

Simple, Section-focused and Case-style answer modes

Adjustable number of retrieved passages

PDF page references for retrieved evidence

Works in Google Colab for testing

Ready for Streamlit Community Cloud

Project Structure

cyber-law-gpt/
├── app.py
├── requirements.txt
├── README.md
└── cyberlaw.pdf

cyberlaw.pdf should be the Pakistani cyber-law document used as the application's legal knowledge base.

How the RAG Pipeline Works

Pakistani Cyber-Law PDF
        ↓
PyMuPDF text extraction
        ↓
Text chunking + page metadata
        ↓
Sentence Transformer embeddings
        ↓
FAISS vector index
        ↓
User question
        ↓
Question embedding
        ↓
Relevant legal passages
        ↓
Groq LLM
        ↓
Grounded answer + PDF page references

Embeddings are generated automatically when the knowledge base is initialized. The application caches the resulting index for the current PDF, and a changed PDF triggers a new index.

Run in Google Colab

Create a new Colab notebook.

Upload app.py, requirements.txt, and cyberlaw.pdf.

Install dependencies:

!pip install -r requirements.txt

Configure your Groq API key in the Colab environment.

Start Streamlit:

!streamlit run app.py &>/content/streamlit.log &

Expose port 8501 using your preferred Colab-compatible tunnel method.

The same project can then be pushed to GitHub and deployed on Streamlit Community Cloud.

Streamlit Cloud

Upload these project files to a GitHub repository:

app.py
requirements.txt
README.md
cyberlaw.pdf

Select app.py as the Streamlit entry point and configure the required Groq credential through the deployment platform's secret-management system.

Important Legal Scope

Cyber Law GPT is a document-grounded educational assistant. It is designed to explain the contents of the supplied Pakistani cyber-law document.

It should not be treated as:

a lawyer,

a legal representative,

a court,

a source of guaranteed legal outcomes, or

a replacement for professional legal advice.

If the supplied PDF does not contain enough information to answer a question, the application is instructed to say so instead of inventing a provision.

Security & Grounding

The application uses a strict RAG prompt that tells the language model to:

prioritize retrieved legal text,

avoid inventing sections or penalties,

provide PDF page references where available,

distinguish document content from explanation,

avoid presenting hypothetical outcomes as guaranteed legal conclusions.

Note

The application's legal knowledge is limited to the PDF supplied as its knowledge source. For real-world legal matters, users should verify the current applicable law and consult a qualified legal professional.
