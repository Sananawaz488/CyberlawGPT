import os
import re
import hashlib
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# Cyber Law GPT
# RAG application for Pakistani cyber-law documents
# Stack: Streamlit + FAISS + Sentence Transformers + Groq
# ============================================================

APP_NAME = "Cyber Law GPT"
DEFAULT_PDF = "cyberlaw.pdf"

DEFAULT_MODEL = "openai/gpt-oss-120b"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }

    .answer-box {
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
    }

    .source-box {
        padding: .8rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,.20);
        margin-bottom: .5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Utility functions
# ============================================================

def get_secret_or_env(name: str) -> str:
    """Read a value from Streamlit secrets first, then environment."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, "")


def clean_text(text: str) -> str:
    """Normalize extracted PDF text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_pages(pdf_path: str):
    """Extract PDF text page by page."""
    document = fitz.open(pdf_path)
    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    finally:
        document.close()

    return pages


def split_text(
    text: str,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Split text into manageable chunks while trying
    to preserve sentence/paragraph boundaries.
    """

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        boundary_candidates = [
            text.rfind("\n\n", start, end),
            text.rfind(". ", start, end),
            text.rfind("۔", start, end),
            text.rfind("?", start, end),
        ]

        boundary = max(boundary_candidates)

        if boundary > start + int(chunk_size * 0.55):
            end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def build_chunks(pdf_path: str):
    """Create chunks while preserving page information."""

    pages = extract_pdf_pages(pdf_path)

    all_chunks = []

    for page in pages:

        page_chunks = split_text(page["text"])

        for chunk_index, chunk in enumerate(
            page_chunks,
            start=1,
        ):

            all_chunks.append(
                {
                    "text": chunk,
                    "page": page["page"],
                    "chunk": chunk_index,
                }
            )

    return all_chunks


# ============================================================
# Embedding model
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# FAISS
# ============================================================

def create_faiss_index(chunks):
    """Create FAISS index using cosine similarity."""

    model = load_embedding_model()

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    # Normalized vectors + inner product = cosine similarity
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


@st.cache_resource(show_spinner=False)
def build_knowledge_base(
    pdf_path: str,
    pdf_signature: str,
):
    """
    Build and cache the knowledge base.

    When the PDF changes, the signature changes and
    Streamlit creates a fresh knowledge base.
    """

    del pdf_signature

    chunks = build_chunks(pdf_path)

    if not chunks:
        raise ValueError(
            "No readable text was found in the PDF. "
            "The document may be image-only or damaged."
        )

    index = create_faiss_index(chunks)

    return index, chunks


def file_signature(path: str) -> str:
    """Create SHA256 signature of the PDF."""

    hasher = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    query,
    index,
    chunks,
    top_k=5,
):
    """Retrieve relevant PDF passages."""

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    # IMPORTANT:
    # Search must be on a separate line.
    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        item = dict(
            chunks[idx]
        )

        item["score"] = float(score)

        results.append(item)

    return results


# ============================================================
# Context
# ============================================================

def make_context(results):

    blocks = []

    for number, item in enumerate(
        results,
        start=1,
    ):

        blocks.append(
            f"[SOURCE {number} | PDF page {item['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(blocks)


# ============================================================
# Response settings
# ============================================================

def response_instruction(
    level: str,
    size: str,
    answer_style: str,
):

    level_instructions = {

        "Beginner": (
            "Explain legal concepts in simple language. "
            "Define difficult legal terms briefly."
        ),

        "Intermediate": (
            "Use moderately technical legal language and explain "
            "important legal concepts without excessive simplification."
        ),

        "Expert": (
            "Use precise legal terminology and provide section-focused "
            "analysis, distinctions, exceptions, and procedural context "
            "when the retrieved material supports them."
        ),
    }

    size_instructions = {

        "Short": (
            "Keep the answer concise, usually 1–3 short paragraphs."
        ),

        "Medium": (
            "Give a balanced answer with useful detail and bullets "
            "where appropriate."
        ),

        "Detailed": (
            "Give a thorough answer with structured headings, "
            "relevant sections, implications, and limitations."
        ),
    }

    style_instructions = {

        "Simple explanation": (
            "Prioritize clarity and practical understanding."
        ),

        "Section-focused": (
            "Prioritize the relevant Act section(s), wording, "
            "offence, punishment, and conditions."
        ),

        "Case-style analysis": (
            "Analyze the user's scenario carefully: identify the "
            "potentially relevant provision, explain why it may apply, "
            "and state what facts could change the analysis."
        ),
    }

    return (
        level_instructions[level]
        + " "
        + size_instructions[size]
        + " "
        + style_instructions[answer_style]
    )


# ============================================================
# Groq
# ============================================================

def answer_with_groq(
    question,
    context,
    level,
    size,
    answer_style,
    model_name,
):

    api_key = get_secret_or_env(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Please configure it in Streamlit Secrets."
        )

    client = Groq(
        api_key=api_key
    )

    instructions = response_instruction(
        level,
        size,
        answer_style,
    )

    system_prompt = f"""
You are Cyber Law GPT, a document-grounded educational assistant
for Pakistani cyber law.

Your primary legal source is the supplied PDF context.

IMPORTANT GROUNDING RULES:

1. Answer from the retrieved PDF context whenever possible.

2. Do not invent sections, penalties, definitions, procedures,
authorities, dates, or legal rights.

3. If the PDF does not contain enough information to answer a question,
clearly say that the supplied document does not provide enough information.

4. Do not pretend that a general legal assumption is stated in the Act.

5. Distinguish between what the document explicitly says and your explanation.

6. When relevant, cite the source using the PDF page number shown in
the context, for example: "PDF page 12".

7. If the question describes a hypothetical situation, explain the
potentially relevant provision but do not claim that a court would
definitely reach a particular outcome.

8. This is an educational information tool, not legal representation
or professional legal advice.

9. Do not help a user commit, conceal, evade, or facilitate cybercrime.

10. If a request asks how to perform wrongdoing, do not provide
operational instructions. You may instead explain the relevant legal
prohibition, risk, or lawful defensive concept.

USER PREFERENCE:

Technical level: {level}
Response size: {size}
Answer style: {answer_style}

{instructions}
""".strip()

    user_prompt = f"""
RETRIEVED LEGAL CONTEXT
=======================

{context}


USER QUESTION
=============

{question}


TASK
====

Answer the user's question using the retrieved context.

Include relevant PDF page references when available.

If the retrieved context is insufficient,
say so explicitly.
""".strip()

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.15,
        max_completion_tokens={
            "Short": 500,
            "Medium": 900,
            "Detailed": 1600,
        }[size],
    )

    return completion.choices[0].message.content


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    technical_level = st.selectbox(
        "Technical level",
        [
            "Beginner",
            "Intermediate",
            "Expert",
        ],
        index=0,
        help=(
            "Controls how technical the legal explanation should be."
        ),
    )

    response_size = st.selectbox(
        "Response size",
        [
            "Short",
            "Medium",
            "Detailed",
        ],
        index=1,
    )

    answer_style = st.selectbox(
        "Answer style",
        [
            "Simple explanation",
            "Section-focused",
            "Case-style analysis",
        ],
        index=1,
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=2,
        max_value=8,
        value=5,
        help=(
            "Number of relevant PDF passages sent to the language model."
        ),
    )

    model_name = st.selectbox(
        "Groq model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
        ],
        index=0,
    )

    st.divider()

    st.subheader("📄 Knowledge source")

    uploaded_pdf = st.file_uploader(
        "Upload Pakistani cyber-law PDF",
        type=["pdf"],
        help=(
            "Optional when cyberlaw.pdf already exists "
            "in the project folder."
        ),
    )

    if uploaded_pdf is not None:

        uploaded_path = Path(
            "uploaded_cyberlaw.pdf"
        )

        if (
            not uploaded_path.exists()
            or st.session_state.get(
                "uploaded_pdf_name"
            ) != uploaded_pdf.name
        ):

            uploaded_path.write_bytes(
                uploaded_pdf.getbuffer()
            )

            st.session_state[
                "uploaded_pdf_name"
            ] = uploaded_pdf.name

            st.cache_resource.clear()

            st.rerun()

        pdf_path = str(
            uploaded_path
        )

    elif Path(DEFAULT_PDF).exists():

        pdf_path = DEFAULT_PDF

    else:

        pdf_path = None

    st.divider()

    st.caption(
        "Cyber Law GPT answers from the supplied legal document. "
        "Always verify important legal matters with a qualified "
        "legal professional."
    )


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">⚖️ Cyber Law GPT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered RAG assistant for the supplied Pakistani "
    "cyber-law document"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Knowledge base
# ============================================================

if not pdf_path:

    st.warning(
        "No cyber-law PDF was found. Upload the PDF from the "
        "sidebar, or place it in the project folder as "
        "`cyberlaw.pdf`."
    )

    st.stop()


try:

    signature = file_signature(
        pdf_path
    )

    with st.spinner(
        "Building the legal knowledge base and generating embeddings..."
    ):

        index, chunks = build_knowledge_base(
            pdf_path,
            signature,
        )

    st.success(
        f"Knowledge base ready — "
        f"{len(chunks)} searchable passages indexed."
    )

except Exception as error:

    st.error(
        f"Knowledge-base setup failed: {error}"
    )

    st.stop()


# ============================================================
# Example questions
# ============================================================

with st.expander(
    "💡 Try example questions"
):

    examples = [

        "What is the purpose and scope of the Prevention of Electronic Crimes Act?",

        "What does the Act say about unauthorized access to an information system?",

        "What is spoofing and what punishment is provided for it?",

        "What does the Act say about investigation agencies and digital forensics?",

        "What is the legal treatment of cyber offences under the Act?",

        "Explain the relevant law for a hypothetical online cybercrime situation.",
    ]

    for example in examples:

        if st.button(
            example,
            key=f"example_{hash(example)}",
        ):

            st.session_state[
                "question"
            ] = example


# ============================================================
# Chat input
# ============================================================

question = st.text_area(
    "Ask a question about Pakistani cyber law",

    value=st.session_state.get(
        "question",
        "",
    ),

    height=120,

    placeholder=(
        "Example: What is unauthorized access under "
        "the Act, and what punishment is provided?"
    ),
)


ask = st.button(
    "🔎 Analyze with Cyber Law GPT",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Generate answer
# ============================================================

if ask:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

        st.stop()

    # -------------------------
    # Retrieve relevant chunks
    # -------------------------

    with st.spinner(
        "Searching the cyber-law document..."
    ):

        retrieved = retrieve(
            question.strip(),
            index,
            chunks,
            top_k=top_k,
        )

    if not retrieved:

        st.warning(
            "No relevant passage was found."
        )

        st.stop()

    # -------------------------
    # Build context
    # -------------------------

    context = make_context(
        retrieved
    )

    # -------------------------
    # Generate answer
    # -------------------------

    with st.spinner(
        "Generating a grounded legal explanation..."
    ):

        try:

            answer = answer_with_groq(
                question=question.strip(),
                context=context,
                level=technical_level,
                size=response_size,
                answer_style=answer_style,
                model_name=model_name,
            )

        except Exception as error:

            st.error(
                f"Groq request failed: {error}"
            )

            st.stop()

    # -------------------------
    # Answer
    # -------------------------

    st.markdown(
        "### 🧠 Answer"
    )

    st.markdown(
        answer
    )

    # -------------------------
    # Sources
    # -------------------------

    st.markdown(
        "### 📚 Retrieved legal sources"
    )

    for number, item in enumerate(
        retrieved,
        start=1,
    ):

        with st.expander(
            f"Source {number} — "
            f"PDF page {item['page']} — "
            f"similarity {item['score']:.3f}"
        ):

            st.write(
                item["text"]
            )

    st.info(
        "Legal information only: this answer is generated from "
        "the supplied document and should not be treated as "
        "professional legal advice."
    )
