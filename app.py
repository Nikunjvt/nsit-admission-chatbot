import streamlit as st
import os
import pickle
import hashlib
from pathlib import Path

import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import groq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSIT Admission Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --primary: #1a3a5c;
    --accent:  #c8922a;
    --light:   #f5f0e8;
    --border:  #d4c5a9;
    --text:    #1e1e1e;
    --muted:   #6b6b6b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--light);
    color: var(--text);
}

/* Header */
.nsit-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2340 100%);
    color: white;
    padding: 1.8rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 4px 20px rgba(26,58,92,0.3);
}
.nsit-header .icon { font-size: 2.8rem; }
.nsit-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    margin: 0;
    letter-spacing: 0.02em;
}
.nsit-header p { margin: 0.2rem 0 0; font-size: 0.85rem; opacity: 0.75; }

/* Chat bubbles */
.msg-user {
    background: var(--primary);
    color: white;
    padding: 0.85rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(26,58,92,0.2);
}
.msg-bot {
    background: white;
    color: var(--text);
    padding: 0.85rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem auto 0.5rem 0;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.6;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.msg-bot .source-tag {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.5rem;
    border-top: 1px solid var(--border);
    padding-top: 0.4rem;
}

/* Contact card */
.contact-card {
    background: #fff8ee;
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d2340 !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stFileUploader label { color: #c8922a !important; font-weight: 500; }

/* Input box */
.stChatInputContainer { border-top: 2px solid var(--border); padding-top: 0.5rem; }

/* Status badge */
.status-ok  { background:#1a5c3a; color:white; padding:0.2rem 0.6rem; border-radius:20px; font-size:0.75rem; }
.status-off { background:#5c1a1a; color:white; padding:0.2rem 0.6rem; border-radius:20px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CONTACT_INFO = """
📞 **Admission Office Contact**
- 📧 Email: admissions@nsit.ac.in
- 📱 Phone: +91-XXXXXXXXXX
- 🏫 Office: Ground Floor, Admin Block, NSIT Campus
- 🕐 Timings: Mon–Sat, 9:00 AM – 5:00 PM
"""

SYSTEM_PROMPT = """You are the official Admission Assistant for Narnarayan Shastri Institute of Technology (NSIT).

STRICT RULES — follow these without exception:
1. Answer ONLY using the context provided below from the official NSIT documents.
2. Do NOT use any outside knowledge, assumptions, or general information.
3. If the answer is NOT found in the context, respond EXACTLY with: "NOT_IN_DOCS"
4. Be concise, polite, and professional.
5. Never make up eligibility criteria, fees, dates, or any numbers.
6. If a question is partially answered, provide what you know and note what isn't in the documents.

Context from NSIT official documents:
{context}

Answer the student's question based strictly on the above context."""

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_client():
    return chromadb.Client(Settings(anonymized_telemetry=False))

def extract_pdf_text(uploaded_file) -> list[dict]:
    """Extract text chunks from PDF with page info."""
    chunks = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text:
                continue
            # Split into ~400 char chunks with overlap
            words = text.split()
            chunk_size, overlap = 80, 15
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk.strip()) > 30:
                    chunks.append({
                        "text": chunk.strip(),
                        "page": page_num,
                        "source": uploaded_file.name
                    })
    return chunks

def index_pdfs(files, embedder, chroma_client):
    """Index all PDFs into ChromaDB."""
    collection_name = "nsit_docs"
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    collection = chroma_client.create_collection(collection_name)

    all_chunks, all_ids, all_metas = [], [], []
    for f in files:
        chunks = extract_pdf_text(f)
        for i, c in enumerate(chunks):
            uid = hashlib.md5(f"{f.name}-{i}-{c['text'][:30]}".encode()).hexdigest()
            all_chunks.append(c["text"])
            all_ids.append(uid)
            all_metas.append({"source": c["source"], "page": c["page"]})

    if not all_chunks:
        return None, 0

    embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()
    collection.add(documents=all_chunks, embeddings=embeddings,
                   ids=all_ids, metadatas=all_metas)
    return collection, len(all_chunks)

def retrieve_context(query, collection, embedder, top_k=5):
    """Retrieve top-k relevant chunks for query."""
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return docs, metas, dists

def ask_groq(context, question, api_key):
    """Call Groq API with strict RAG prompt."""
    client = groq.Groq(api_key=api_key)
    prompt = SYSTEM_PROMPT.format(context=context)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": question}
        ],
        temperature=0.1,   # low temp = less hallucination
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "collection" not in st.session_state: st.session_state.collection = None
if "indexed"    not in st.session_state: st.session_state.indexed    = False
if "chunk_count" not in st.session_state: st.session_state.chunk_count = 0

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nsit-header">
  <div class="icon">🎓</div>
  <div>
    <h1>Narnarayan Shastri Institute of Technology</h1>
    <p>AI-Powered Admission Assistant · Ask anything about Admissions, Courses & Fees</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")

    api_key = st.text_input("🔑 Groq API Key", type="password",
                            placeholder="gsk_...",
                            help="Get free key at groq.com")

    st.markdown("---")
    st.markdown("### 📄 Upload Knowledge Base PDFs")
    uploaded_files = st.file_uploader(
        "Upload admission / course / fee PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="All answers will come strictly from these PDFs only."
    )

    if uploaded_files and api_key:
        if st.button("🚀 Index PDFs & Start Chatbot", use_container_width=True):
            with st.spinner("Reading & indexing PDFs..."):
                embedder   = load_embedder()
                chroma_cli = get_chroma_client()
                collection, count = index_pdfs(uploaded_files, embedder, chroma_cli)
                if collection:
                    st.session_state.collection  = collection
                    st.session_state.indexed     = True
                    st.session_state.chunk_count = count
                    st.session_state.messages    = []
                    st.success(f"✅ Indexed {count} chunks from {len(uploaded_files)} PDF(s)!")
                else:
                    st.error("❌ Could not extract text. Check your PDFs.")
    elif uploaded_files and not api_key:
        st.warning("⚠️ Please enter your Groq API key above.")

    st.markdown("---")
    # Status
    if st.session_state.indexed:
        st.markdown(f'<span class="status-ok">● Chatbot Active</span>', unsafe_allow_html=True)
        st.caption(f"{st.session_state.chunk_count} knowledge chunks loaded")
    else:
        st.markdown('<span class="status-off">● Not Ready</span>', unsafe_allow_html=True)
        st.caption("Upload PDFs + API key to start")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("This chatbot answers **only** from uploaded PDFs. It will never guess or use outside knowledge.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat area ─────────────────────────────────────────────────────────────────
chat_col, _ = st.columns([3, 0.01])

with chat_col:
    if not st.session_state.indexed:
        st.info("👈 **To get started:** Enter your Groq API key and upload your PDF documents in the sidebar, then click **Index PDFs & Start Chatbot**.")
        st.markdown("#### 💡 What can you ask?")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("🎓 **Admissions**\n- Eligibility criteria\n- Application process\n- Important dates")
        with cols[1]:
            st.markdown("💰 **Fees**\n- Course-wise fees\n- Scholarship info\n- Payment schedule")
        with cols[2]:
            st.markdown("📚 **Courses**\n- Available programs\n- Duration & seats\n- Specializations")
    else:
        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-user">🧑‍🎓 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                content = msg["content"]
                sources = msg.get("sources", "")
                st.markdown(
                    f'<div class="msg-bot">🤖 {content}'
                    + (f'<div class="source-tag">📄 Source: {sources}</div>' if sources else "")
                    + "</div>",
                    unsafe_allow_html=True
                )

        # Chat input
        if prompt := st.chat_input("Ask about admissions, courses, fees, eligibility..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Searching documents..."):
                embedder = load_embedder()
                docs, metas, dists = retrieve_context(
                    prompt, st.session_state.collection, embedder
                )

                # Confidence threshold — if best match is too far, treat as not found
                DISTANCE_THRESHOLD = 1.4
                if dists[0] > DISTANCE_THRESHOLD:
                    answer  = "NOT_IN_DOCS"
                    sources = ""
                else:
                    context = "\n\n".join(docs)
                    answer  = ask_groq(context, prompt, api_key)
                    sources = ", ".join(
                        set(f"{m['source']} (p.{m['page']})" for m in metas[:3])
                    )

            # Handle "not in docs"
            if "NOT_IN_DOCS" in answer or answer.strip() == "":
                display = (
                    "I'm sorry, I couldn't find information about that in the uploaded documents.\n\n"
                    + CONTACT_INFO
                )
                sources = ""
            else:
                display = answer

            st.session_state.messages.append({
                "role": "assistant",
                "content": display,
                "sources": sources
            })
            st.rerun()
