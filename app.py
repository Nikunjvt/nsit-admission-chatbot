import streamlit as st
import os
import hashlib
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
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

section[data-testid="stSidebar"] { background: #0d2340 !important; }
section[data-testid="stSidebar"] * { color: white !important; }

.status-ok  { background:#1a5c3a; color:white; padding:0.2rem 0.6rem; border-radius:20px; font-size:0.75rem; }
.status-off { background:#5c1a1a; color:white; padding:0.2rem 0.6rem; border-radius:20px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CONTACT_INFO = """
📞 **Admission Office — Narnarayan Shastri Institute of Technology**
- 📧 Email: admissions@nsit.ac.in
- 📱 Phone: +91-XXXXXXXXXX
- 🏫 Office: Ground Floor, Admin Block, NSIT Campus
- 🕐 Timings: Mon–Sat, 9:00 AM – 5:00 PM
"""

SYSTEM_PROMPT = """You are the official Admission Assistant for Narnarayan Shastri Institute of Technology (NSIT).

STRICT RULES — follow without exception:
1. Answer ONLY using the context provided below from official NSIT documents.
2. Do NOT use any outside knowledge, assumptions, or general information.
3. If the answer is NOT found in the context, respond EXACTLY with the word: NOT_IN_DOCS
4. Be concise, polite, and professional.
5. Never make up eligibility criteria, fees, dates, or any numbers.

Context from NSIT official documents:
{context}

Answer the student's question strictly based on the above context only."""

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_chunks(uploaded_file):
    chunks, metas = [], []
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text:
                continue
            words = text.split()
            size, overlap = 80, 15
            for i in range(0, len(words), size - overlap):
                chunk = " ".join(words[i:i + size]).strip()
                if len(chunk) > 30:
                    chunks.append(chunk)
                    metas.append({"source": uploaded_file.name, "page": page_num})
    return chunks, metas

# ── FAISS index builder ───────────────────────────────────────────────────────
def build_index(files, embedder):
    all_chunks, all_metas = [], []
    for f in files:
        chunks, metas = extract_chunks(f)
        all_chunks.extend(chunks)
        all_metas.extend(metas)

    if not all_chunks:
        return None, None, None, 0

    embeddings = embedder.encode(all_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine similarity
    index.add(embeddings)

    return index, all_chunks, all_metas, len(all_chunks)

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query, index, chunks, metas, embedder, top_k=5):
    q_emb = embedder.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], metas[idx], float(score)))
    return results

# ── Groq call ─────────────────────────────────────────────────────────────────
def ask_groq(context, question, api_key):
    client = groq.Groq(api_key=api_key)
    prompt = SYSTEM_PROMPT.format(context=context)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": question}
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "messages": [],
    "faiss_index": None,
    "chunks": None,
    "metas": None,
    "indexed": False,
    "chunk_count": 0,
    "api_key": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nsit-header">
  <div class="icon">🎓</div>
  <div>
    <h1>Narnarayan Shastri Institute of Technology</h1>
    <p>AI-Powered Admission Assistant · Admissions · Courses · Fee Structure</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")

    api_key = st.text_input("🔑 Groq API Key", type="password",
                            placeholder="gsk_...",
                            help="Get free key at console.groq.com")

    st.markdown("---")
    st.markdown("### 📄 Upload Knowledge Base PDFs")
    uploaded_files = st.file_uploader(
        "Upload admission / course / fee PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Chatbot will answer ONLY from these PDFs."
    )

    if uploaded_files and api_key:
        if st.button("🚀 Index PDFs & Start Chatbot", use_container_width=True):
            with st.spinner("Reading & indexing PDFs... please wait"):
                embedder = load_embedder()
                index, chunks, metas, count = build_index(uploaded_files, embedder)
                if index is not None:
                    st.session_state.faiss_index  = index
                    st.session_state.chunks       = chunks
                    st.session_state.metas        = metas
                    st.session_state.indexed      = True
                    st.session_state.chunk_count  = count
                    st.session_state.messages     = []
                    st.session_state.api_key      = api_key  # ← save key in session
                    st.success(f"✅ Indexed {count} chunks from {len(uploaded_files)} PDF(s)!")
                else:
                    st.error("❌ Could not extract text from PDFs.")
    elif uploaded_files and not api_key:
        st.warning("⚠️ Please enter your Groq API key above first.")

    st.markdown("---")
    if st.session_state.indexed:
        st.markdown('<span class="status-ok">● Chatbot Active</span>', unsafe_allow_html=True)
        st.caption(f"{st.session_state.chunk_count} knowledge chunks loaded")
    else:
        st.markdown('<span class="status-off">● Not Ready</span>', unsafe_allow_html=True)
        st.caption("Upload PDFs + API key to activate")

    st.markdown("---")
    st.caption("🔒 Answers strictly from your uploaded PDFs only. No hallucination.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("👈 **To get started:** Enter your Groq API key + upload PDFs in the sidebar → click Index PDFs.")
    st.markdown("#### 💡 Sample Questions You Can Ask")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("🎓 **Admissions**\n- Eligibility criteria\n- Application process\n- Important dates")
    with c2:
        st.markdown("💰 **Fee Structure**\n- Course-wise fees\n- Scholarship info\n- Payment schedule")
    with c3:
        st.markdown("📚 **Courses**\n- Available programs\n- Duration & seats\n- Specializations")
else:
    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">🧑‍🎓 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            src = msg.get("sources", "")
            st.markdown(
                f'<div class="msg-bot">🤖 {msg["content"]}'
                + (f'<div class="source-tag">📄 Source: {src}</div>' if src else "")
                + "</div>",
                unsafe_allow_html=True
            )

    # Chat input
    if prompt := st.chat_input("Ask about admissions, courses, fees, eligibility..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Searching documents..."):
            embedder = load_embedder()
            results  = retrieve(
                prompt,
                st.session_state.faiss_index,
                st.session_state.chunks,
                st.session_state.metas,
                embedder
            )

            THRESHOLD = 0.30  # cosine similarity threshold
            if not results or results[0][2] < THRESHOLD:
                answer  = "NOT_IN_DOCS"
                sources = ""
            else:
                context = "\n\n".join([r[0] for r in results])
                answer  = ask_groq(context, prompt, st.session_state.api_key)
                sources = ", ".join(
                    set(f"{r[1]['source']} (p.{r[1]['page']})" for r in results[:3])
                )

        if "NOT_IN_DOCS" in answer or answer.strip() == "":
            display = (
                "I'm sorry, I couldn't find that information in the uploaded documents.\n\n"
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
