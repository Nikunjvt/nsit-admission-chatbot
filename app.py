import streamlit as st
import pdfplumber
import groq
import math
import re
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSIT Admission Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.header-box {
    background: linear-gradient(135deg, #1a3a5c, #0d2340);
    color: white; padding: 1.4rem 1.8rem; border-radius: 10px;
    margin-bottom: 1.2rem;
}
.header-box h2 { margin: 0; font-size: 1.4rem; }
.header-box p  { margin: 0.2rem 0 0; font-size: 0.8rem; opacity: 0.75; }
.bubble-user {
    background: #1a3a5c; color: white;
    padding: 0.7rem 1rem; border-radius: 16px 16px 4px 16px;
    margin: 0.4rem 0 0.4rem 20%; font-size: 0.9rem;
}
.bubble-bot {
    background: #ffffff; color: #111;
    padding: 0.7rem 1rem; border-radius: 16px 16px 16px 4px;
    margin: 0.4rem 20% 0.4rem 0; font-size: 0.9rem;
    border: 1px solid #ddd; line-height: 1.6;
}
.src { font-size: 0.7rem; color: #888; margin-top: 0.3rem; }
section[data-testid="stSidebar"] { background: #0d2340 !important; }
section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CONTACT = (
    "I could not find that in our documents. Please contact:\n\n"
    "**Admission Office — NSIT**\n"
    "📧 admissions@nsit.ac.in\n"
    "📞 +91-XXXXXXXXXX\n"
    "🕐 Mon–Sat, 9 AM – 5 PM"
)

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE TF-IDF RETRIEVAL  (zero heavy dependencies)
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

def build_tfidf(chunks):
    N = len(chunks)
    df = Counter()
    tokenized = []
    for chunk in chunks:
        tokens = set(tokenize(chunk[0]))
        tokenized.append(tokens)
        for t in tokens:
            df[t] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) for t in df}
    return tokenized, idf

def score_chunk(query_tokens, chunk_tokens, idf):
    score = 0.0
    for t in query_tokens:
        if t in chunk_tokens:
            score += idf.get(t, 0)
    return score

def retrieve(query, chunks, tokenized, idf, top_k=3):
    q_tokens = set(tokenize(query))
    scores = [score_chunk(q_tokens, ct, idf) for ct in tokenized]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i][0], chunks[i][1], scores[i]) for i in top_idx]

# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_chunks(uploaded_file):
    chunks = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text:
                continue
            words = text.split()
            size, overlap = 60, 10
            for i in range(0, len(words), size - overlap):
                chunk = " ".join(words[i: i + size]).strip()
                if len(chunk) > 30:
                    meta = uploaded_file.name + " (page " + str(page_num) + ")"
                    chunks.append((chunk, meta))
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# GROQ CALL  — kept minimal and safe
# ─────────────────────────────────────────────────────────────────────────────
def ask_groq(context, question, api_key):
    context = context[:800]   # hard cap — well within token limits
    msg = (
        "You are an admission assistant for Narnarayan Shastri Institute of Technology. "
        "Use ONLY the context below to answer. "
        "If the answer is not in the context write: NOT_IN_DOCS\n\n"
        "Context: " + context + "\n\nQuestion: " + question
    )
    client = groq.Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": msg}],
        temperature=0.0,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "chunks"     not in st.session_state: st.session_state.chunks     = []
if "tokenized"  not in st.session_state: st.session_state.tokenized  = []
if "idf"        not in st.session_state: st.session_state.idf        = {}
if "indexed"    not in st.session_state: st.session_state.indexed    = False
if "api_key"    not in st.session_state: st.session_state.api_key    = ""

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h2>🎓 Narnarayan Shastri Institute of Technology</h2>
  <p>AI Admission Assistant &nbsp;·&nbsp; Admissions &nbsp;·&nbsp; Courses &nbsp;·&nbsp; Fee Structure</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    api_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("### 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Admission / Course / Fee PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Index PDFs & Activate", use_container_width=True):
        if not api_input:
            st.error("Please enter your Groq API key.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Reading PDFs..."):
                all_chunks = []
                for f in uploaded_files:
                    all_chunks.extend(extract_chunks(f))
                if all_chunks:
                    tokenized, idf = build_tfidf(all_chunks)
                    st.session_state.chunks    = all_chunks
                    st.session_state.tokenized = tokenized
                    st.session_state.idf       = idf
                    st.session_state.indexed   = True
                    st.session_state.api_key   = api_input
                    st.session_state.messages  = []
                    st.success("Done! " + str(len(all_chunks)) + " chunks indexed.")
                else:
                    st.error("Could not extract text. Check your PDFs.")

    st.markdown("---")
    if st.session_state.indexed:
        st.success("● Chatbot Active")
        st.caption(str(len(st.session_state.chunks)) + " knowledge chunks loaded")
    else:
        st.warning("● Not Ready — upload PDFs above")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("👈 Enter your Groq API key, upload PDFs, then click **Index PDFs & Activate**.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🎓 Admissions**\n- Eligibility criteria\n- Application process\n- Key dates")
    with col2:
        st.markdown("**💰 Fee Structure**\n- Course-wise fees\n- Scholarships\n- Payment schedule")
    with col3:
        st.markdown("**📚 Courses**\n- Available programs\n- Duration & seats\n- Specializations")
else:
    # Render history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown('<div class="bubble-user">🧑 ' + msg["content"] + '</div>', unsafe_allow_html=True)
        else:
            src_html = ""
            if msg.get("sources"):
                src_html = '<div class="src">📄 ' + msg["sources"] + '</div>'
            st.markdown('<div class="bubble-bot">🤖 ' + msg["content"] + src_html + '</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask about admissions, courses, fees, eligibility...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Searching documents..."):
            results = retrieve(
                user_input,
                st.session_state.chunks,
                st.session_state.tokenized,
                st.session_state.idf,
                top_k=3,
            )

            best_score = results[0][2] if results else 0

            if best_score < 0.01:
                display = CONTACT
                sources = ""
            else:
                context = "\n\n".join([r[0] for r in results])
                answer  = ask_groq(context, user_input, st.session_state.api_key)
                if "NOT_IN_DOCS" in answer or not answer:
                    display = CONTACT
                    sources = ""
                else:
                    display = answer
                    sources = " | ".join(set(r[1] for r in results))

        st.session_state.messages.append({
            "role": "assistant",
            "content": display,
            "sources": sources,
        })
        st.rerun()
