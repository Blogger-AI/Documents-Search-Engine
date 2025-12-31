import streamlit as st
import os
import re

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="AI Document Search Engine",
    page_icon="📄",
    layout="centered"
)


# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("📘 Project Information")
st.sidebar.markdown("""
**Project:** Document-Based Search Engine  
**Domain:** Artificial Intelligence  
**Database:** PDF Knowledge Base  
**Technique:** TF-IDF + Similarity Search  
**Answer Type:** Contextual Extraction  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍🎓 Search Engine")


# ---------------------------------
# Configuration
# ---------------------------------
DOC_FOLDER = "knowledge_base"
SIMILARITY_THRESHOLD = 0.1   # prevents irrelevant answers


# ---------------------------------
# Helper Functions
# ---------------------------------
def extract_text_from_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def split_sentences(text):
    """
    Regex-based sentence splitting.
    Lightweight and stable (no NLTK).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


# ---------------------------------
# Load PDF Documents (Database)
# ---------------------------------
documents = []
document_names = []

for file in os.listdir(DOC_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(DOC_FOLDER, file)
        documents.append(extract_text_from_pdf(path))
        document_names.append(file)


# ---------------------------------
# Create Search Index
# ---------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(documents)


# ---------------------------------
# Main UI
# ---------------------------------
st.title("📄 AI Document Search Engine")
st.markdown(
    "Search from a **domain-specific AI knowledge base** and receive "
    "**contextual answers** extracted directly from documents."
)

st.markdown("---")

query = st.text_input(
    "🔍 Enter your question",
    placeholder="Example: What is unsupervised learning?"
)


# ---------------------------------
# Search & Answer Extraction
# ---------------------------------
if query:
    with st.spinner("Searching documents and extracting answer..."):

        # Vectorize query
        query_vector = vectorizer.transform([query])

        # -------- Document-Level Search --------
        similarities = cosine_similarity(query_vector, document_vectors)
        best_doc_index = similarities.argmax()
        best_doc_score = similarities[0][best_doc_index]

        # Threshold check (IMPORTANT)
        if best_doc_score < SIMILARITY_THRESHOLD:
            st.warning("❌ No relevant answer found for this question.")
            st.stop()

        best_doc_text = documents[best_doc_index]
        best_doc_name = document_names[best_doc_index]

        # -------- Sentence-Level Search --------
        sentences = split_sentences(best_doc_text)
        sentence_vectors = vectorizer.transform(sentences)
        sentence_similarities = cosine_similarity(query_vector, sentence_vectors)

        idx = sentence_similarities.argmax()
        best_sentence_score = sentence_similarities[0][idx]

        # Optional stricter check
        if best_sentence_score < SIMILARITY_THRESHOLD:
            st.warning("❌ The question does not match the document content.")
            st.stop()

        # -------- Contextual Answer (Option 1) --------
        start = max(0, idx - 1)
        end = min(len(sentences), idx + 2)
        contextual_answer = " ".join(sentences[start:end])


    # ---------------------------------
    # Display Result
    # ---------------------------------
    st.success("Answer extracted successfully!")

    st.markdown("### 📌 Search Result")
    st.markdown(f"**📄 Document Name:** `{best_doc_name}`")
    st.markdown("**🧠 Extracted Answer (with context):**")
    st.info(contextual_answer)


# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption("© SEM Project | Document-Based AI Search Engine")
