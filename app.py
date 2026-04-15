import logging
from pathlib import Path

import streamlit as st

from src.document_loader import ResearchPaperLoader
from src.vector_store import ResearchVectorStore
from src.retriever import ResearchPaperRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("research_paper_finder")

VECTOR_STORE_DIR = "faiss_store"
DATA_PATH = "dataset.csv"


def log_error(e: Exception) -> str:
    logger.exception("Application error")
    return f"Error: {str(e)}"


def get_documents():
    try:
        st.info("Loading research papers...")
        loader = ResearchPaperLoader(DATA_PATH)
        documents = loader.create_documents()
        st.success(f"{len(documents)} research papers loaded successfully.")
        return documents
    except Exception as e:
        st.error(log_error(e))
        return None


def create_new_vector_store():
    try:
        st.info("Creating new vector store...")
        documents = get_documents()
        if not documents:
            return None

        vector_store = ResearchVectorStore(store_path=VECTOR_STORE_DIR)
        st.info("Generating embeddings and building FAISS index...")
        vector_store.create_vector_store(documents)
        st.success("FAISS vector store created and saved successfully.")
        return vector_store
    except Exception as e:
        st.error(log_error(e))
        return None


def load_existing_vector_store():
    try:
        st.info("Loading existing FAISS vector store...")
        vector_store = ResearchVectorStore.Load(VECTOR_STORE_DIR)
        st.success("FAISS vector store loaded successfully.")
        return vector_store
    except Exception as e:
        st.error(log_error(e))
        return None


def initialize_retrieval_system():
    try:
        faiss_index_path = Path(VECTOR_STORE_DIR) / "faiss_index.bin"
        metadata_path = Path(VECTOR_STORE_DIR) / "metadata.pkl"
        documents_path = Path(VECTOR_STORE_DIR) / "documents.pkl"

        if all(p.exists() for p in [faiss_index_path, metadata_path, documents_path]):
            vector_store = load_existing_vector_store()
        else:
            vector_store = create_new_vector_store()

        if not vector_store:
            return None

        st.info("Initializing paper retriever...")
        retriever = ResearchPaperRetriever(vector_store)
        return retriever

    except Exception as e:
        st.error(log_error(e))
        return None


def render_search_results(query, nod, use_recency=False):
    try:
        with st.spinner("Searching for relevant papers..."):
            if use_recency:
                results = st.session_state.retriever.retrieve_papers_with_recency(query, k=nod)
            else:
                results = st.session_state.retriever.retrieve_papers(query, k=nod)

        if not results:
            st.info("No matching papers found. Try a different query.")
            return

        st.subheader(f"Found {len(results)} relevant papers")

        for i, paper in enumerate(results, 1):
            title = paper.get("title", "Untitled")
            year = paper.get("year", "N/A")

            with st.expander(f"{i}. {title} ({year})"):
                st.write(f"**Authors:** {paper.get('authors', '')}")
                st.write(f"**Published in:** {paper.get('venue', '')}")
                st.write(f"**Citations:** {paper.get('citations', 0)}")
                st.write(f"**Abstract:** {paper.get('abstract', '')}")
                st.write(f"**Similarity Score:** {paper.get('similarity_score', 0.0):.4f}")

    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")


def main():
    st.set_page_config(page_title="Academic Research Paper Finder", page_icon="📚", layout="wide")

    st.title("📚 Academic Research Paper Finder")
    st.write(
        "Find relevant academic papers using FAISS-based semantic retrieval. "
        "Enter your research topic or question to discover related papers."
    )

    if "retriever" not in st.session_state:
        st.session_state.retriever = initialize_retrieval_system()

    query = st.text_input(
        "Enter your research topic or question:",
        placeholder="e.g. Quantum computing for cryptography"
    )

    nod = st.number_input(
        "Number of documents",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    use_recency = st.checkbox(
        "Prioritize recent papers",
        value=False,
        help="When checked, results consider both similarity and publication year, then show the selected papers in recency order."
    )

    search_button = st.button("Search")

    if search_button:
        if st.session_state.retriever is None:
            st.error("Retriever could not be initialized.")
        else:
            render_search_results(query, int(nod), use_recency)


if __name__ == "__main__":
    main()