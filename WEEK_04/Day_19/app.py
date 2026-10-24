
import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    CollectionConfigurationError,
    ask_question,
    build_index,
)

load_dotenv()


def _format_answer(answer: str) -> str:
    """Replace HTML line-break tags with Markdown-friendly separators."""
    return re.sub(r"<br\s*/?>", "; ", answer, flags=re.IGNORECASE)


def _render_answer_and_citations(message: dict[str, Any]) -> None:
    """Render a chat answer followed by citation verification statuses."""
    citations = message.get("citations", [])
    if citations and all(not citation.get("verified", False) for citation in citations):
        st.warning("All citations are unverified.")

    answer = message.get("content", "")
    if message.get("found") is False:
        answer = "I could not find this in the uploaded documents"
    st.markdown(_format_answer(answer))

    if citations:
        st.markdown("**Citations**")
        for citation in citations:
            if citation.get("verified"):
                st.caption("✅ VERIFIED")
            else:
                st.caption("⚠️ UNVERIFIED")
            st.text(
                f"{citation.get('file', 'Unknown')} — "
                f"page {citation.get('page', '?')}"
            )
            st.text(f"Quote: {citation.get('quote', '')}")


def _file_content_hash(file_path: str) -> str:
    """Return a SHA-256 digest for a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deduplicate_pdf_files(
    pdf_paths: list[str],
    source_paths: list[str],
) -> tuple[list[str], list[str]]:
    """Keep the first path for each unique PDF content hash."""
    if len(pdf_paths) != len(source_paths):
        raise ValueError("Each PDF path must have a corresponding source path.")

    seen_hashes: set[str] = set()
    unique_pdf_paths: list[str] = []
    unique_source_paths: list[str] = []
    for pdf_path, source_path in zip(pdf_paths, source_paths):
        content_hash = _file_content_hash(pdf_path)
        if content_hash in seen_hashes:
            continue

        seen_hashes.add(content_hash)
        unique_pdf_paths.append(pdf_path)
        unique_source_paths.append(source_path)

    return unique_pdf_paths, unique_source_paths


st.set_page_config(
    page_title="Folder RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Folder RAG Chatbot")
st.write(
    "Ask questions across multiple PDFs. "
    "Answers include source documents and page numbers."
)

with st.sidebar:
    st.header("Document collection")

    uploads = st.file_uploader(
        "Upload a folder containing PDFs",
        accept_multiple_files="directory",
    )
    st.caption("All PDFs in the selected folder will be processed.")

    folder_path = st.text_input(
        "Or enter a local folder path",
        placeholder=r"C:\Documents\HRPolicies",
    )

    process = st.button(
        "Process PDFs",
        type="primary",
        use_container_width=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

if "indexed_pdf_files" not in st.session_state:
    st.session_state.indexed_pdf_files = None

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.active_collection = None
    st.session_state.indexed_pdf_files = None

collection_name = f"my_pdf_{st.session_state.session_id}"


if process:
    candidate_pdf_paths = []
    candidate_pdf_names = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded in uploads or []:
            if Path(uploaded.name).suffix.lower() != ".pdf":
                continue

            relative_path = PurePosixPath(
                uploaded.name.replace("\\", "/")
            )
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Invalid uploaded file path: {uploaded.name}"
                )

            path = Path(temp_dir).joinpath(*relative_path.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            contents = uploaded.getvalue()
            path.write_bytes(contents)
            candidate_pdf_paths.append(str(path))
            candidate_pdf_names.append(relative_path.as_posix())

        if folder_path.strip():
            folder = Path(folder_path.strip()).expanduser()

            if not folder.is_dir():
                st.error("The specified folder does not exist.")
                st.stop()

            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() == ".pdf":
                    candidate_pdf_paths.append(str(path))
                    candidate_pdf_names.append(
                        path.relative_to(folder).as_posix()
                    )

        candidate_count = len(candidate_pdf_paths)
        pdf_paths, pdf_names = _deduplicate_pdf_files(
            candidate_pdf_paths,
            candidate_pdf_names,
        )
        duplicate_count = candidate_count - len(pdf_paths)
        if duplicate_count:
            st.info(f"Skipped {duplicate_count} duplicate PDF file(s).")

        if not pdf_paths:
            st.warning(
                "Upload PDFs or enter a valid folder path first."
            )
        else:
            with st.spinner(
                f"Indexing {len(pdf_paths)} PDF file(s)..."
            ):
                try:
                    result = build_index(
                        pdf_paths=pdf_paths,
                        collection_name=collection_name,
                        source_paths=pdf_names,
                    )

                    st.session_state.active_collection = collection_name
                    st.session_state.indexed_pdf_files = pdf_names
                    st.session_state.messages = []

                    st.success(
                        f"Found {len(pdf_paths)} PDF(s); indexed "
                        f"{result['files']} readable PDF(s), "
                        f"{result['pages']} page(s), "
                        f"{result['chunks']} chunks."
                    )

                    for skipped in result["skipped"]:
                        st.warning(skipped)

                except Exception as error:
                    st.error(f"Indexing failed: {error}")


st.subheader("Chat")

if not os.getenv("GROQ_API_KEY"):
    st.info(
        "Configure GROQ_API_KEY in your .env file before asking questions."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        _render_answer_and_citations(message)
        if message.get("inventory_count") is not None:
            st.caption(
                "Answered from the full document catalog "
                f"({message['inventory_count']} documents)."
            )

        if message.get("sources"):
            with st.expander("Sources used"):
                if message.get("rewritten_query"):
                    st.caption(
                        f"Search query: {message['rewritten_query']}"
                    )
                best_match = message["sources"][0].get(
                    "source_path",
                    message["sources"][0]["file"],
                )
                st.markdown(
                    f"**Best-matching PDF:** {best_match}"
                )
                for source in message["sources"]:
                    st.markdown(
                        f"**{source.get('source_path', source['file'])}** "
                        f"— page {source['page']}"
                    )
                    if source.get("reranker_score") is not None:
                        st.caption(
                            f"Reranker score: "
                            f"{source['reranker_score']:.4f}"
                        )
                    else:
                        st.caption(
                            "Reranker score unavailable; using fallback ranking."
                        )
                    st.markdown("**Expanded context**")
                    st.text(
                        source.get("expanded_text", source["preview"])
                    )

question = st.chat_input("Ask about your PDF documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.active_collection:
            st.warning(
                "Process your PDF collection before asking questions."
            )
        else:
            try:
                with st.spinner(
                    "Searching documents and generating an answer..."
                ):
                    chat_history = [
                        {
                            "role": history_message["role"],
                            "content": history_message["content"],
                        }
                        for history_message in st.session_state.messages[:-1]
                        if history_message["role"] in {"user", "assistant"}
                    ][-8:]
                    result = ask_question(
                        question=question,
                        collection_name=st.session_state.active_collection,
                        chat_history=chat_history,
                    )

                answer = _format_answer(result["answer"])
                answer_message = {
                    "content": answer,
                    "found": result.get("found"),
                    "citations": result.get("citations", []),
                }
                _render_answer_and_citations(answer_message)
                if result.get("inventory_count") is not None:
                    st.caption(
                        "Answered from the full document catalog "
                        f"({result['inventory_count']} documents)."
                    )

                if result["sources"]:
                    with st.expander("Sources used", expanded=True):
                        if result.get("rewritten_query"):
                            st.caption(
                                f"Search query: {result['rewritten_query']}"
                            )
                        best_match = result["sources"][0].get(
                            "source_path",
                            result["sources"][0]["file"],
                        )
                        st.markdown(
                            f"**Best-matching PDF:** {best_match}"
                        )
                        for source in result["sources"]:
                            st.markdown(
                                f"**{source.get('source_path', source['file'])}** "
                                f"— page {source['page']}"
                            )
                            if source.get("reranker_score") is not None:
                                st.caption(
                                    f"Reranker score: "
                                    f"{source['reranker_score']:.4f}"
                                )
                            else:
                                st.caption(
                                    "Reranker score unavailable; "
                                    "using fallback ranking."
                                )
                            st.markdown("**Expanded context**")
                            st.text(
                                source.get("expanded_text", source["preview"])
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "found": result.get("found"),
                    "citations": result.get("citations", []),
                    "sources": result["sources"],
                    "inventory_count": result.get("inventory_count"),
                    "rewritten_query": result.get("rewritten_query"),
                })

            except CollectionConfigurationError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"Could not answer: {error}")
