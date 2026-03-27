#!/usr/bin/env python3
"""Streamlit web UI for the research assistant."""

import sys
import uuid
from pathlib import Path
from typing import Optional

# Allow running directly: streamlit run ui/web.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Assistant | 研究助手",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_config() -> dict:
    import yaml  # type: ignore

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


@st.cache_resource
def _get_ollama_client(base_url: str, model: str, temperature: float, max_tokens: int):
    from src.llm.ollama_client import OllamaClient  # type: ignore

    return OllamaClient(
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@st.cache_resource
def _get_vector_store(persist_dir: str, collection: str, embedding_model: str):
    from src.rag.embeddings import EmbeddingManager  # type: ignore
    from src.rag.vector_store import VectorStore  # type: ignore

    emb = EmbeddingManager(model_name=embedding_model)
    return VectorStore(
        persist_directory=persist_dir,
        collection_name=collection,
        embedding_manager=emb,
    )


@st.cache_resource
def _get_db(db_path: str, max_history: int):
    from src.memory.conversation_db import ConversationDB  # type: ignore

    return ConversationDB(db_path=db_path, max_history=max_history)


def _load_prompts() -> dict:
    import json

    prompts_path = Path(__file__).parent.parent / "config" / "prompts.json"
    try:
        with open(prompts_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "domain" not in st.session_state:
        st.session_state.domain = "machine_learning"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(cfg: dict) -> dict:
    """Render the sidebar controls and return the active settings."""
    st.sidebar.title("⚙️ 设置 / Settings")

    llm_cfg = cfg.get("llm", {})
    rag_cfg = cfg.get("rag", {})

    # Model selection
    st.sidebar.subheader("🤖 模型 / Model")
    model = st.sidebar.text_input(
        "Model name", value=llm_cfg.get("model", "llama2"), key="sidebar_model"
    )
    base_url = st.sidebar.text_input(
        "Ollama URL", value=llm_cfg.get("base_url", "http://localhost:11434"), key="sidebar_url"
    )
    temperature = st.sidebar.slider(
        "Temperature", 0.0, 1.0, float(llm_cfg.get("temperature", 0.7)), step=0.05
    )
    max_tokens = st.sidebar.slider(
        "Max tokens", 256, 4096, int(llm_cfg.get("max_tokens", 2048)), step=128
    )

    # Domain selection
    st.sidebar.subheader("🔬 领域 / Domain")
    domain = st.sidebar.selectbox(
        "Research domain",
        ["machine_learning", "statistics", "mathematics", "transfer_learning", "general"],
        index=["machine_learning", "statistics", "mathematics", "transfer_learning", "general"].index(
            st.session_state.domain
        ),
    )
    st.session_state.domain = domain

    # Connection status
    st.sidebar.subheader("📡 状态 / Status")
    client = _get_ollama_client(base_url, model, temperature, max_tokens)
    if client.check_connection():
        st.sidebar.success("✅ Ollama connected")
    else:
        st.sidebar.error("❌ Ollama not reachable")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Session: `{st.session_state.session_id}`")
    if st.sidebar.button("🗑️ Clear history"):
        st.session_state.chat_history = []
        st.rerun()

    return {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": rag_cfg.get("top_k", 5),
        "embedding_model": rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
    }


# ---------------------------------------------------------------------------
# Tab: Chat
# ---------------------------------------------------------------------------

def _tab_chat(settings: dict, cfg: dict) -> None:
    st.header("💬 智能对话 / Chat")

    client = _get_ollama_client(
        settings["base_url"],
        settings["model"],
        settings["temperature"],
        settings["max_tokens"],
    )
    db = _get_db(
        cfg.get("memory", {}).get("db_path", "./data/memory.db"),
        cfg.get("memory", {}).get("max_history", 100),
    )

    # Display history
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    use_rag = st.toggle("🔍 Use RAG (search knowledge base)", value=False)

    if prompt := st.chat_input("Ask a research question…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        db.save_message(
            st.session_state.session_id, "user", prompt, domain=st.session_state.domain
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                if use_rag:
                    vs_cfg = cfg.get("vector_store", {})
                    vs = _get_vector_store(
                        vs_cfg.get("persist_directory", "./data/chroma_db"),
                        vs_cfg.get("collection_name", "research_papers"),
                        settings["embedding_model"],
                    )
                    from src.rag.retriever import Retriever  # type: ignore

                    retriever = Retriever(vs, client, k=settings["top_k"])
                    result = retriever.rag_query(prompt)
                    answer = result["answer"]

                    if result["sources"]:
                        sources_md = "\n".join(
                            f"- {s['source']} (p. {s['page']})" for s in result["sources"]
                        )
                        answer += f"\n\n---\n**Sources:**\n{sources_md}"
                else:
                    messages = [{"role": t["role"], "content": t["content"]}
                                for t in st.session_state.chat_history]
                    try:
                        answer = client.chat(messages, model=settings["model"])
                    except Exception as exc:
                        answer = f"⚠️ Error: {exc}"

            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        db.save_message(
            st.session_state.session_id, "assistant", answer, domain=st.session_state.domain
        )


# ---------------------------------------------------------------------------
# Tab: Writing
# ---------------------------------------------------------------------------

def _tab_writing(settings: dict, cfg: dict) -> None:
    st.header("✍️ 写作助手 / Writing Assistant")

    prompts = _load_prompts()
    client = _get_ollama_client(
        settings["base_url"],
        settings["model"],
        settings["temperature"],
        settings["max_tokens"],
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        content_type = st.selectbox(
            "Content type / 内容类型",
            ["Abstract", "Introduction", "Related Work", "Methodology",
             "Experiments", "Conclusion", "Cover Letter"],
        )
        topic = st.text_input("Topic / 主题", placeholder="e.g. LoRA fine-tuning for NLP")
        requirements = st.text_area(
            "Additional requirements / 额外要求",
            placeholder="Mention specific techniques, length, tone…",
            height=100,
        )

    with col2:
        grammar_text = st.text_area(
            "Grammar Check / 语法检查 (paste text here)",
            height=200,
        )

    col_a, col_b, _ = st.columns([1, 1, 3])
    generate_btn = col_a.button("✨ Generate / 生成", type="primary")
    grammar_btn = col_b.button("🔍 Check Grammar / 检查语法")

    if generate_btn and topic:
        template = prompts.get("writing_assistant", {}).get("template", "Write {content_type} about {topic}.")
        prompt = template.format(
            content_type=content_type,
            topic=topic,
            context="",
            requirements=requirements,
        )
        with st.spinner("Generating…"):
            try:
                result = client.generate(prompt, model=settings["model"])
            except Exception as exc:
                result = f"⚠️ Error: {exc}"
        st.subheader("Result / 结果")
        st.markdown(result)
        st.download_button("⬇️ Download", result, file_name="writing_output.md")

    if grammar_btn and grammar_text:
        template = prompts.get("grammar_check", {}).get("template", "Check grammar:\n{text}")
        prompt = template.format(text=grammar_text)
        with st.spinner("Checking…"):
            try:
                result = client.generate(prompt, model=settings["model"])
            except Exception as exc:
                result = f"⚠️ Error: {exc}"
        st.subheader("Grammar Check Result / 检查结果")
        st.markdown(result)


# ---------------------------------------------------------------------------
# Tab: Review
# ---------------------------------------------------------------------------

def _tab_review(settings: dict, cfg: dict) -> None:
    st.header("📋 论文审稿 / Paper Review")

    prompts = _load_prompts()
    client = _get_ollama_client(
        settings["base_url"],
        settings["model"],
        settings["temperature"],
        settings["max_tokens"],
    )

    review_type = st.radio(
        "Review type / 审稿类型",
        ["Full Paper Review", "Rebuttal Helper", "Summary"],
        horizontal=True,
    )

    uploaded = st.file_uploader("Upload paper (PDF or TXT) / 上传论文", type=["pdf", "txt"])
    if uploaded is None:
        st.info("Upload a paper to get started.")
        return

    text_input = st.text_area(
        "Or paste paper text / 或粘贴论文文本",
        height=200,
        placeholder="Paste abstract or paper excerpt…",
    )

    reviewer_comment = ""
    if review_type == "Rebuttal Helper":
        reviewer_comment = st.text_area("Reviewer comment / 审稿人意见", height=100)

    if st.button("🚀 Analyse / 分析", type="primary"):
        with st.spinner("Processing…"):
            if uploaded:
                raw_bytes = uploaded.read()
                if uploaded.name.endswith(".pdf"):
                    import tempfile, os

                    # Write to a named temp file so PyMuPDF can open it
                    tmp_path = Path("./data/cache") / f"_upload_{uuid.uuid4().hex}.pdf"
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(raw_bytes)
                    try:
                        from src.utils.pdf_processor import PDFProcessor  # type: ignore

                        processor = PDFProcessor()
                        paper_text = processor.extract_text(tmp_path)[:6000]
                    finally:
                        tmp_path.unlink(missing_ok=True)
                else:
                    paper_text = raw_bytes.decode("utf-8", errors="replace")[:6000]
            else:
                paper_text = text_input[:6000]

            if not paper_text.strip():
                st.warning("No text found.")
                return

            if review_type == "Full Paper Review":
                template = prompts.get("paper_review", {}).get("template", "Review:\n{paper_content}")
                prompt = template.format(paper_content=paper_text)
            elif review_type == "Rebuttal Helper":
                template = prompts.get("rebuttal_helper", {}).get(
                    "template", "Write rebuttal for:\n{reviewer_comment}\n\nPaper:\n{paper_context}"
                )
                prompt = template.format(
                    reviewer_comment=reviewer_comment,
                    paper_context=paper_text,
                    additional_info="",
                )
            else:
                template = prompts.get("summary", {}).get("template", "Summarise:\n{paper_content}")
                prompt = template.format(paper_content=paper_text)

            try:
                result = client.generate(prompt, model=settings["model"])
            except Exception as exc:
                result = f"⚠️ Error: {exc}"

        st.subheader("Result / 结果")
        st.markdown(result)
        st.download_button("⬇️ Download Review", result, file_name="review_output.md")


# ---------------------------------------------------------------------------
# Tab: Knowledge Base
# ---------------------------------------------------------------------------

def _tab_knowledge_base(settings: dict, cfg: dict) -> None:
    st.header("📚 知识库 / Knowledge Base")

    vs_cfg = cfg.get("vector_store", {})
    rag_cfg = cfg.get("rag", {})

    vs = _get_vector_store(
        vs_cfg.get("persist_directory", "./data/chroma_db"),
        vs_cfg.get("collection_name", "research_papers"),
        settings["embedding_model"],
    )

    info = vs.get_collection_info()
    col1, col2, col3 = st.columns(3)
    col1.metric("Collection", info["collection_name"])
    col2.metric("Documents", info["document_count"])
    col3.metric("Storage", info["persist_directory"])

    st.divider()

    st.subheader("📤 Add Papers / 添加论文")
    uploaded_files = st.file_uploader(
        "Upload PDF files / 上传PDF文件",
        type=["pdf"],
        accept_multiple_files=True,
    )
    domain_tag = st.selectbox(
        "Domain tag", ["machine_learning", "statistics", "mathematics", "transfer_learning"]
    )

    if st.button("📥 Index Papers / 索引论文", disabled=not uploaded_files):
        from src.rag.document_loader import DocumentLoader  # type: ignore

        loader = DocumentLoader(
            chunk_size=rag_cfg.get("chunk_size", 1000),
            chunk_overlap=rag_cfg.get("chunk_overlap", 200),
        )
        progress = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            tmp_path = Path("./data/cache") / f"_upload_{uuid.uuid4().hex}.pdf"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded_file.read())
            try:
                docs = loader.load_and_split(tmp_path)
                for doc in docs:
                    doc.metadata["domain"] = domain_tag
                    doc.metadata["original_filename"] = uploaded_file.name
                vs.add_documents(docs)
                st.success(f"✅ Indexed {len(docs)} chunks from {uploaded_file.name}")
            except Exception as exc:
                st.error(f"Failed to index {uploaded_file.name}: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)
            progress.progress((i + 1) / len(uploaded_files))
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("🔍 Search Knowledge Base / 搜索知识库")
    search_query = st.text_input("Search query / 搜索查询")
    if st.button("Search", disabled=not search_query):
        results = vs.similarity_search(search_query, k=settings["top_k"])
        if results:
            for j, doc in enumerate(results, 1):
                with st.expander(
                    f"Result {j}: {doc.metadata.get('source', 'unknown')} "
                    f"(page {doc.metadata.get('page', 'N/A')})"
                ):
                    st.markdown(doc.page_content)
        else:
            st.info("No results found.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()
    cfg = _load_config()
    settings = _render_sidebar(cfg)

    tabs = st.tabs(["💬 智能对话", "✍️ 写作助手", "📋 论文审稿", "📚 知识库"])

    with tabs[0]:
        _tab_chat(settings, cfg)
    with tabs[1]:
        _tab_writing(settings, cfg)
    with tabs[2]:
        _tab_review(settings, cfg)
    with tabs[3]:
        _tab_knowledge_base(settings, cfg)


if __name__ == "__main__":
    main()
