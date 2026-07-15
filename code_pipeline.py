import os
import shutil
import stat
import subprocess

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# Models are loaded once when this file is imported.
# ---------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile")

CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php"}
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", "venv", ".venv"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "poetry.lock"}

EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".jsx": Language.JS,
    ".tsx": Language.TS,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
}


def _remove_readonly(func, path, exc_info):
    """Windows sometimes marks files inside .git as read-only; this clears
    that flag so shutil.rmtree can actually delete them."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repo(repo_url, local_path="cloned_repo"):
    if os.path.exists(local_path):
        shutil.rmtree(local_path, onerror=_remove_readonly)

    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, local_path],
        check=True
    )
    return local_path


def collect_source_files(repo_path):
    source_files = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            if filename in SKIP_FILES:
                continue

            ext = os.path.splitext(filename)[1]
            if ext in CODE_EXTENSIONS:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, repo_path).replace(os.sep, "/")
                source_files.append((relative_path, full_path))

    return source_files


def build_code_documents(source_files):
    all_chunks = []

    for relative_path, full_path in source_files:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            continue

        ext = os.path.splitext(relative_path)[1]
        language = EXTENSION_TO_LANGUAGE.get(ext)

        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=800,
                chunk_overlap=100
            )
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        doc = Document(page_content=content, metadata={"file_path": relative_path})
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    return all_chunks


def build_vectorstore(chunks):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="codebase"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})


def index_repo(repo_url):
    """Runs the full ingestion pipeline for one repo URL and returns
    (retriever, source_files, num_chunks) — everything app.py needs."""
    repo_path = clone_repo(repo_url)
    source_files = collect_source_files(repo_path)
    chunks = build_code_documents(source_files)
    retriever = build_vectorstore(chunks)
    return retriever, source_files, len(chunks)


def generate_answer(question, retriever):
    retrieved_docs = retriever.invoke(question)

    context = ""
    for doc in retrieved_docs:
        file_path = doc.metadata["file_path"]
        context += f"File: {file_path}\n```\n{doc.page_content}\n```\n\n"

    prompt = f"""You are a code assistant answering questions about a codebase.
Use only the code excerpts below to answer. For every claim, mention the exact file path it came from.

Code excerpts:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, retrieved_docs