import os
import shutil
import stat
import subprocess
import tempfile
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

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
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repo(repo_url, target_dir):
    # Sanitize URL to ensure it starts with standard HTTPS
    if not (repo_url.startswith("https://github.com/") or repo_url.startswith("http://github.com/")):
        raise ValueError("Invalid GitHub repository URL.")

    subprocess.run(
        ["git", "clone", "--depth", "1", "--", repo_url, target_dir],
        check=True
    )


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
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue

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
    collection_name = f"codebase_{uuid.uuid4().hex[:8]}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})


def index_repo(repo_url):
    """Runs the full ingestion pipeline for one repo URL and returns
    (retriever, source_files, num_chunks) — everything app.py needs."""
    repo_path = clone_repo(repo_url)
    source_files = collect_source_files(repo_path)
    chunks = build_code_documents(source_files)
    retriever = build_vectorstore(chunks)
    
    # Make sure you are returning 3 items here:
    return retriever, source_files, len(chunks)


def generate_answer(question, retriever):
    retrieved_docs = retriever.invoke(question)

    context = ""
    for doc in retrieved_docs:
        file_path = doc.metadata["file_path"]
        context += f"File: {file_path}\n```\n{doc.page_content}\n```\n\n"

    prompt = f"""You are a code assistant answering questions about a codebase.
Base your answer on the code excerpts below. You may reason about what the code
does and how it's architected, even if specific terms aren't literally present
in the code. For any specific fact or implementation detail, cite the exact file path.

Code excerpts:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, retrieved_docs