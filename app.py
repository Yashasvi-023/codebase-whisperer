import streamlit as st
import os
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from code_pipeline import index_repo, generate_answer

st.set_page_config(page_title="Codebase Whisperer", page_icon="💻", layout="wide")
st.title("💻 Codebase Whisperer")
st.write("Paste a public GitHub repo URL, then ask how the code works.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_repo" not in st.session_state:
    st.session_state.current_repo = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Indexed files")
    if st.session_state.indexed_files:
        for path in st.session_state.indexed_files:
            st.text(path)
    else:
        st.write("No repo indexed yet.")

repo_url = st.text_input("GitHub repo URL")

if repo_url and repo_url != st.session_state.current_repo:
    with st.spinner("Cloning repo and building index... this can take a minute"):
        try:
            retriever, source_files, num_chunks = index_repo(repo_url)
            st.session_state.retriever = retriever
            st.session_state.current_repo = repo_url
            st.session_state.indexed_files = sorted(set(path for path, _ in source_files))
            st.session_state.messages = []
            st.success(f"Indexed {len(source_files)} files, {num_chunks} chunks.")
        except Exception as e:
            st.error(f"Couldn't process this repo: {e}")
            st.session_state.retriever = None
            st.session_state.current_repo = None

if st.session_state.retriever is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask how the code works...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Reading the code..."):
                answer, retrieved_docs = generate_answer(question, st.session_state.retriever)
                st.markdown(answer)

                for doc in retrieved_docs:
                    file_path = doc.metadata["file_path"]
                    with st.expander(f"📄 {file_path}"):
                        ext = file_path.split(".")[-1]
                        st.code(doc.page_content, language=ext)

        st.session_state.messages.append({"role": "assistant", "content": answer})