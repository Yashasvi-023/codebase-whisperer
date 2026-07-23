import streamlit as st
from supabase import create_client

supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("💻 Codebase Whisperer")
    st.write("Sign in to continue.")

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in"):
            try:
                result = supabase.auth.sign_in_with_password({
                    "email": login_email,
                    "password": login_password
                })
                st.session_state.user = result.user
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab_signup:
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign up"):
            try:
                result = supabase.auth.sign_up({
                    "email": signup_email,
                    "password": signup_password
                })
                st.success("Account created! Please log in.")
            except Exception as e:
                st.error(f"Sign up failed: {e}")

    st.stop()


import os
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from code_pipeline import index_repo, generate_answer

st.set_page_config(page_title="Codebase Whisperer", page_icon="💻", layout="wide")
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.5rem;
    }
    .stExpander {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }
</style>
""", unsafe_allow_html=True)
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
    st.write(f"Logged in as **{st.session_state.user.email}**")
    if st.button("Log out"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.session_state.retriever = None
        st.session_state.current_repo = None
        st.rerun()

    st.divider()

    st.header("Recent repos")
    history = supabase.table("repo_history") \
        .select("repo_url") \
        .eq("user_id", st.session_state.user.id) \
        .order("created_at", desc=True) \
        .limit(10) \
        .execute()

    for row in history.data:
        if st.button(row["repo_url"], key=f"history_{row['repo_url']}"):
            st.session_state.prefill_url = row["repo_url"]
            st.rerun()

    st.divider()

    st.header("Indexed files")
    if st.session_state.indexed_files:
        for path in st.session_state.indexed_files:
            st.text(path)
    else:
        st.write("No repo indexed yet.")

if "prefill_url" not in st.session_state:
    st.session_state.prefill_url = ""

repo_url = st.text_input("GitHub repo URL", value=st.session_state.prefill_url)

if repo_url and repo_url != st.session_state.current_repo:
    with st.spinner("Cloning repo and building index... this can take a minute"):
        try:
            retriever, source_files, num_chunks = index_repo(repo_url)
            st.session_state.retriever = retriever
            st.session_state.current_repo = repo_url
            st.session_state.indexed_files = sorted(set(path for path, _ in source_files))
            st.session_state.messages = []

            supabase.table("repo_history").insert({
                "user_id": st.session_state.user.id,
                "repo_url": repo_url
            }).execute()

            st.success(f"Indexed {len(source_files)} files, {num_chunks} chunks.")
            st.rerun()
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