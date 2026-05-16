import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import retrieve_chunks
from memory import ChatMemory
from config import Config

load_dotenv()

# Page config 
st.set_page_config(
    page_title="ML/DL Tutor With Chat History",
    page_icon="⬡",
    layout="centered"
)

# Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp { background-color: #0d0f12; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Title */
.tutor-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.1em;
    color: #4f8ef7;
    margin-bottom: 4px;
}
.tutor-sub {
    font-size: 12px;
    color: #3d4755;
    margin-bottom: 24px;
}

/* Source tags */
.source-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.source-tag {
    background: #0f1318;
    border: 1px solid #252b35;
    border-radius: 5px;
    padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #64748b;
}

/* Chat input */
.stChatInputContainer { border-top: 1px solid #252b35 !important; background: #0d0f12 !important; }
</style>
""", unsafe_allow_html=True)

# Session state
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemory()

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, sources}

if "llm" not in st.session_state:
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

# Header 
st.markdown('<div class="tutor-title">⬡ ML / DL TUTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="tutor-sub">Answers grounded in your textbooks</div>', unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.rerun()

# Render chat history 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            tags = "".join(
                f'<span class="source-tag">📄 {s["file"]} · p.{s["page"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(f'<div class="source-row">{tags}</div>', unsafe_allow_html=True)

# Chat input 
if prompt := st.chat_input("Ask anything about ML & Deep Learning…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner(""):
            llm = st.session_state.llm
            memory = st.session_state.memory

            # Query rewriting 
            from langchain_core.messages import HumanMessage
            search_query = prompt
            if not memory.is_empty():
                rephrase_prompt = (
                    f"Given the following conversation history and the user's new question, "
                    f"rephrase the new question into a standalone question for searching. "
                    f"If already standalone, return it unchanged.\n\n"
                    f"History: {memory.get_history()}\n\n"
                    f"New Question: {prompt}\n\nStandalone Question:"
                )
                search_query = llm.invoke([HumanMessage(content=rephrase_prompt)]).content.strip()

            # Retrieval 
            docs = retrieve_chunks(search_query)
            context = "\n\n".join([doc[0].page_content for doc in docs])

            # Build messages and call LLM 
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=Config.SYSTEM_PROMPT.format(context=context))]
            messages.extend(memory.get_history())
            messages.append(HumanMessage(content=prompt))
            response = llm.invoke(messages)
            answer = response.content

            # Update memory 
            memory.add_user_message(prompt)
            memory.add_ai_message(answer)

            # Sources 
            sources = []
            seen = set()
            for doc, _ in docs[:3]:
                src = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                if (src, page) not in seen:
                    seen.add((src, page))
                    sources.append({"file": src, "page": page})

        # Render answer + sources
        st.markdown(answer)
        if sources:
            tags = "".join(
                f'<span class="source-tag">📄 {s["file"]} · p.{s["page"]}</span>'
                for s in sources
            )
            st.markdown(f'<div class="source-row">{tags}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})