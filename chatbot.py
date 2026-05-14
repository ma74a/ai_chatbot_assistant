from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

from retriever import retrieve_chunks
from config import Config
from memory import ChatMemory

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    google_api_key=api_key
)


def ask(question: str, memory) -> str:
    docs = retrieve_chunks(question)

    context = "\n\n".join(
        [doc[0].page_content for doc in docs]
    )
    # print(context)
    history = memory.format_for_prompt()

    prompt = Config.SYSTEM_PROMPT.format(
        context=context,
        history=history,
        question=question
    )

    response = llm.invoke(prompt)

    memory.add(role="user", content=question)
    memory.add(role="assistant", content=response)

    return response.content


def show_sources(docs) -> None:
    """Print source attribution — shows which book/page was used."""
    print("\n  Sources:")
    seen = set()
    for doc, score in docs[:3]:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            # Lower score = more similar in Chroma's L2 distance
            print(f"    - {os.path.basename(src)}, page {page}")

if __name__ == "__main__":
    memory = ChatMemory()
    print("ML/DL Tutor (with memory). Commands: 'clear' to reset, 'q' to quit.\n")
 
    while True:
        query = input("User: ").strip()
 
        if not query:
            continue
        if query.lower() in ["q", "quit", "exit", "Q"]:
            break
        if query.lower() == "clear":
            memory.clear()
            print("Memory cleared.\n")
            continue
 
        answer = ask(question=query, memory=memory)
        print(f"\nBot: {answer}\n")
 
        # Optional: show which book pages were used
        docs = retrieve_chunks(query)
        show_sources(docs)
        print()