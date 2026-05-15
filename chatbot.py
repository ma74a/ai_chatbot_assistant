from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

import os
from dotenv import load_dotenv
from typing import List, Tuple

from retriever import retrieve_chunks
from config import Config
from memory import ChatMemory

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=api_key
)

def ask(question: str, memory: ChatMemory) -> Tuple[str, Document]:
    search_query = question
    if not memory.is_empty():
        rephrase_prompt = f"Given the following conversation history and the user's new question, rephrase the new question into a standalone question that can be used to search for relevant information. If the new question is already standalone, just return it as is.\n\nHistory: {memory.get_history()}\n\nNew Question: {question}\n\nStandalone Question:"
        search_query = llm.invoke([HumanMessage(content=rephrase_prompt)]).content.strip()

    docs = retrieve_chunks(query=search_query)
    context = "\n\n".join(
        [doc[0].page_content for doc in docs]
    )

    message = [
        SystemMessage(content=Config.SYSTEM_PROMPT.format(context=context))
    ]

    message.extend(memory.get_history())
    message.append(HumanMessage(content=question))

    response = llm.invoke(message)
    answer = response.content

    memory.add_user_message(question)
    memory.add_ai_message(answer)

    return answer, docs


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
    print("ML/DL Tutor. Commands: 'clear' or 'c' to reset memory, 'q' or 'quit' or 'exit' to quit.\n")
    while True:
        query = input("User: ")
        if not query:
            continue
        if query.lower in ['q', "quit", "exit"]:
            print("BYE")
            break
        if query.lower in ['c', "clear"]:
            memory.clear()
            print("Memory cleared.\n")
            continue

        answer, docs = ask(question=query, memory=memory)
        print(f"Bot: {answer}")

        show_sources(docs)
        print()

