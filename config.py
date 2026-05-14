# from langchain_huggingface import HuggingFaceEmbeddings

class Config:
    DATA_PATH="data"
    CHROMA_PATH="Chroma"

    MEMORY_WINDOW = 6

    # EMBEDDING_MODEL = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )

    SYSTEM_PROMPT = """
    You are an expert AI assistant specialized in Machine Learning and Deep Learning.

    Answer the user's question using ONLY the provided context.

    If the answer is not found in the context,
    say:
    "I could not find the answer in the provided documents."

    Be clear, educational, and concise.

    Context:
    {context}

    Conversation history:
    {history}

    Question:
    {question}

    Answer:
    """