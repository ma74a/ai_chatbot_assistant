from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from config import Config


embedding_model = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={'allow_download': 'True'}
)


# retriever = db.as_retriever(search_kwargs={"k":4})

def retrieve_chunks(query: str):
    db = Chroma(persist_directory=Config.CHROMA_PATH, embedding_function=embedding_model)
    docs = db.similarity_search_with_score(
    query,
    k=4)
    return docs