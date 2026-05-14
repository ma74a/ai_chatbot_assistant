from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

import os
import shutil
from typing import List

from config import Config

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
embedding_model = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={'allow_download': 'True'}
)
def load_documents() -> List[Document]:
    "Load PDFs documents"
    loader = DirectoryLoader(
        path=Config.DATA_PATH,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        use_multithreading=True
    )
    documents = loader.load()
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    "Split documents into small chunks"
    split_docs = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = split_docs.split_documents(documents=documents)
    # for i, chunk in enumerate(chunks[:5]):
    #     print(type(chunk.page_content), repr(chunk.page_content[:100]))

    return chunks

def save_to_chroma(chunks: List[Document]):
    "save to vectordb"

    # Remove the old and save to new
    if os.path.exists(Config.CHROMA_PATH):
        shutil.rmtree(Config.CHROMA_PATH)

    db = Chroma.from_documents(documents=chunks, embedding=embedding_model,
                               persist_directory=Config.CHROMA_PATH)
    
    print(f"Saved {len(chunks)} chunks to {Config.CHROMA_PATH}.")

def create_vectordb() -> None:
    documents = load_documents()
    chunks = split_documents(documents=documents)
    save_to_chroma(chunks=chunks)


if __name__ == "__main__":
    # create_vectordb()
    pass