import buildtask
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

vector_store = FAISS(embedding_function=embeddings)

from langchain_community.document_loaders.pdf import BasePDFLoader

loader = BasePDFLoader("tmp.pdf")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")