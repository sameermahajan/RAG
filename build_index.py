# build_index.py

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "./docs"             # put PDFs/TXT/MD here
VECTOR_DIR = "./vectorstore"    # persisted DB directory

# Load any local documents
loader = DirectoryLoader(DOCS_DIR, glob="**/*.*")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# Create embedding model
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build persistent Chroma DB
db = Chroma.from_documents(
    chunks,
    emb,
    persist_directory=VECTOR_DIR
)

db.persist()
print("✅ Vector store built and persisted.")