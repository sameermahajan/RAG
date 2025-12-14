# query_index.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DIR = "./vectorstore"

# Load embedding model (must match original one)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load persisted vector DB
db = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=emb
)

retriever = db.as_retriever(search_kwargs={"k": 5})

# Query it
query = "What does the contract say about termination clauses?"
docs = retriever.invoke(query)   # ← Correct method

for i, d in enumerate(docs, 1):
    print(f"\n=== RESULT {i} ===\n")
    print(d.page_content)
