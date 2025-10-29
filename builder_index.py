from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === STEP 1: Load your pharmacology book ===
loader = PyPDFLoader("theBook.pdf")
documents = loader.load()

# === STEP 2: Split the book into smaller chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# === STEP 3: Create embeddings using Hugging Face (free & local) ===
print("🔍 Generating embeddings with HuggingFace...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === STEP 4: Build a FAISS vector index ===
print("⚙️ Building FAISS vector index...")
vectorstore = FAISS.from_documents(docs, embeddings)

# === STEP 5: Save the FAISS index locally ===
vectorstore.save_local("theBook_faiss_index")

print("✅ Index successfully built and saved as 'theBook_faiss_index'")
