import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

path = "pdf_db"
client = chromadb.PersistentClient(path)
embedding_function = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

file_path = "chromadb/nature_deep_learning.pdf"

loader = PyPDFLoader(file_path)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(document)

db = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embedding_function
)

if __name__ == "__main__":
    query = "What is Deep Learning?"
    docs = db.similarity_search(query)

    for doc in docs:
        print(doc.page_content)
        print()