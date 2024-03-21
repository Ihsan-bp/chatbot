from langchain_community.chat_models.anyscale import ChatAnyscale
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
#loading data
loader = TextLoader("./paul_graham_essay.txt")
text = loader.load()
# splitting data
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50,
)
chunks = text_splitter.split_documents(text)
# Initialize embedding model
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)    
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# Initialize vector store
db = Chroma.from_documents(chunks, embedding_function, persist_directory="./chroma_db")