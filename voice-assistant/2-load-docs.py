import os
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def load_docs(root_dir, filename):
  docs = []
  try:
    loader = TextLoader(os.path.join(root_dir, filename), encoding='utf-8')
    docs.extend(loader.load_and_split())
  except Exception as e:
    pass

  return docs

def split_docs(docs):
  text_splitter = CharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=0,
  )
  return text_splitter.split_documents(docs)

docs = load_docs("./", "huggingface_docs.txt")
texts = split_docs(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
client = QdrantClient()
client.create_collection(
    collection_name="huggingface_docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="huggingface_docs",
    embedding=embeddings,
)
uuids = [str(uuid4()) for _ in range(len(texts))]
vector_store.add_documents(documents=docs, ids=uuids)
