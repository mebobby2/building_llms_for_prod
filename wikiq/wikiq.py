import logging
import sys
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
import qdrant_client
from llama_index.core import download_loader
#from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()

documents = loader.load_data(pages=["Natural Language Processing", "Artificial Intelligence"])
print(len(documents))

parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))

client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="wikiq")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
