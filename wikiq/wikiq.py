import logging
import sys
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
import qdrant_client
from llama_index.core import download_loader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

from dotenv import load_dotenv
load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = Gemini(model="models/gemini-1.5-flash")


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

query_engine = index.as_query_engine()

question1 = "What does NLP stand for?"
print(f"Question: {question1}")
response = query_engine.query(question1)
print("Answer:", response.response)

question2 = "What does the Chinese room experiment have to do with NLP?"
print(f"Question: {question2}")
response = query_engine.query(question2)
print("Answer:", response.response)

question3 = "In what year did the Goldman Sachs Research Paper entitled 'AI Data Centers and the Coming US Power Demand Surge' come out?"
print(f"Question: {question3}")
response = query_engine.query(question3)
print("Answer:", response.response) # Answer = 2024

# When I asked the question "In what year did the Goldman Sachs Research Paper entitled 'AI Data Centers and the Coming US Power Demand Surge' come out?" to gemini-1.5-flash, it returned the following answer:
# There is no publicly available Goldman Sachs research paper with the exact title "AI Data Centers and the Coming US Power Demand Surge."  While Goldman Sachs has published extensively on the energy demands of data centers and the impact of AI, a paper with that precise title doesn't appear in their readily accessible research archives.  To find a specific paper, you would need a more precise title or a link to the publication.

# This highlights the power of RAG-based models in providing contextually relevant answers to questions, even when the information is not directly available in the training data.
