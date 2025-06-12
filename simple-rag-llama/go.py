from llama_index.vector_stores.qdrant import QdrantVectorStore
import os
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
import qdrant_client
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank

from dotenv import load_dotenv
load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash")
Settings.chunk_size = 512
Settings.chunk_overlap = 64


documents = SimpleDirectoryReader("./paul_graham").load_data()

client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="paul_graham")

if client.collection_exists(collection_name="paul_graham"):
  index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
else:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

## Simple Query Engine
# query_engine = index.as_query_engine(streaming=True, similartiy_top_k=10)

# streaming_response = query_engine.query("What does Paul Graham do?")
# streaming_response.print_response_stream()

## Sub Question Query Engine
# query_engine = index.as_query_engine(similarity_top_k=10)
# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=query_engine,
#         metadata=ToolMetadata(
#             name="paul_graham_query_engine",
#             description="Useful for answering questions about Paul Graham's essays.",
#         ),
#     )
# ]
# query_engine = SubQuestionQueryEngine.from_defaults(
#     query_engine_tools=query_engine_tools,
#     use_async=False,
# )
# response = query_engine.query("How was Paul Grahams life before, during, and after YC?")
# print(">>> The final response:\n", response)

# Cohere Rerank
cohere_rerank = CohereRerank(api_key=os.environ.get("COHERE_API_KEY"), top_n=2)
query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[cohere_rerank])

response = query_engine.query("Who was in charge of marketing at a Boston investment bank?")
print(response)
