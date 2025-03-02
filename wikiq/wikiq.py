import os
import logging
import sys
from llama_index.core import download_loader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()

documents = loader.load_data(pages=["Natural Language Processing", "Artificial Intelligence"])
print(len(documents))

parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))

my_activeloop_org_id = "mebobby"
my_activeloop_dataset_name = "wikipedia-nlp-ai"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
