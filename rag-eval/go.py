from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import generate_question_context_pairs, RetrieverEvaluator

import pandas as pd
import asyncio

from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenAI(model="models/gemini-2.5-pro")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

reader = SimpleDirectoryReader(input_files=["venus_transmission.txt"])

docs = reader.load_data()
print(f"Loaded {len(docs)} documents.")

node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(docs)
vector_index = VectorStoreIndex(nodes)

# query_engine = vector_index.as_query_engine()

# response_vector = query_engine.query("""What was The first beings to inhabit the planet?""")
# print( response_vector.response )
# print("======================================")
# print(response_vector.source_nodes[0].get_text())

qa_dataset = generate_question_context_pairs(
    nodes, llm, num_questions_per_chunk=2)

queries = list(qa_dataset.queries.values())
print(queries[0:10])

retriever = vector_index.as_retriever(similarity_top_k=2)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever)

eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(qa_dataset))

print(f"Evaluated {len(eval_results)} results.")


def display_results(name, eval_results):
    """Display results from evaluate."""
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)
        full_df = pd.DataFrame(metric_dicts)
        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()
        metric_df = pd.DataFrame(
            {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
        )
    return metric_df


print(display_results("HuggingFace Embedding Retriever", eval_results))
#                     Retriever Name  Hit Rate       MRR
# 0  HuggingFace Embedding Retriever  0.692308  0.557692
