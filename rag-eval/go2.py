from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import generate_question_context_pairs, RetrieverEvaluator
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator, BatchEvalRunner
import asyncio

from dotenv import load_dotenv
load_dotenv()


async def main():
    gem25pro = GoogleGenAI(model="models/gemini-2.5-pro")
    Settings.llm = gem25pro
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    reader = SimpleDirectoryReader(input_files=["venus_transmission.txt"])

    docs = reader.load_data()
    print(f"Loaded {len(docs)} documents.")

    node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
    nodes = node_parser.get_nodes_from_documents(docs)
    vector_index = VectorStoreIndex(nodes)

    qa_dataset = generate_question_context_pairs(
        nodes, gem25pro, num_questions_per_chunk=2)

    queries = list(qa_dataset.queries.values())
    print(queries[0:10])

    gem25flashlite = GoogleGenAI(model="models/gemini-2.5-flash-lite")
    vector_index = VectorStoreIndex(nodes, llm=gem25flashlite)
    query_engine = vector_index.as_query_engine()

    eval_query = queries[10]
    response_vector = query_engine.query(eval_query)

    print("> eval_query: ", eval_query)
    print("> response_vector:", response_vector)

    relevancy_gem25pro = RelevancyEvaluator(llm=gem25pro)
    faithfulness_gem25pro = FaithfulnessEvaluator(llm=gem25pro)

    def display_eval_df(
        query: str, response, eval_result
    ):
        print("Query:", query)
        print("Response:", str(response))
        print("Source:", response.source_nodes[0].node.text[:1000] + "...",)
        print("Evaluation Result:", "Pass" if eval_result.passing else "Fail")
        print("Reasoning:", eval_result.feedback)

    print("======================================")
    eval_result = faithfulness_gem25pro.evaluate_response(
        response=response_vector)
    display_eval_df(eval_query, response_vector, eval_result)

    print("======================================")
    eval_result = relevancy_gem25pro.evaluate_response(
        query=eval_query, response=response_vector)
    display_eval_df(eval_query, response_vector, eval_result)

    print("======================================BATCH")

    batch_eval_queries = queries[0:10]
    runner = BatchEvalRunner(
        {"faithfulness": faithfulness_gem25pro, "relevancy": relevancy_gem25pro}, workers=8)

    eval_results = await runner.aevaluate_queries(
        query_engine, queries=batch_eval_queries
    )

    faithfulness_score = sum(
        result.passing for result in eval_results["faithfulness"]) / len(eval_results["faithfulness"])

    relevancy_score = sum(
        result.passing for result in eval_results["relevancy"]) / len(eval_results["relevancy"])

    print("> faithfulness_score", faithfulness_score)
    print("> relevancy_score", relevancy_score)
    # > faithfulness_score 0.8
    # > relevancy_score 0.9

asyncio.run(main())
