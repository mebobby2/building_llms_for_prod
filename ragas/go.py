from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas.testset import TestsetGenerator
from ragas.metrics import (
    Faithfulness,
)
from ragas.metrics import AspectCritic
from ragas.integrations.llama_index import evaluate
from ragas.llms import LlamaIndexLLMWrapper
from llama_index.core import SimpleDirectoryReader


from dotenv import load_dotenv
load_dotenv()

# Define settings for LlamaIndex
Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.chunk_size = 512
Settings.chunk_overlap = 64

# Load the dataset
documents = SimpleDirectoryReader("./nyc_wikipedia").load_data()

# Building the vector index
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=StorageContext.from_defaults()
)
query_engine = vector_index.as_query_engine()

# Test the query engine
response_vector = query_engine.query("How did New York City get its name?")
print(response_vector.response)

generator = TestsetGenerator.from_llama_index(
    llm=Settings.llm,
    embedding_model=Settings.embed_model,
)
testset = generator.generate_with_llamaindex_docs(
    documents,
    testset_size=10,
)

# Build the evaluation dataset
df = testset.to_pandas()
# print(df.to_string())

print(f"Question: {df["user_input"][0]}")
response_vector = query_engine.query(df["user_input"][0])
print(f"Answer: {response_vector.response}")


# Evaluate the model using Ragas metrics
evaluator_llm = LlamaIndexLLMWrapper(GoogleGenAI(model="models/gemini-1.5-flash"))
metrics = [
    Faithfulness(llm=evaluator_llm),
    AspectCritic(
        name="harmlessness",
        definition="Is the submission harmless?",
        llm=evaluator_llm,
    ),
    AspectCritic(
        name="maliciousness",
        definition="Is the submission malicious?",
        llm=evaluator_llm,
    ),
    AspectCritic(
        name="coherence",
        definition="Is the submission coherent?",
        llm=evaluator_llm,
    ),
    AspectCritic(
        name="conciseness",
        definition="Is the submission concise?",
        llm=evaluator_llm,
    ),
    AspectCritic(
        name="correctness",
        definition="Is the submission correct?",
        llm=evaluator_llm,
    ),
]
result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=testset.to_evaluation_dataset(),
)

print(result)
