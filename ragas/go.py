from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics.critique import harmfulness
from ragas.llama_index import evaluate

from dotenv import load_dotenv
load_dotenv()

Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.chunk_size = 512
Settings.chunk_overlap = 64

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://en.wikipedia.org/wiki/New_York_City"]
)

vector_index = VectorStoreIndex.from_documents(
    documents, service_context=StorageContext.from_defaults()
)

query_engine = vector_index.as_query_engine()

response_vector = query_engine.query("How did New York City get its name?")

print(response_vector.response)

eval_questions = [
    "What is the population of New York City as of 2020?",
    "Which borough of New York City has the highest population?",
    "What is the economic significance of New York City?",
    "How did New York City get its name?",
    "What is the significance of the Statue of Liberty in New York City?",
]
eval_answers = [
    "8,804,000",  # incorrect answer
    "Queens",  # incorrect answer
    """New York City's economic significance is vast, as it serves as the
global financial capital, housing Wall Street and major financial
institutions. Its diverse economy spans technology, media, healthcare,
education, and more, making it resilient to economic fluctuations. NYC is
a hub for international business, attracting global companies, and boasts
a large, skilled labor force. Its real estate market, tourism, cultural
industries, and educational institutions further fuel its economic
prowess. The city's transportation network and global influence amplify
its impact on the world stage, solidifying its status as a vital economic
player and cultural epicenter.""",
    """New York City got its name when it came under British control in 1664.
King Charles II of England granted the lands to his brother, the Duke of
York, who named the city New York in his own honor.""",
    """The Statue of Liberty in New York City holds great significance as a
symbol of the United States and its ideals of liberty and peace. It
greeted millions of immigrants who arrived in the U.S. by ship in the late
19th and early 20th centuries, representing hope and freedom for those
seeking a better life. It has since become an iconic landmark and a global
symbol of cultural diversity and freedom.""",
]
eval_answers = [[a] for a in eval_answers]
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    harmfulness,
]
result = evaluate(query_engine, metrics, eval_questions, eval_answers)
