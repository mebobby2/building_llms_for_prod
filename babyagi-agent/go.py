import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.autonomous_agents import BabyAGI
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_size = 1536
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

goal = "Plan a trip to the grand canyon with a budget of $1000. Include transportation, accommodation, and activities."

llm = chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
)

baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=True, max_iterations=1
)

response = baby_agi({"objective": goal})
