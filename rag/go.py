from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

text = """ Google opens up its AI language model PaLM to challenge
OpenAI and GPT-3 Google offers developers access to one of its most
advanced AI language models: PaLM. The search giant is launching an
API for PaLM alongside a number of AI enterprise tools it says will
help businesses "generate text, images, code, videos, audio, and
more from simple natural language prompts."
PaLM is a large language model, or LLM, similar to the GPT series
created by OpenAI or Meta's LLaMA family of models. Google first
announced PaLM in April 2022. Like other LLMs, PaLM is a flexible
system that can potentially carry out all sorts of text generation
and editing tasks. You could train PaLM to be a conversational
chatbot like ChatGPT, for example, or you could use it for tasks
like summarizing text or even writing code. (It's similar to
features Google also announced today for its Workspace apps like
Google Docs and Gmail.)
"""

with open("my_file.txt", "w") as f:
    f.write(text)

loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

# Use CharacterTextSplitter to split the documents into text snippets called 'chunk'.
# Chunk_overlap is the number of characters that overlap between two chunks. It preserves context
# and improves coherence by ensuring that important information is not cut off at the boundaries of chunks.
text_splitter = CharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, separator="")

docs = text_splitter.split_documents(docs_from_file)
print(docs[0])
print('====================')
print(docs[1])
print(len(docs))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)

uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)

retriever = vector_store.as_retriever()

# retriever.invoke just does a vector search
query = "How Google plans to challenge OpenAI?"
print(retriever.invoke(query)[0])
# Google opens up its AI language model PaLM to challenge
# OpenAI and GPT-3 Google offers developers access to one of its most
# advanced AI language models: PaLM. The search giant is launching an

query = "When was PaLM first announced?"
print(retriever.invoke(query)[0])
# PaLM is a large language model, or LLM, similar to the GPT series
# created by OpenAI or Meta's LLaMA family of models. Google first
# announced PaLM in April 2022. Like other LLMs, PaLM is a fl

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)
# this actually does a vector search and then passes the result to the LLM - so the results from the LLM are more relevant to the actual query
query = "How Google plans to challenge OpenAI?"
response = qa_chain.invoke(query)
print(response)
# Google plans to challenge OpenAI by offering developers access to its advanced AI language model, PaLM, through an API.
# This will allow businesses to generate various types of content, such as text, images, code, videos, and audio, from natural language prompts.

query = "When was PaLM first announced?"
response = qa_chain.invoke(query)
print(response)
# Google first announced PaLM in April 2022
