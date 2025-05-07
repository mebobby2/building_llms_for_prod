from langchain_community.document_loaders import SeleniumURLLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
load_dotenv()

###### Data Preparation ######

# urls = ['https://beebom.com/what-is-nft-explained/',
#         'https://beebom.com/how-delete-spotify-account/',
#         'https://beebom.com/how-download-gif-twitter/',
#         'https://beebom.com/how-use-chatgpt-linux-terminal/',
#         'https://beebom.com/how-delete-spotify-account/',
#         'https://beebom.com/how-save-instagram-story-with-music/',
#         'https://beebom.com/how-install-pip-windows/',
#         'https://beebom.com/how-check-disk-usage-linux/']

# loader = SeleniumURLLoader(urls=urls)
# docs_not_splitted = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="")
# docs = text_splitter.split_documents(docs_not_splitted)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# client = QdrantClient()

# client.create_collection(
#     collection_name="support_chatbot",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )

# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="support_chatbot",
#     embedding=embeddings,
# )

# uuids = [str(uuid4()) for _ in range(len(docs))]
# vector_store.add_documents(documents=docs, ids=uuids)


###### Checking ######
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")
client = QdrantClient()
vector_store = QdrantVectorStore(
    client=client,
    collection_name="support_chatbot",
    embedding=embeddings,
)

# query = "how to check disk usage in linux?"
# docs = vector_store.similarity_search(query)
# print(docs[0].page_content)

# query = "how dangerous is it to delete your spotify account?"
# docs = vector_store.similarity_search(query)
# print(docs[0].page_content)

######## LLM Chain ########

template = """You are an exceptional customer support chatbot that gently
answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information
from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["chunks_formatted", "query"],
)

query = "Is the Linux distribution free?"

docs = vector_store.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

chunks_formatted = "\n\n".join(retrieved_chunks)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

chain = prompt | llm

answer = chain.invoke({'chunks_formatted': chunks_formatted, 'query': query})
print(answer.content)
