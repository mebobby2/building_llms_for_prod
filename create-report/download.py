import os
import requests
from qdrant_client.http.models import Distance, VectorParams
from newspaper import Article
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
}

article_urls = [
    """https://www.artificialintelligence-news.com/2023/05/23/meta-open-
source-speech-ai-models-support-over-1100-languages/""",
    """https://www.artificialintelligence-news.com/2023/05/18/beijing-
launches-campaign-against-ai-generated-misinformation/"""
    """https://www.artificialintelligence-news.com/2023/05/16/openai-ceo-ai-
regulation-is-essential/""",
    """https://www.artificialintelligence-news.com/2023/05/15/jay-migliaccio-
ibm-watson-on-leveraging-ai-to-improve-productivity/""",
    """https://www.artificialintelligence-news.com/2023/05/15/iurii-
milovanov-softserve-how-ai-ml-is-helping-boost-innovation-and-
personalisation/""",
    """https://www.artificialintelligence-news.com/2023/05/11/ai-and-big-
data-expo-north-america-begins-in-less-than-one-week/""",
    """https://www.artificialintelligence-news.com/2023/05/11/eu-committees-
green-light-ai-act/""",
    """https://www.artificialintelligence-news.com/2023/05/09/wozniak-warns-
ai-will-power-next-gen-scams/""",
    """https://www.artificialintelligence-news.com/2023/05/09/infocepts-ceo-
shashank-garg-on-the-da-market-shifts-and-impact-of-ai-on-data-
analytics/""",
    """https://www.artificialintelligence-news.com/2023/05/02/ai-godfather-
warns-dangers-and-quits-google/""",
    """https://www.artificialintelligence-news.com/2023/04/28/palantir-demos-
how-ai-can-used-military/""",
    """https://www.artificialintelligence-news.com/2023/04/26/ftc-chairwoman-
no-ai-exemption-to-existing-laws/""",
    """https://www.artificialintelligence-news.com/2023/04/24/bill-gates-ai-
teaching-kids-literacy-within-18-months/""",
    """https://www.artificialintelligence-news.com/2023/04/21/google-creates-
new-ai-division-to-challenge-openai/"""
]
session = requests.Session()
pages_content = []  # where we save the scraped articles

for url in article_urls:
    try:
        time.sleep(2)  # be kind, don't spam the server
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            print(
                f"Successfully retrieved {url}: Status code {response.status_code}")
            article = Article(url)
            article.download()
            article.parse()
            pages_content.append({"url": url, "text": article.text})
        else:
            print(
                f"Failed to retrieve {url}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error processing {url}: {e}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

all_texts = []
for d in pages_content:
    chunks = text_splitter.split_text(d["text"])
    for chunk in chunks:
        all_texts.append(chunk)

client = QdrantClient(check_compatibility=False)
# Uncomment the following line if running for the first time to create the collection
# client.create_collection(
#     collection_name="research_articles",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="research_articles",
    embedding=embeddings,
)

vector_store.add_texts(texts=all_texts)
