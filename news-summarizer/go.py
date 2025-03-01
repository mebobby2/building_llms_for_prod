from newspaper import Article
import requests
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
}

article_url = "https://www.artificialintelligence-news.com/news/you-com-ari-professional-grade-ai-research-agent-for-businesses/"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        print(f"Title: {article.title}")
        print(f"Text: {article.text}")

    else:
        print(f"Failed to fetch article: {article_url}")
except Exception as e:
    print(
        f"Error occurred while fetching article at: {article_url}. Error: {e}")

article_title = article.title
article_text = article.text

template = """You are a very good assistant that summarizes online articles.

Here's the article you want to summarize:.

=====================
Title: {article_title}

{article_text}
=====================

Now, provide a summarized version of the article in bulleted list format, in Chinese.
"""

prompt = template.format(article_title=article_title, article_text=article_text)

messages = [HumanMessage(content=prompt)]

chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

summary = chat.invoke(messages)
print("Summary:")
print(summary.content)
