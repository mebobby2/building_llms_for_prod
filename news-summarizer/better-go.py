from newspaper import Article
import requests
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

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

template = """
As an advanced AI, you've been tasked to summarize online articles into
bulleted points. Here are a few examples of how you've done this in the
past:

Example 1:
Original Article: 'The Effects of Climate Change
Summary:
- Climate change is causing a rise in global temperatures.
- This leads to melting ice caps and rising sea levels.
- Resulting in more frequent and severe weather conditions.

Example 2:
Original Article: 'The Evolution of Artificial Intelligence
Summary:
- Artificial Intelligence (AI) has developed significantly over the past
decade.
- AI is now used in multiple fields such as healthcare, finance, and
transportation.
- The future of AI is promising but requires careful regulation.

Now, here's the article you need to summarize:
==================
Title: {article_title}

{article_text}
==================

Please provide a summarized version of the article in a bulleted list
format.
"""

prompt = template.format(article_title=article_title, article_text=article_text)

messsages = [HumanMessage(content=prompt)]

chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

summary = chat.invoke(messsages)
print("Summary:")
print(summary.content)
