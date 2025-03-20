from newspaper import Article
import requests
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain.prompts import PromptTemplate
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
You are a very good assistant that summarizes online articles.
Here's the article you want to summarize.
==================
Title: {article_title}

{article_text}
==================

{format_instructions}
"""

class ArticleSumary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(
        description="""Bulleted list summary of the article""")

    # validating whether the generated summary has at least three lines
    @field_validator('summary')
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError(
                """Generated summary has less than three bullet points!""")
        return list_of_lines


parser = PydanticOutputParser(pydantic_object=ArticleSumary)

prompt_template = PromptTemplate(template=template, input_variables=[
                                 "article_title", "article_text"], partial_variables={"format_instructions": parser.get_format_instructions()})


chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

chain = prompt_template | chat | parser

summary = chain.invoke({"article_title": article_title, "article_text": article_text})

print(summary)
