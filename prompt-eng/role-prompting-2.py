from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
)
template = """
As a futuristic robot band conductor, I need you to help me come up with a song title
"""
prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

input_data = {"theme": "interstellar travel", "year": "3030"}

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)
