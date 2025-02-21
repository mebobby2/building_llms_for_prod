import os
from dotenv import load_dotenv
load_dotenv()


from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

template_question = """What is the name of the famous scientest who
developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

template_fact = """Provide a brief description of {scientist}'s
theory of general relativity.
Answer: """
prompt_fact = PromptTemplate(template=template_fact, input_variables=["scientist"])

chain_question = LLMChain(llm=llm, prompt=prompt_question)

response_question = chain_question.run({})
scientist = response_question.split(":")[1].strip()

chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

input_data = {"scientist": scientist}

response_fact = chain_fact.run(input_data)

print("Scientist:", scientist)
print("Theory of general relativity:", response_fact)
