from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
)

########################## Without Role Prompting ##########################

template = """
I need you to help me come up with a song title
"""
prompt = PromptTemplate(template=template)

chain = prompt | llm

response = chain.invoke({})
print("AI-generated song title:", response.content)


########################## Role Prompting ##########################

# template = """
# As a futuristic robot band conductor, I need you to help me come up with a song title
# """
# prompt = PromptTemplate(template=template)

# chain = prompt | llm

# response = chain.invoke({})
# print("AI-generated song title:", response.content)


########################## Zero Shot Prompting ##########################

# template = """
# Given the movie 'toy story', describe it using emojis:\nEmojis:
# """
# prompt = PromptTemplate(template=template)

# chain = prompt | llm

# response = chain.invoke({})
# print(response.content)


########################## Few Shot Prompting ##########################

# examples = [
#     {"movie": "titanic", "emojis": "🛳️ 🌊 🤎"},
#     {"movie": "the lion king", "emojis": "🦁 👑"},
# ]

# example_formatter_template = """
# Movie Name: {movie}
# Emoji Description: {emojis}\n
# """

# example_prompt = PromptTemplate(
#     input_variables=["movie", "emojis"],
#     template=example_formatter_template,
# )

# few_shot_prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     prefix="""Here are some examples of movie titles adn emojis that describe them:\n\n""",
#     suffix="""\n\nNow, given a new movie, describe it using emojis:\n\nMovie: {input}\nEmojis:""",
#     input_variables=["input"],
#     example_separator="\n",
# )

# input = "toy story"
# formatted_prompt = few_shot_prompt.format(input=input)

# prompt=PromptTemplate(template=formatted_prompt, input_variables=[])
# chain = prompt | llm

# response = chain.invoke({})

# print(f"Movie: {input}")
# print("Emoji Description:", response.content)

########################## Without Chain of Thought ##########################
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
# )

# template_question = """
# 5 people can create 5 donuts every 5 minutes. How much time would it take 25 people to make 100 donuts?
# Return the answer immediately.
# """
# prompt_question = PromptTemplate(template=template_question, input_variables=[])
# chain = prompt_question | llm

# response = chain.invoke({})
# print(response.content)

# Model gemini-1.5-pro gets it right without CoT. Correct answer is 20mins
# This is because this model is 'https://ai.google.dev/gemini-api/docs/models/gemini' https://ai.google.dev/gemini-api/docs/models/gemini

########################## Chain of Thought ##########################
# cot_template_question = """
# Question: 11 factories can make 22 cars per hour. How much time would it take 22 factories to make 88 cars?
# Answer: A factory can make 22/11=2 cars per hour. 22 factories can make 22*2=44 cars per hour. Making 88 cars would take 88/44=2 hours. The answer is 2 hours.
# Question: 5 people can create 5 donuts every 5 minutes. How much time would it take 25 people to make 100 donuts?
# Answer:"""
# prompt_question = PromptTemplate(template=cot_template_question, input_variables=[])

# chain = prompt_question | llm

# response = chain.invoke({})
# print(response.content)


########################## Prompt Chaining ##########################
# template_question = """What is the name of the famous scientest who
# developed the theory of general relativity?
# Answer: """
# prompt_question = PromptTemplate(template=template_question, input_variables=[])

# template_fact = """Provide a brief description of {scientist}'s
# theory of general relativity.
# Answer: """
# prompt_fact = PromptTemplate(template=template_fact, input_variables=["scientist"])

# chain_question = prompt_question | llm

# response_question = chain_question.invoke({})
# scientist = response_question.content.split(":")[1].strip()

# chain_fact = prompt_fact | llm

# input_data = {"scientist": scientist}

# response_fact = chain_fact.invoke(input_data)

# print("Scientist:", scientist)
# print("Theory of general relativity:", response_fact.content)
