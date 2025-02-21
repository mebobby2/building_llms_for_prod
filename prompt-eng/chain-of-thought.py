import os
from dotenv import load_dotenv
load_dotenv()


from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

# Sometimes LLMs can return non-satisfactory answers. To simulate that behavior, you can implement a phrase like "Return the answer
# immediately" in your prompt.
template_question = """
5 people can create 5 donuts every 5 minutes. How much time would it take 25 people to make 100 donuts?
Return the answer immediately.
"""
prompt_question = PromptTemplate(template=template_question, input_variables=[])
chain = LLMChain(llm=llm, prompt=prompt_question)
print(chain.run({}))


# 5mins // which is wrong
# model gemini-1.5-pro gets it right with CoT thought. Correct answer is 20mins


######################## Chain of Thought
cot_template_question = """
Question: 11 factories can make 22 cars per hour. How much time would it take 22 factories to make 88 cars?
Answer: A factory can make 22/11=2 cars per hour. 22 factories can make 22*2=44 cars per hour. Making 88 cars would take 88/44=2 hours. The answer is 2 hours.
Question: 5 people can create 5 donuts every 5 minutes. How much time would it take 25 people to make 100 donuts?
Answer:"""
prompt_question = PromptTemplate(template=cot_template_question, input_variables=[])
chain = LLMChain(llm=llm, prompt=prompt_question)
print(chain.run({}))

# It would take 20 minutes //correct

# Here's how to solve the donut problem:

# 1. Find the individual production rate:

# If 5 people make 5 donuts in 5 minutes, that means one person makes one donut in 5 minutes.
# 2. Calculate the total production rate:

# With 25 people, and each person making one donut every 5 minutes, they would make 25 donuts every 5 minutes.
# 3. Determine the time to make 100 donuts:

# Since they make 25 donuts every 5 minutes, it will take 4 sets of 5 minutes to make 100 donuts (100 donuts / 25 donuts per 5 minutes = 4 sets).
# Answer: It would take 25 people 20 minutes (4 sets x 5 minutes per set) to make 100 donuts.
