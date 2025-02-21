from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

examples = [
    {
        "query": "What's the secret to happiness?",
        "answer": """Finding balance in life and learning to enjoy the small
moments."""
    }, {
        "query": "How can I become more productive?",
        "answer": """Try prioritizing tasks, setting goals, and maintaining a
healthy work-life balance."""
    }
]

example_template = """
User: {query}
AI: {answer}
"""
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI
life coach. The assistant provides insightful and practical advice to the
users' questions. Here are some examples:
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)
user_query = "What are some tips for improving communication skills?"
response = chain.run({"query": user_query})
print("User Query:", user_query)
print("AI Response:", response)

# AI Response: Practice active listening, express yourself clearly and respectfully,
# and seek feedback to understand how others perceive you.


# Without examples
# prefix = """
# You are a life coach that provides insightful and practical advice to the
# users' questions.
# """
# template_question = "What are some tips for improving communication skills?"
# prompt_question = PromptTemplate(template=template_question, input_variables=[], prefix=prefix)
# chain = LLMChain(llm=llm, prompt=prompt_question)
# response = chain.run({})
# print("User Query:", user_query)
# print("AI Response:", response)
# Output, generates way too much information, not as succint as the few-shot prompt examples
