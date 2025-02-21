import os
from dotenv import load_dotenv
load_dotenv()


from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""

example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""Here are some examples of colors and the emotions associated
with them:\n\n""",
    suffix="""\n\nNow, given a new color, identify the emotion
associated with it:\n\nColor: {input}\nEmotion:""",
    input_variables=["input"],
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="purple")

chain = LLMChain(llm=llm, prompt=PromptTemplate(template=formatted_prompt,
                                                input_variables=[]))
# Run the LLMChain to get the AI-generated emotion associated with the
input
# color
response = chain.run({})
print("Color: purple")
print("Emotion:", response)
