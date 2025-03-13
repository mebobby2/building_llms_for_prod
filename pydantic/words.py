from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


class Suggestions(BaseModel):
    words: List[str] = Field(description="list of suggested words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

    @field_validator("words")
    def not_start_with_number(cls, field):
        for word in field:
            if word[0].isdigit():
                raise ValueError("Word should not start with a number")
        return field

    @field_validator("reasons")
    def end_with_dot(cls, field):
        for idx, item in enumerate(field):
          if item[-1] != ".":
            field[idx] += "."
        return field


parser = PydanticOutputParser(pydantic_object=Suggestions)

template = """
Offer a list of suggestions to substitue the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

target_word = "behaviour"
context = "The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."

# input_variables=variables are assigned later through the .format_prompt()
# partial_variables, variables defined immediately.
prompt_template = PromptTemplate(template=template, input_variables=[
                                 "target_word", "context"], partial_variables={"format_instructions": parser.get_format_instructions()})

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

chain = prompt_template | model | parser

output = chain.invoke({"target_word": target_word, "context": context})

print(output)
