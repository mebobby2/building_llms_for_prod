import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Zero shot prompting

prompt_system = """
You are a helpful assistant whose goal is to write short poems.
"""
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash', system_instruction=prompt_system)

prompt = "Write a short poem about {topic}."

response = model.generate_content(prompt.format(topic="summer"))
print(response.text)

# Sun-drenched days and skies so bright,
# Warm breeze whispers, pure delight.
# Long, slow evenings, fireflies gleam,
# Summer's magic, a vibrant dream.

# Few shot prompting

prompt_system = """
You are a helpful assistant whose goal is to write short poems.
"""
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash', system_instruction=prompt_system)

prompt = """
Write a short poem about nature.
Poem:
Birdsong fills the air
Mountains high and valleys deep
Nature's music sweet.

Write a short poem about winter.
Poem:
Snow blankets the ground
Silence is the only sound
Winter's beauty found.

Write a short poem about summer.
Poem:
"""


response = model.generate_content(prompt)
print(response.text)

# Golden sun shines bright
# Warm breeze whispers through the trees
# Summer's gentle light.
