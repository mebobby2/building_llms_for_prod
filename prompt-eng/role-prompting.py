from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))

#################################################################

prompt_system = """
You are a helpful assistant whose goal is to help write stories.
"""
model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=prompt_system)

prompt = """Continue the following story. Write no more than 50 words.
Once upon a time, in a world where animals could speak, a courageous mouse
named Benjamin decided to"""

response = model.generate_content(prompt)
print(response.text)

################################################################
prompt_system = """
You are a helpful assistant whose goal is to help write product descriptions.
"""
model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=prompt_system)

prompt = """Write a captivating product description for a luxurious,
handcrafted, limited-edition fountain pen made from rosewood and gold.
Write no more than 50 words."""

response = model.generate_content(prompt)
print(response.text)
