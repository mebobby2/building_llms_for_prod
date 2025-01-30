from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai

# English text to translate
english_text = "Hello, how are you?"

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))

prompt = """
You are a professional translater "
"""
model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=prompt)

response = model.generate_content(f"Translate the following English text to French: {english_text}")
print(f"English: {english_text}, French: {response.text}")
