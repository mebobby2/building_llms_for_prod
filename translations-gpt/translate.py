from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai

# English text to translate
english_text = "Hello, how are you?"

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-1.5-flash')

response = model.generate_content(f"You are a helpful assistant. Translate the following English text to French: {english_text}")
print(f"English: {english_text}, French: {response.text}")
