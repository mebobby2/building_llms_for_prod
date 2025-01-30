from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai


genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')


# prompt = """
# Question: Sort Tiger, Bear, Dog
# Answer: Bear > Tiger > Dog}
# Question: Sort Cat, Elephant, Zebra
# Answer: Elephant > Zebra > Cat}
# Question: Sort Whale, Goldfish, Monkey
# Answer:"""

movie = "Toy Story"
withoutFewShotPrompt = f"""
Question: Describe the movie 'Toy Story' using emojis
Answer:"""

withFewShotPrompt = f"""
Question: Describe the movie 'Titanic' using emojis
Answer:🛳️ 🌊 🤎
Question: Describe the movie 'The Lion King' using emojis
Answer:🦁 👑
Question: Describe the movie 'Toy Story' using emojis
Answer:"""

# Without few-shot prompting
# 🤠🚀🧸👧👦🐕‍🦺📦✨  ➡️  😭😂❤️友情

# With few-shot prompting
# 🧸🤠🚀
response = model.generate_content(withFewShotPrompt)
print(response.text)
