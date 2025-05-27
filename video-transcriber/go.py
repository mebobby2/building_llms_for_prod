import yt_dlp
from langchain_google_genai import ChatGoogleGenerativeAI
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap
from dotenv import load_dotenv
load_dotenv()


def download_mp4_from_youtube(url):
    filename = 'lecuninterview.mp4'
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      result = ydl.extract_info(url, download=True)

# url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
# download_mp4_from_youtube(url)

# model = whisper.load_model("base")
# result = model.transcribe("lecuninterview.mp4", language="English")
# print(result['text'])

# with open("transcription.txt", "w") as f:
#     f.write(result['text'])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-preview-05-06",
    temperature=0,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separators=[" ", ",", "\n"]
)

with open("transcription.txt", "r") as f:
    transcription_text = f.read()


texts = text_splitter.split_text(transcription_text)
docs = [Document(page_content=text) for text in texts[:4]]

chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.invoke(docs)

wrapped_text = textwrap.fill(output_summary['output_text'], width=100)
print(wrapped_text)
