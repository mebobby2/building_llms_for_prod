import yt_dlp
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from uuid import uuid4
from langchain.prompts import PromptTemplate
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap
from dotenv import load_dotenv
load_dotenv()


def download_mp4_from_youtube(urls, job_id):
    video_info = []

    for i, url in enumerate(urls):
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': file_temp,
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")
            video_info.append((file_temp, title, author))

    return video_info


# urls = ["https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
#         "https://www.youtube.com/watch?v=cjs7QKJNVYM",]
# vides_details = download_mp4_from_youtube(urls, "video_transcriber_downloads")


# model = whisper.load_model("base")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")

client = QdrantClient()
# client.create_collection(
#     collection_name="video_transcripts",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
# )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="video_transcripts",
    embedding=embeddings,
)

# for video in vides_details:
#     result = model.transcribe(video[0], language="English")
#     print(f"Transcription for {video[0]}:\n{result['text']}\n")

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=0,
#         separators=[" ", ",", "\n"]
#     )
#     texts = text_splitter.split_text(result['text'])
#     docs = [Document(page_content=text) for text in texts[:4]]

#     uuids = [str(uuid4()) for _ in range(len(docs))]
#     vector_store.add_documents(documents=docs, ids=uuids)


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-preview-05-06",
    temperature=0,
)

prompt_template = """Use the following pieces of transcripts from a video
to answer the question in bullet points and summarized. If you don't know
the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Summarized answer in bullter points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {
    "prompt": PROMPT,
}
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
)

results = qa.invoke("Summarize the mentions of google according to their AI program")

wrapped_text = textwrap.fill(results['result'], width=100)
print(wrapped_text)

# The following code shows the prompt template used with the map_reduce
# chain type. The map-reduce process first summarizes each document
# separately using a language model (Map step), turning each into a new
# document. Then, it combines all of them into one document (Reduce step)
# to form the final summary.
# The "stuff" approach involves using all text from the transcribed video in a
# single prompt, which is a basic and straightforward method. However, it
# might not be the most efficient for handling large volumes of text.
# The 'refine' summarization chain is an approach designed to generate more
# precise and context-sensitive summaries. This method follows an iterative
# process to enhance the summary by incorporating additional context as
# needed. In practice, it initiates by summarizing the first text chunk.
# Subsequently, the evolving summary is enriched with new information from
# each subsequent chunk. It can produce more accurate and context-aware
# summaries than chains like 'stuff' and 'map_reduce'.
# chain = load_summarize_chain(llm, chain_type="map_reduce")

# output_summary = chain.invoke(docs)

# wrapped_text = textwrap.fill(output_summary['output_text'], width=100)
# print(wrapped_text)
# print (chain.llm_chain.prompt.template)
