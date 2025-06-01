import os
import base64
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit_chat import message
from langchain_huggingface import HuggingFaceEmbeddings
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from audio_recorder_streamlit import audio_recorder
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
import streamlit as st

from dotenv import load_dotenv
load_dotenv()


def load_embeddings_and_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    client = QdrantClient()
    client.create_collection(
        collection_name="video_transcripts",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="video_transcripts",
        embedding=embeddings,
    )
    return vector_store


def transcribe_audio(audio_bytes):
    message = HumanMessage(
        content=[
            {"type": "text", "text": "transcribe the following audio:"},
            {
                "type": "media",
                "data": base64.b64encode(audio_bytes).decode('utf-8'),
                "mime_type": "audio/mp3",
            },
        ],
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)
    transcription = llm.invoke([message])
    return transcription.content


def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
    else:
        st.write("Error: No transcription available.")


def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")


def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Transcribe"):
            transcription = transcribe_audio(audio_bytes)
            display_transcription(transcription)

    return transcription


def search_db(user_input, db):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")
    client = QdrantClient()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="huggingface_docs",
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-preview-05-06", temperature=0)

    qa = RetrievalQA.from_llm(llm, retriever=retriever,
                              return_source_documents=True)

    return qa.invoke({"query": user_input})


def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))
        text = history["generated"][i]

        client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        audio = client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        st.audio(b"".join(list(audio)), format='audio/mp3')


def main():
    st.write("# JarvisBase  ")
    db = load_embeddings_and_database
    transcription = record_and_transcribe_audio()
    user_input = get_user_input(transcription)

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    if user_input:
        output = search_db(user_input, db)
        print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output['result'])
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        display_conversation(st.session_state)


if __name__ == "__main__":
    main()
