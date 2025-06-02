# Google Speech-to-Text Audio Transcripts
The SpeechToTextLoader allows to transcribe audio files with the Google Cloud Speech-to-Text API and loads the transcribed text into documents.

To use it, you should have the google-cloud-speech python package installed, and a Google Cloud project with the Speech-to-Text API enabled.

```
rom langchain_google_community import SpeechToTextLoader

project_id = "<PROJECT_ID>"
file_path = "gs://cloud-samples-data/speech/audio.flac"
# or a local file path: file_path = "./audio.wav"

loader = SpeechToTextLoader(project_id=project_id, file_path=file_path)

docs = loader.load()
```

## Source
https://python.langchain.com/docs/integrations/document_loaders/google_speech_to_text/


# Different Speech to Text models offered by Google
The earlier models like “Chirp 2” were very specific to speech, they can convert speech to text. So, if you want some response from LLM using speech, first you need to hit speech-to-text model then send the text to some text processing LLM model to get the response.

Now, Gemini has multimodal capabilities.

In Gemini-1.5 models, you can insert multimodal input (like text, image, audio, video) and get the response as text only.

Gemini-2.0 series are even more powerful, you can insert multimodal input (like text, image, audio, video) and get multimodal response as well. But, speech and image output is even in private/public preview, so text as output only.

Coming to clicking on green “Try Gemini 2.0 Flash, our newest model with low latency and enhanced performance” : It is taking you to Vertex AI Studio where you can try new Gemini 2.0 Flash model. So, either you can record your voice or upload any audio file, add it to prompt and just write “Transcribe” it will convert voice/audio to text as it is.

Similarly, there is Google AI Studio where you can try latest Gemini/Gemma models.

You are correct : write prompt “please transcribe the following audio” or just “Transcribe” that will convert speech to text.

## Source
https://discuss.ai.google.dev/t/different-speech-to-text-models-offered-by-google/69511/3

# How to use multimodal prompts
Here we demonstrate how to use prompt templates to format multimodal inputs to models.

To use prompt templates in the context of multimodal data, we can templatize elements of the corresponding content block. For example, below we define a prompt that takes a URL for an image as a parameter:
```
from langchain_core.prompts import ChatPromptTemplate

# Define prompt
prompt = ChatPromptTemplate(
    [
        {
            "role": "system",
            "content": "Describe the image provided.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "{image_url}",
                },
            ],
        },
    ]
)
```

Let's use this prompt to pass an image to a chat model:
```
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

chain = prompt | llm
response = chain.invoke({"image_url": url})
print(response.text())
```

Note that we can templatize arbitrary elements of the content block:
```
prompt = ChatPromptTemplate(
    [
        {
            "role": "system",
            "content": "Describe the image provided.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "{image_mime_type}",
                    "data": "{image_data}",
                    "cache_control": {"type": "{cache_type}"},
                },
            ],
        },
    ]
)
```

```
import base64

import httpx

image_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

chain = prompt | llm
response = chain.invoke(
    {
        "image_data": image_data,
        "image_mime_type": "image/jpeg",
        "cache_type": "ephemeral",
    }
)
print(response.text())
```



## Source
https://python.langchain.com/docs/how_to/multimodal_prompts/

# Multimodality
Multimodality refers to the ability to work with data that comes in different forms, such as text, audio, images, and video. Multimodality can appear in various components, allowing models and systems to handle and process a mix of these data types seamlessly.

* Chat Models: These could, in theory, accept and generate multimodal inputs and outputs, handling a variety of data types like text, images, audio, and video.
* Embedding Models: Embedding Models can represent multimodal content, embedding various forms of data—such as text, images, and audio—into vector spaces.
* Vector Stores: Vector stores could search over embeddings that represent multimodal data, enabling retrieval across different types of information.

## Multimodality in chat models

Some models can accept multimodal inputs, such as images, audio, video, or files. The types of multimodal inputs supported depend on the model provider. For instance, OpenAI, Anthropic, and Google Gemini support documents like PDFs as inputs.

The gist of passing multimodal inputs to a chat model is to use content blocks that specify a type and corresponding data. For example, to pass an image to a chat model as URL:
```
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        {
            "type": "image",
            "source_type": "url",
            "url": "https://...",
        },
    ],
)
response = model.invoke([message])
```

We can also pass the image as in-line data:
```
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        {
            "type": "image",
            "source_type": "base64",
            "data": "<base64 string>",
            "mime_type": "image/jpeg",
        },
    ],
)
response = model.invoke([message])
```


## Source
https://python.langchain.com/docs/concepts/multimodality/
