# Building LLMs for Production

## Setup
* python3 -m venv my_venv_name
* source my_venv_name/bin/activate
* deactivate
* python3 -m pip install -r requirements.txt
* python3 -m pip uninstall -y -r requirements.txt

## Set up Vector DB
Qdrant
```
docker pull qdrant/qdrant

docker run -p 6333:6333 -v /Users/BobbyLei/Desktop/learn/building_llms_for_prod/.qdrantdata:/qdrant/storage qdrant/qdrant
```

Navgiate to: `http://localhost:6333/dashboard#/welcome`

## Upto
Page 173


Establish a storage context using the

Before that - figure out why the activeloop data source isn't created. Activeloop usability sucks, switch to pgvector instead https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/

then, switch from the default llama-index openai embedding model to a open source hugging face one
https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/
