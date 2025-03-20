# Building LLMs for Production

## Setup
* python3 -m venv my_venv_name (`python3.10 -m venv my_venv_name` to create a virtual env with a another Python version)
* source my_venv_name/bin/activate
* deactivate
* pip3 install -r requirements.txt
* pip3 uninstall -y -r requirements.txt
* pip3 list

## Set up Vector DB
Qdrant
```
docker pull qdrant/qdrant

docker run -p 6333:6333 -v /Users/BobbyLei/Desktop/learn/building_llms_for_prod/.qdrantdata:/qdrant/storage qdrant/qdrant
```

Navgiate to: `http://localhost:6333/dashboard#/welcome`

## Upto
Page 225

In the previous code, we used the prompt to extract relation triplets
