# Building LLMs for Production

## Setup
* python -m venv my_venv_name
* source my_venv_name/bin/activate
* python3 -m pip install -r requirements.txt
* deactivate

## Upto
Page 64

First, use AutoModelForCausalLM and AutoTokenizer

But first, why can't I run the code:
```
Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes: `pip install -i https://pypi.org/simple/ bitsandbytes`
```
