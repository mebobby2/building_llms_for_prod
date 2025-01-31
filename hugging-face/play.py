from transformers import AutoModelForCausalLM, AutoTokenizer

# https://stackoverflow.com/questions/76924239/accelerate-and-bitsandbytes-is-needed-to-install-but-i-did
# load_in_4bit=True or load_in_8bit=True can't be used on CPU. We now return a better error message stating that no GPU was found.
# OPT = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", load_in_8bit=True)
OPT = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

inp = "The quick brown fox jumps over the lazy dog"
inp_tokenized = tokenizer(inp, return_tensors="pt")
print(inp_tokenized['input_ids'].size())
print(inp_tokenized)
