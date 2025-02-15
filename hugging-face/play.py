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
#  print(OPT.model)

# Embedding layer
embedded_input = OPT.model.decoder.embed_tokens(inp_tokenized['input_ids'])
# print("Layer: \t", OPT.model.decoder.embed_tokens)
# print("Size: \t", embedded_input.size())
# print("Output: \t", embedded_input)


# Postional layer
embed_pos_input = OPT.model.decoder.embed_positions(inp_tokenized['attention_mask'])
# print("Layer: \t", OPT.model.decoder.embed_positions)
# print("Size: \t", embed_pos_input.size())
# print("Output: \t", embed_pos_input)

# Self attention layer
embed_position_input = embedded_input + embed_pos_input
hidden_states, _, _ = OPT.model.decoder.layers[0].self_attn(embed_position_input)
print("Layer: \t", OPT.model.decoder.layers[0].self_attn)
print("Size: \t", hidden_states.size())
print("Output: \t", hidden_states)
