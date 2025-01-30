# LLM Architectures and Landscape
The original architecture was designed for sequence-to-sequence tasks (where a sequence is inputted and an output is generated based on it), such as translation. In this process, the encoder creates a representation of the input phrase, and the decoder generates its output using this representation as a reference.

Further research into architecture resulted in its division into three unique
categories, distinguished by their versatility and specialized capabilities in
handling different tasks.
* The encoder-only category is dedicated to extracting context-aware
representations from input data. A representative model from this
category is BERT, which can be useful for classification tasks.
* The encoder-decoder category facilitates sequence-to-sequence
tasks such as translation, summarization and training multimodal
models like caption generators. An example of a model under this
classification is BART.
* The decoder-only category is specifically designed to produce
outputs by following the instructions provided, as demonstrated in
LLMs. A representative model in this category is the GPT family.

Next, we will cover the contrasts between these design choices and their
effects on different tasks. However, as you can see from the diagram,
several building blocks, like embedding layers and the attention
mechanism, are shared on both the encoder and decoder components.
Understanding these elements will help improve your understanding of how
the models operate internally. This section outlines the key components and
then demonstrates how to load an open-source model to trace each step.

## Input Embedding
As we’ve seen in the transformer architecture, the initial step is to turn input
tokens (words or subwords) into embeddings. These embeddings are high-
dimensional vectors that capture the semantic features of the input tokens.
You can see them as a large list of characteristics representing the words
being embedded. This list contains thousands of numbers that the model
learns by itself to represent our world. Instead of working with sentences,
words, and synonyms to compare things together, requiring an
understanding of our language, it works with these lists of numbers to
compare them numerically with basic calculations, subtracting and adding
those vectors together to see if they are similar or not. It looks much more
complex than understanding words themselves, doesn’t it? This is why the
size of these embedding vectors is pretty large. When you cannot
understand meanings and words, you need thousands of values representing
them. This size varies depending on the model’s architecture. GPT-3 by
OpenAI, for example, employs 12,000-dimensional embedding vectors, but
smaller models such as BERT employ 768-dimensional embeddings. This
layer enables the model to understand and process the inputs effectively,
serving as the foundation for all subsequent layers.

## Positional Encoding
Earlier models, such as Recurrent Neural Networks (RNNs), processed
inputs sequentially, one token at a time, naturally preserving the text’s
order. Unlike these, transformers do not have built-in sequential processing
capabilities. Instead, they employ positional encodings to maintain the order of words within a phrase for the next layers. These encodings are vectors
filled with unique values at each index, which, when combined with input
embeddings, provide the model with data regarding the tokens’ relative or
absolute positions within the sequence. These vectors encode each word’s
position, ensuring that the model identifies word order, which is essential
for interpreting the context and meaning of a sentence.

## Self-Attention Mechanism
The self-attention mechanism is at the heart of the transformer model,
calculating a weighted total of the embeddings of all words in a phrase.
These weights are calculated using learned “attention” scores between
words. Higher “attention” weights will be assigned to terms that are more
relevant to one another. Based on the inputs, this is implemented using
Query, Key, and Value vectors. Here is a brief description of each vector.
* Query Vector: This is the word or token for which the attention
weights are calculated. The Query vector specifies which sections of
the input sequence should be prioritized. When you multiply word
embeddings by the Query vector, you ask, “What should I pay
attention to?”
* Key Vector: The set of words or tokens in the input sequence
compared to the Query. The Key vector aids in identifying the
important or relevant information in the input sequence. When you
multiply word embeddings by the Key vector, you ask yourself,
“What is important to consider?”
* Value Vector: It stores the information or features associated with
each word or token in the input sequence. The Value vector contains
the actual data that will be weighted and mixed in accordance with
the attention weights calculated between the Query and Key. The
Value vector answers the query, “What information do we have?”
