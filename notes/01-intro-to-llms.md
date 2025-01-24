# Introduction to LLMs
The primary goal of LLMs is to interpret and create human-like text that captures the nuances ofnatural language, including syntax (the arrangement of words) and semantics (the meaning of words).

## Key LLM Terminologies
### The Transformer
The foundation of a language model that makes it powerful lies in its architecture. Recurrent Neural Networks (RNNs) were traditionally used for text processing due to their ability to process sequential data. They maintain an internal state that retains information from previous words, facilitating sequential understanding. However, RNNs encounter challenges with long sequences where they forget older information in favor of recently processed input. This is primarily caused by the vanishing gradient problem, a phenomenon where the gradients, which are used to update the network’s weights during training, become increasingly smaller as they are propagated back through each timestep of the sequence. As a result, the weights associated with early inputs change very little, hindering the network’s ability to learn from and remember long-term dependencies within the data.

Transformer-based models addressed these challenges and emerged as the preferred architecture for natural language processing tasks. This architecture introduced in the influential paper “Attention Is All You Need” is a pivotal innovation in natural language processing. It forms the foundation for cutting-edge models like GPT-4, Claude, and LLaMA. The architecture was originally designed as an encoder-decoder framework. This setting uses an encoder to process input text, identifying important parts and creating a representation of the input. Meanwhile, the decoder is capable of transforming the encoder’s output, a vector of high dimensionality, back into readable text for humans. These networks can be useful in tasks such as summarization, where the decoder generates summaries conditioned based on the articles passed to the encoder. It offers additional flexibility across a wide range of tasks since the components of this architecture, the encoder, and decoder, can be used jointly or independently. Some models use the encoder part of the network to transform the text into a vector representation or use only the decoder block, which is the backbone of the Large Language Models. The next chapter will cover each of these
components.

### Language Modeling
With the rise of LLMs, language modeling has become an essential part of
natural language processing. It means learning the probability distribution
of words within a language based on a large corpus. This learning process
typically involves predicting the next token in a sequence using either
classical statistical methods or novel deep learning techniques.

Large language models are trained based on the same objective to predict
the next word, punctuation mark, or other elements based on the seen
tokens in a text. These models become proficient by understanding the
distribution of words within their training data by guessing the probability
of the next word based on the context. For example, the model can
complete a sentence beginning with “I live in New” with a word like
“York” rather than an unrelated word such as “shoe”.

In practice, the models work with tokens, not complete words. This
approach allows for more accurate predictions and text generation by more
effectively capturing the complexity of human language.

### Tokenization
Tokenization is the initial phase of interacting with LLMs. It involves
breaking down the input text into smaller pieces known as tokens. Tokens
can range from single characters to entire words, and the size of these
tokens can greatly influence the model’s performance. Some models adopt
subword tokenization, breaking words into smaller segments that retain
meaningful linguistic elements.

Consider the following sentence, “The child’s coloring book.”

If tokenization splits the text after every white space character. The result
will be:
```
["The", "child's", “coloring”, "book."]
```
In this approach, you’ll notice that the punctuation remains attached to the
words like “child’s” and “book.”

Alternatively, tokenization can be done by separating text based on both
white spaces and punctuation; the output would be:
```
["The", "child", "'" , "s", “coloring”, "book", "."]
```

The tokenization process is model-dependent. It’s important to remember
that the models are released as a pair of pre-trained tokenizers and
associated model weights. There are more advanced techniques, like the
Byte-Pair encoding, which is used by most of the recently released models.
As demonstrated in the example below, this method also divides a word
such as “coloring” into two parts.

```
["The", "child", "'", "s", “color”, “ing”, "book", "."]
```

Subword tokenization further enhances the model’s language understanding
by splitting words into meaningful segments, like breaking “coloring” into
“color” and “ing.” This expands the model’s vocabulary and improves its
ability to grasp the nuances of language structure and morphology.
Understanding that the “ing” part of a word indicates the present tense
allows us to simplify how we represent words in different tenses. We no
longer need to keep separate entries for the base form of a word, like “play,”
and its present tense form, “playing.” By combining “play” with “ing,” we
can express “playing” without needing two separate entries. This method
increases the number of tokens to represent a piece of text but dramatically
reduces the number of tokens we need to have in the dictionary.

The tokenization process involves scanning the entire text to identify
unique tokens, which are then indexed to create a dictionary. This
dictionary assigns a unique token ID to each token, enabling a standardized
numerical representation of the text. When interacting with the models, this
conversion of text into token IDs allows the model to efficiently process
and understand the input, as it can quickly reference the dictionary to
decode the meaning of each token. We will see an example of this process
later in the book.

Once we have our tokens, we can process the inner workings of
transformers: embeddings.

### Embeddings
The next step after tokenization is to turn these tokens into something the
computer can understand and work with—this is where embeddings come
into play. Embeddings are a way to translate the tokens, which are words or
pieces of words, into a language of numbers that the computer can grasp.
They help the model understand relationships and context. They allow the
model to see connections between words and use these connections to
understand text better, mainly through the attention process, as we will see.

An embedding gives each token a unique numerical ID that captures its
meaning. This numerical form helps the computer see how similar two
tokens are, like knowing that “happy” and “joyful” are close in meaning,
even though they are different words.

This step is essential because it helps the model make sense of language in
a numerical way, bridging the gap between human language and machine
processing.

Initially, every token is assigned a random set of numbers as its embedding.
As the model is trained—meaning as it reads and learns from lots of text—
it adjusts these numbers. The goal is to tweak them so that tokens with
similar meanings end up with similar sets of numbers. This adjustment is
done automatically by the model as it learns from different contexts in
which the tokens appear.

While the concept of numerical sets, or vectors, might sound complex, they
are just a way for the model to store and process information about tokens
efficiently. We use vectors because they are a straightforward method for
the model to keep track of how tokens are related to each other. They are
basically just large lists of numbers.

### Training/Fine-Tuning
LLMs are trained on a large corpus of text with the objective of correctly
predicting the next token of a sequence. As we learned in the previous
language modeling subsection, the goal is to adjust the model’s parameters
to maximize the probability of a correct prediction based on the observed
data. Typically, a model is trained on a huge general-purpose dataset of
texts from the Internet, such as The Pile or CommonCrawl. Sometimes,
more specific datasets, such as the StackOverflow Posts dataset, are also an
example of acquiring domain-specific knowledge. This phase is also known
as the pre-training stage, indicating that the model is trained to learn
language comprehension and is prepared for further tuning.

The training process adjusts the model’s weights to increase the
likelihood of predicting the next token in a sequence. This adjustment
is based on the training data, guiding the model towards accurate token
predictions.

After pre-training, the model typically undergoes fine-tuning for a specific
task. This stage requires further training on a smaller dataset for a task (e.g.,
text translation) or a specialized domain (e.g., biomedical, finance, etc.).
Fine-tuning allows the model to adjust its previous knowledge of the
specific task or domain, enhancing its performance.

The fine-tuning process can be intricate, particularly for advanced models
such as GPT-4. These models employ advanced techniques and leverage
large volumes of data to achieve their performance levels.

### Prediction
The model can generate text after the training or fine-tuning phase by
predicting subsequent tokens in a sequence. This is achieved by inputting
the sequence into the model, producing a probability distribution over the
potential next tokens, essentially assigning a score to every word in the
vocabulary. The next token is selected according to its score. The generation
process will be repeated in a loop to predict one word at a time, so
generating sequences of any length is possible. However, keeping the
model’s effective context size in mind is essential.

### Context Size
The context size, or context window, is a crucial aspect of LLMs. It refers
to the maximum number of tokens the model can process in a single
request. Context size influences the length of text the model can handle at
any one time, directly affecting the model’s performance and the outcomes
it produces.

Different LLMs are designed with varying context sizes. For example,
OpenAI’s “gpt-3.5-turbo-16k” model has a context window capable of
handling 16,000 tokens. There is an inherent limit to the number of tokens a
model can generate. Smaller models may have a capacity of up to 1,000
tokens, while larger ones like GPT-4 can manage up to 32,000 tokens as of
the time we wrote this book.

### Hallucinations and Biases in LLMs
Hallucinations in AI systems refer to instances where these systems
produce outputs, such as text or visuals, inconsistent with facts or the
available inputs. One example would be if ChatGPT provides a compelling
but factually wrong response to a question. These hallucinations show a
mismatch between the AI’s output and real-world knowledge or context.

In LLMs, hallucinations occur when the model creates outputs that do
not correspond to real-world facts or context. This can lead to the
spread of disinformation, especially in crucial industries like
healthcare and education, where information accuracy is critical. Bias
in LLMs can also result in outcomes that favor particular perspectives
over others, possibly reinforcing harmful stereotypes and
discrimination.

An example of a hallucination could be if a user asks, “Who won the World
Series in 2025?” and the LLM responds with a specific winner. As of the
current date (Jan 2024), the event has yet to occur, making any response
speculative and incorrect.

Additionally, Bias in AI and LLMs is another critical issue. It refers to these
models’ inclination to favor specific outputs or decisions based on their
training data. If the training data primarily originates from a particular
region, the model may be biased toward that region’s language, culture, or
viewpoints. In cases where the training data encompasses biases, like
gender or race, the resulting outputs from the AI system could be biased or
discriminatory.

For example, if a user asks an LLM, “Who is a nurse?” and it responds, “She is
a healthcare professional who cares for
patients in a hospital,” this demonstrates a gender bias. The paradigm
inherently associates nursing with women, which needs to adequately
reflect the reality that both men and women can be nurses.

Mitigating hallucinations and bias in AI systems involves refining model
training, using verification techniques, and ensuring the training data is
diverse and representative. Finding a balance between maximizing the
model’s potential and avoiding these issues remains challenging.
