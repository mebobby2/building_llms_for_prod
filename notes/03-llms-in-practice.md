# LLMs in Practice
## Hallucinations and Bias
### Improving LLM Accuracy
#### Tuning the Text Generation Parameters
Parameters such as temperature, frequency penalty, presence penalty, and
top-p significantly influence LLM output—a lower temperature value
results in more predictable and reproducible responses. The frequency
penalty results in a more conservative use of repeated tokens. Increasing the
presence penalty encourages the model to generate new tokens that haven’t
previously occurred in the generated text. The “top-p” parameters control
response diversity by defining a cumulative probability threshold for
selecting words and customizing the model’s response range. All these
factors contribute to reducing the risk of hallucinations.

#### Leveraging External Documents with Retrievers Architectures
LLM accuracy can be improved by incorporating domain-specific
knowledge through external documents. This process updates the model’s
knowledge base with relevant information, enabling it to base its responses
on the new knowledge base. When a query is submitted, relevant
documents are retrieved using a “retriever” module, which improves the
model’s response. This method is integral to retriever architectures. These
architectures function as follows:
1. Upon receiving a question, the system generates an embedding
representation of it.
2. This embedding is used to conduct a semantic search within a
database of documents (by comparing embeddings and
computing similarity scores).
3. The LLM uses the top-ranked retrieved texts as context to
provide the final response. Typically, the LLM must carefully
extract the answer from the context paragraphs and not write
anything that cannot be inferred from them.

### Bias in LLMs
Large Language Models, including GPT-3.5 and GPT-4, have raised
significant privacy and ethical concerns. Studies indicate that these models
can harbor intrinsic biases, leading to the generation of biased or offensive
language. This amplifies the problems related to their application and
regulation.

LLM biases emerge from various sources, including the data, the
annotation process, the input representations, the models, and the
research methodology.

Training data lacking linguistic diversity can lead to demographic biases.
Large Language Models (LLMs) may unintentionally learn stereotypes
from their training data, leading them to produce discriminatory content
based on race, gender, religion, and ethnicity. For instance, if the training
data contains biased information, an LLM might generate content depicting
women in a subordinate role or characterizing certain ethnicities as
inherently violent or unreliable. Likewise, training the model on hate
speech or toxic content data could generate harmful outputs that reinforce
negative stereotypes and biases.

## Controlling LLM Outputs

### Parameters That Influence Text Generation
In addition to decoding, several parameters can be adjusted to influence text
generation. Key parameters, which include temperature, stop sequences,
frequency, and presence penalties, can be adjusted with the most popular
LLM APIs and Hugging Face models.

#### Temperature
The temperature parameter is critical in balancing text generation’s
unpredictability and determinism. A lower temperature setting produces
more deterministic and concentrated outputs, and a higher temperature
setting introduces randomness, producing diverse outputs. This parameter
functions by adjusting the logits before applying softmax in the text
generation process. This ensures the balance between the diversity of output
and its quality.
1. Logits: At the core of a language model’s prediction process is
the generation of a logit vector. Each potential next token has a
corresponding logit, reflecting its initial, unadjusted prediction
score.
2. Softmax: This function transforms logits into probabilities. A
key feature of the softmax function is ensuring that these
probabilities collectively equal 1.
3. Temperature: This parameter dictates the output’s randomness.
Before the softmax stage, the logits are divided by the
temperature value.
  * High temperature (e.g., > 1): As temperatures rise, the logits
  decrease, resulting in a more uniform softmax output. This
  enhances the possibility of the model selecting fewer likely
  terms, resulting in more diversified and innovative outputs,
  occasionally with higher errors or illogical phrases.
  * Low temperature (e.g., < 1): Lower temperatures cause an
  increase in logits, resulting in a more concentrated softmax
  output. As a result, the model is more likely to select the most
  probable word, resulting in more accurate and conservative
  outputs with a greater probability but less diversity.
  * Temperature = 1: There is no scaling of logits when the
  temperature is set to 1, preserving the underlying probability
  distribution. This option is seen as balanced or neutral.

In summary, the temperature parameter is a knob that controls the trade-off
between diversity (high temperature) and correctness (low temperature).

#### Stop Sequences
Stop sequences are designated character sequences that terminate the text
generation process upon their appearance in the output. These sequences
enable control over the length and structure of the generated text, ensuring
that the output adheres to specifications.

#### Frequency and Presence Penalties
Frequency and presence penalties are mechanisms that manage the
repetition of words in the generated text. The frequency penalty reduces the
probability of the model reusing repeatedly occurring tokens. The presence
penalty aims to prevent the model from repeating any token that has
occurred in the text, regardless of its frequency.
