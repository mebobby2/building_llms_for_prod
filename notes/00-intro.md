# Introduction
...concentrate on the essential tech stack identified for adapting a large language model (LLM) to a specific use case and achieving a sufficient threshold of accuracy and reliability for scalable use by paying customers. Specifically, it will cover Prompt Engineering, Fine-tuning, and Retrieval-Augmented Generation (RAG).

There are various ways to adapt an off-the-shelf “foundation model” LLM to a specific application and use case. The initial decision is whether to use an LLM via API or a more flexible platform where you have full access to the model weights. Some may also want to experiment with training their own models; however, in our opinion, this will rarely be practical or economical outside the leading AI labs and tech companies.

## Why Prompt Engineering, Fine-Tuning, and RAG?
LLMs such as GPT-4 often lack domain-specific knowledge, making generating accurate or relevant responses in specialized fields challenging. They can also struggle with handling large data volumes, limiting their utility in data-intensive scenarios. Another critical limitation is their difficulty processing new or technical terms, leading to misunderstandings or incorrect information. Hallucinations, where LLMs produce false or misleading information, further complicate their use. Hallucinations are a direct result of the model training goal of the next token prediction - to some extent, they are a feature that allows “creative” model answers. However, it is difficult for an LLM to know when it is answering from memorized facts and imagination. This creates many errors in LLM-assisted workflows, making them difficult to identify. Alongside hallucinations, LLMs sometimes also simply fail to use available data effectively, leading to irrelevant or incorrect responses.

LLMs are generally used in production for performance and productivity- enhancing “copilot” use cases, with a human still fully in the loop rather than for fully automated tasks due to these limitations. But there is a long journey from a basic LLM prompt to sufficient accuracy, reliability, and observability for a target copilot use case. This journey is called the “march of 9s” and is popularized in self-driving car development. The term describes the gradual improvement in reliability, often measured in the number of nines (e.g., 99.9% reliability) needed to reach human-level performance eventually.

RAG consists of augmenting LLMs with specific data and requiring the model to use and source this data in its answer rather than relying on what it may or may not have memorized in its model weights. We love RAG because it helps with:
1. Reducing hallucinations by limiting the LLM to answer based on
existing chosen data.
2. Helping with explainability, error checking, and copyright issues
by clearly referencing its sources for each comment.
3. Giving private/specific or more up-to-date data to the LLM.
4. Not relying too much on black box LLM training/fine-tuning for
what the models know and have memorized.

Another way to increase LLM performance is through good prompting. Multiple techniques have been found to improve model performance. These methods can be simple, such as giving detailed instructions to the models or breaking down big tasks into smaller ones to make them easier for the model to handle. Some prompting techniques are:
1. “Chain of Thought” prompting involves asking the model to think
through a problem step by step before coming up with a final answer.
The key idea is that each token in a language model has a limited
“processing bandwidth” or “thinking capacity.” The LLMs need these
tokens to figure things out. By asking it to reason through a problem
step by step, we use the model’s total capacity to think and help it
arrive at the correct answer.
2. “Few-Shot Prompting” is when we show the model examples of
the answers we seek based on some given questions similar to those
we expect the model to receive. It’s like showing the model a pattern
of how we want it to respond.
3. “Self-Consistency” involves asking the same question to multiple
versions of the model and then choosing the answer that comes up
most often. This method helps get more reliable answers.
