# Summarize Text
A central question for building a summarizer is how to pass your documents into the LLM's context window. Two common approaches for this are:

1. Stuff: Simply "stuff" all your documents into a single prompt. This is the simplest approach (see here for more on the create_stuff_documents_chain constructor, which is used for this method).
2. Map-reduce: Summarize each document on its own in a "map" step and then "reduce" the summaries into a final summary (see here for more on the MapReduceDocumentsChain, which is used for this method).

Note that map-reduce is especially effective when understanding of a sub-document does not rely on preceding context. For example, when summarizing a corpus of many, shorter documents. In other cases, such as summarizing a novel or body of text with an inherent sequence, iterative refinement may be more effective.

![alt text](<Screenshot 2025-05-27 at 4.21.49 PM.png>)

## Source
https://python.langchain.com/docs/tutorials/summarization/
