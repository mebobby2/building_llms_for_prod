import cohere
import os

os.environ['COHERE_API_KEY'] = "w60qAqxwH9Ce0fxE1WAkr4zTjSNrtK5UXhDfOZ1e"
co = cohere.Client(os.environ['COHERE_API_KEY'])

query = "What is the capital of the United States?"
# query = "What is the capital of Nevada?"
documents = [
    """Carson City is the capital city of the American state of Nevada. At
the 2010 United States Census, Carson City had a population of
55,274.""",
    """The Commonwealth of the Northern Mariana Islands is a group of islands
in the Pacific Ocean that are a political division controlled by the
United States. Its capital is Saipan.""",
    """Charlotte Amalie is the capital and largest city of the United States
Virgin Islands. It has about 20,000 people. The city is on the island of
Saint Thomas.""",
    """Washington, D.C. (also known as simply Washington or D.C., and
officially as the District of Columbia) is the capital of the United
States. It is a federal district. """,
    """Capital punishment (the death penalty) has existed in the United
States since before the United States was a country. As of 2017, capital
punishment is legal in 30 of the 50 states.""",
    """North Dakota is a state in the United States. 672,591 people lived in
North Dakota in the year 2010. The capital and seat of government is
Bismarck."""
]

response = co.rerank(query=query, documents=documents,
                    model='rerank-v3.5',
                    top_n=3)

for idx, r in enumerate(response.results):
    print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
    print(f"Document: {documents[r.index]}")
    print(f"Relevance Score: {r.relevance_score:.2f}")
    print("\n")
