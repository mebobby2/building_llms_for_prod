from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pyvis.network import Network
import networkx as nx
load_dotenv()

KG_TRIPLE_DELIMITER = "<|>"
_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the text."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property.\n\n"
    "EXAMPLE 1\n"
    "It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "EXAMPLE 2\n"
    "I'm going to the store.\n\n"
    "Output: NONE\n"
    "EXAMPLE 3\n"
    "Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "EXAMPLE 4\n"
    "{text}"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

chain = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT | model

text = """The city of Paris is the capital and most populous city of
France. The Eiffel Tower is a famous landmark in Paris."""

triples = chain.invoke({"text": text})

def parse_triples(response, delimiter=KG_TRIPLE_DELIMITER):
    if not response:
        return []
    return response.split(delimiter)

triples_list = parse_triples(triples.content)
print(triples_list)

def create_graph_from_triples(triples_list):
    G = nx.DiGraph()
    for triple in triples_list:
        subject, predicate, obj = triple.split(", ")
        G.add_edge(subject.strip(), obj.strip(), label=predicate.strip())
    return G

def nx_to_pyvis(networkx_graph):
    pyvis_graph = Network(notebook=True)
    for node in networkx_graph.nodes():
        pyvis_graph.add_node(node)
    for edge in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(edge[0], edge[1], title=edge[2]['label'])
    return pyvis_graph

triples = [t.strip("()") for t in triples_list if t.strip()]
graph = create_graph_from_triples(triples)
pyvis_network = nx_to_pyvis(graph)

pyvis_network.toggle_hide_edges_on_drag(True)
pyvis_network.toggle_physics(False)
pyvis_network.set_edge_smooth("discrete")

pyvis_network.show("knowledge_graph.html")
