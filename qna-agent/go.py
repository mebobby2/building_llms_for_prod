from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

tools = load_tools(["google-search", "llm-math"],llm=chat)

agent = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

query = """What's the result of 1000 plus the number of goals scored in
the soccer world cup in 2018?"""
response = agent.invoke(query)
print(response)
