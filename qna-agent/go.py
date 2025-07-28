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



# > Entering new AgentExecutor chain...
# Question: What's the result of 1000 plus the number of goals scored in the soccer world cup in 2018?
# Thought: I need to find out how many goals were scored in the 2018 World Cup and then add 1000 to that number.
# Action: google_search
# Action Input: "number of goals scored in 2018 world cup"
# Observation: Dec 9, 2022 ... Top goalscorers at the 2018 FIFA World Cup in Russia · Harry Kane (England) · Antoine Griezmann · Romelu Lukaku · Denis Cheryshev · Cristiano Ronaldo ... Total number of goals scored: 169 · Average goals per match: 2.64 · Total number of braces: 10 · Total number of hat-tricks: 2 · Total number of penalty kicks ... FIFA World Cup Scoring Stats - 2018 · Lucas Hernández · Antoine Griezmann · Kevin De Bruyne · Eden Hazard · Thomas Meunier · Philippe Coutinho · Viktor Claesson · Artem ... Goal difference in all group matches;; Number of goals scored in all group matches;; Points obtained in the matches played between the teams in question;; Goal ... A chi-square test indicated that there was a significant difference in the type of possession (χ2 (1, n = 103) = 43.58, p = 0.00). The highest number of goals ... Check out the top scorers list of World Cup 2018 with Golden Boot prediction. Get highest or most goal scorer player in 2018 FIFA World Cup. Jan 31, 2020 ... The aim of this study was to analyse the goal scoring patterns during the 2018 FIFA World Cup. All goals scored during the tournament were analysed using the ... Nov 20, 2023 ... He had an incredible World Cup campaign scoring 4 goals and getting 2 assists. He also had a good La Liga campaign, Atletico finished second ... Jan 6, 2023 ... ... goals in 2010 to 2018 World Cup Champions. PLoS ONE 18(1): e0280030 ... Number of goals scored, and goals given up by each team and ... Jan 6, 2023 ... Greater numbers of passes and shorter possession durations result in increased likelihood of goals in 2010 to 2018 World Cup Champions · Abstract.
# Thought:Thought: The 2018 World Cup had 169 goals scored.
# Action: Calculator
# Action Input: 1000 + 169
# Observation: Answer: 1169
# Thought:Thought: I now know the final answer.
# Final Answer: 1169

# > Finished chain.
# {'input': "What's the result of 1000 plus the number of goals scored in\nthe soccer world cup in 2018?", 'output': '1169'}
