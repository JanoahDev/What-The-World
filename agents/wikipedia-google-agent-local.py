import os 
from dotenv import load_dotenv

from langchain import hub
 
#Langchain
from langchain.schema import HumanMessage, SystemMessage

# Langchain Community
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

# Langchain OpenAI
from langchain_openai import ChatOpenAI

# Langchain agents
from langchain.agents import create_openai_functions_agent, AgentExecutor

# Set the OPENAI_API_KEY environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")



print(openai_api_key)


# Init the Large Language Model
llm = ChatOpenAI(
    temperature=0.5, 
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not needed")

# Create the conversation
message = [
    SystemMessage(content="A user will input a year and you will get an event that happened in that year."),
    HumanMessage(content="1999")
]

# Invoke the Large Language Model:
# result = llm.invoke(message)
# print(result)


# Create the prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# Make an instance of wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wikipedia.run("HUNTER X HUNTER")
#print(result)

#Make an instance of google
googleSearch = GoogleSearchAPIWrapper()
googleTool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=googleSearch.run,
)

# Test google search with a random question
#resultgoogle = googleTool.run("Obama's first name?")
#print(resultgoogle)

# Create the tools
tools = [googleTool]

# 1. Link wikipedia superpowers to OpenAI functions 
agent = create_openai_functions_agent(llm, tools, prompt)

# 2. Create an AgentExecuter
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) #Verbose maakt zichtbaar wat de agent aan het doen is

# 3. Excecute the agent
result = agent_executor.invoke({"input": "How do I create a VHS glow effect in blender? Look it up"})

# 4. Print the result
print(result)
