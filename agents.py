import os 
from dotenv import load_dotenv

# Langchain
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain import hub

# Langchain Community
#from langchain.tools import WikipediaQueryRun
#from langchain_community.utilities import WikipediaAPIWrapper

# Import the helper function
from helper import get_image_from_api

# Load environment variables
load_dotenv()

# Set the OPENAI_API_KEY environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the Large Language Model
llm = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key)


# --- Textual agents --- #

# Function that calls the Vision agent
def vision_agent(query: str) -> str:

    # Create the conversation
    message = [
        SystemMessage(content="We have pointed a camera into a space. A user will input what it sees inside a space. Describe what you see"),
        HumanMessage(content=query)
    ]

    # Invoke the Large Language Model:
    result = llm.invoke(message)

    # Extract the content from the result
    content = result.content

    return content


# Function that calls the Commentator agent
def commentator_agent(query: str) -> str:

    # Create the conversation
    message = [
        SystemMessage(content="Your objective:You are an advanced language model trained to understand and analyze human behavior based on limited contextual data from object detection. Your task is to provide a detailed analysis of the perceived behavior and possible scenario based on the number of people and items detected in a room."),
        SystemMessage(content="instructions:Read the object detection data carefully, which includes the number of people and items in the room.Infer a possible scenario or behavior based on the detected objects and their quantities. Provide an insightful explanation based on the inferred scenario."),
        SystemMessage(content="Example: Object Detection Data: number of people: 5. Items detected: 1 projector, 1 screen, 5 chairs, 1 table. Result: The presence of five people in a room with a projector, screen, chairs, and a table suggests a meeting or a presentation scenario. The people are likely gathered to discuss a topic, with one person possibly leading the presentation while the others are participants. The environment indicates a formal or professional setting, and the projector and screen imply that visual aids or slides might be used for the discussion."),
        HumanMessage(content=query)
    ]

    # Invoke the Large Language Model:
    result = llm.invoke(message)

    # Extract the content from the result
    content = result.content

    return content


# Define the tools
vision_tool = Tool(
    name="VisionAgent",
    func=vision_agent,
    description="Generates a description of the scene based on the input prompt."
)

commentator_tool = Tool(
    name="CommentatorAgent",
    func=commentator_agent,
    description="Analyzes the scene description and infers a scenario."
)


def knowledge_aggregator_agent(query):

    #wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # 1. Initialize the agent with the tools using the new constructor method
    tools = [vision_tool, commentator_tool]

    # 2. Create the prompt
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # 3. Link vision & commentator agents to OpenAI functions 
    agent = create_openai_functions_agent(llm, tools, prompt)

    # 4. Create an AgentExecutor to handle the steps
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 5. Invoke the agent
    result = agent_executor.invoke({"input": query})

    result = str(result)

    # 6. Send back the result
    return str(result)



# --- Creative agent -> Poging --- #
 
# Define the tools
generator_tool = Tool(
    name="GeneratorAgent",
    func=get_image_from_api,
    description="Send a prompt to the diffusion model and get an image back."
)

# Define the creative agent
def creative_agent(query: str) -> str:
    # 1. Initialize the agent with the tools using the new constructor method
    tools = [generator_tool]

    # 2. Create the prompt
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # 3. Link vision & commentator agents to OpenAI functions 
    agent = create_openai_functions_agent(llm, tools, prompt)

    # 4. Create an AgentExecutor to handle the steps
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 5. Invoke the agent
    image = agent_executor.invoke({"input": query})

    # 6. Send back the result
    return image
