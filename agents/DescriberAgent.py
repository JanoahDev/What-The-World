import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Set the OPENAI_API_KEY environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

print(openai_api_key)

# Init the Large Language Model
llm = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key)

string = "there are 1 person(s) and there are 1 smartphones"

# Create the conversation
message = [
    SystemMessage(content="We have pointed a camera into a space. A user will input what it sees inside the space. Describe what you see"),
    HumanMessage(content=string)
]

# Invoke the Large Language Model:
result = llm.invoke(message)

# Extract the content from the result
content = result.content
print("Content:", content)