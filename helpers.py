import requests
import cv2
import numpy as np
import base64
import os
from dotenv import load_dotenv

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()



## --- Diffusion Functionality --- ##

#Create the api url based on the Runpod ID
api_url = "https://"+os.getenv("RUNPOD_ID")+"-5000.proxy.runpod.net/generate"

# Function that calls the api with a prompt and returns the image
def get_image_from_api(prompt):
    response = requests.post(api_url, json={'prompt': prompt})
    data = response.json()
    img_str = data['image_url'].split(",")[1]
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img



## --- Langchain Functionality --- ##

# Set the OPENAI_API_KEY environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Init the Large Language Model
llm = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key)


# Function that calls the describer agent
def agent_that_describes(prompt):
    string = prompt

    # Create the conversation
    message = [
        SystemMessage(content="We have pointed a camera into a space. A user will input what it sees inside a space. Describe what you see"),
        HumanMessage(content=string)
    ]

    # Invoke the Large Language Model:
    result = llm.invoke(message)

    # Extract the content from the result
    content = result.content

    return content


# Function that calls the describer agent
def agent_that_creates(prompt):
    string = prompt

    # Create the conversation
    message = [
        SystemMessage(content="You get a description of what is happening inside a room. Create an abstract narrative of max 77 characters. This will act as the prompt for a diffusion model which will generate an art frame based on the narrative."),
        HumanMessage(content=string)
    ]

    # Invoke the Large Language Model:
    result = llm.invoke(message)

    # Extract the content from the result
    content = result.content

    return content



## ---  --- ##