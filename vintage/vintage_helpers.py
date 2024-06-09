import requests
import cv2
import numpy as np
import base64
import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()



## --- Diffusion Functionality --- ##

#Create the api url based on the Runpod ID
api_url = "https://"+os.getenv("RUNPOD_ID")+"-5000.proxy.runpod.net/generate"

# Function that calls the api with a prompt and returns the image
def get_image_from_api(prompt: str):
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


# Function that calls the Vision agent
def vision_agent(prompt):
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


# Function that calls the Commentator agent
def commentator_agent( query: str) -> str:
    string = query

    # Create the conversation
    message = [
        SystemMessage(content="Your objective:You are an advanced language model trained to understand and analyze human behavior based on limited contextual data from object detection. Your task is to provide a detailed analysis of the perceived behavior and possible scenario based on the number of people and items detected in a room."),
        SystemMessage(content="instructions:Read the object detection data carefully, which includes the number of people and items in the room.Infer a possible scenario or behavior based on the detected objects and their quantities.Provide an insightful explanation of the perceived intentions, emotions, or outcomes based on the inferred scenario."),
        SystemMessage(content="Example: Object Detection Data: number of people: 5. Items detected: 1 projector, 1 screen, 5 chairs, 1 table. Analysis: The presence of five people in a room with a projector, screen, chairs, and a table suggests a meeting or a presentation scenario. The people are likely gathered to discuss a topic, with one person possibly leading the presentation while the others are participants. The environment indicates a formal or professional setting, and the projector and screen imply that visual aids or slides might be used for the discussion."),
        HumanMessage(content=string)
    ]

    # Invoke the Large Language Model:
    result = llm.invoke(message)

    # Extract the content from the result
    content = result.content

    return content