# What the world?! - Digital installation

This is a project made for a school assigment, the objective was to create an interactive art installation that analyzes visitor behavior in a museum and translates it into dynamic artworks. We didn't actually have to install the camera inside a museum, so I made two versions of the code. One which is based on a sample video captured in a museum and one which actually is fed live video, in this case a webcam; But this could be any camera source you link it too.

During the lessons, we learned to work with YOLO for object recognition, Langchain for creating LLM applications and agents, and diffusion models for generating images (Stable Diffusion).

This project merges all of the above techniques together to become a digital installation.


## Explanation of main code

1. Clone the github code into a local directory

2. Make sure that you have a diffusion runpod running (instructions underneath)

3. From that local directory, install the required packages. (Preferrably inside a virtual environment.)

`pip install -r requirements.txt`

4. Set the environment variables:

    - OPENAI_API_KEY   
    - RUNPOD_ID

5. Run the startup script

    - `python app-webcam.py` -> From your webcam feed

    - `python app-video.py` -> From a local video (located in video/yourvideo.mp4)


## Runpod instructions -> Diffusion API

1. Initiate a runpod with a graphics card of choice
2. Edit your pod and add port 5000 to the exposed HTTP list
3. In the workspace folder, upload the 2 files from the `/runpod_building_blocks/` Folder
4. From that folder, run the following commands:
    - `chmod +x run_app.sh`
    - `python3 -m venv venv`
    - `source venv/bin/activate`
    - `pip install flask transformers diffusers torch accelerate`


## Building blocks

The working model is based upon a couple of building blocks:

- Detecting objects from a webcam stream -> Turn them into a string

- Run that string through several AI agents who turn the detected objects & persons into a real narrative

- Send that narrative to a diffusion API which is ran on a runpod in this case

- Receive digital art based upon video feed as feedback


## References

- [Open CV](https://opencv.org/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Computer Vision Datasets](https://public.roboflow.com/)
- [ChatGPT](https://platform.openai.com/)
- [Langchain](https://www.langchain.com)
- [Langchain Library](https://python.langchain.com/v0.2/docs/integrations/tools/wikipedia/)


## Author

- Jonah De Smet (https://www.linkedin.com/in/jonah-de-smet-550214231/)


