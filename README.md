# What the world?! - Digital installation

This is a project made for a school assigment, the objective was to create an interactive art installation that analyzes visitor behavior in a museum and translates it into dynamic artworks. We didn't actually have to install the camera inside a museum, so I made two versions of the code. One which is based on a sample video captured in a museum and one which actually is fed live video, in this case a webcam; But this could be any camera source you link it too.

During the lessons, we learned to work with YOLO for object recognition, Langchain for creating LLM applications and agents, and diffusion models for generating images (Stable Diffusion).

This project merges all of the above techniques together to become a digital installation.


## Explanation of code

1. 

2. 



## Building blocks

The working model is based upon a couple of building blocks:

- Detecting objects from a webcam stream -> Turn them into a string
- 
- Send a string to a diffusion API which is ran on a runpod in this case
- ...


## Training YOLO to detect new objects (optional)

This code is made to work out of the box with the objects and behaviour we trained it to detect. If you wish to train it any further you can always do so following this manual:

1. Install the needed library in conda environment

`pip install ultralytics`

2. Start training the model via

`yolo detect train data=data.yaml model=yolov9c.pt epochs=10 imgsz=640 batch=8`


## Runpod instructions -> Diffusion API

1. Initiate a runpod with a graphics card of choice
2. Edit your pod and add port 5000 to the exposed HTTP list
3. In the workspace folder, upload the 2 files from the /building_blocks/runpod_diffusion_server/ Folder
4. From that folder, run the following commands:
5. `chmod +x run_app.sh`
6. `python3 -m venv venv`
7. `source venv/bin/activate`
8. `pip install flask transformers diffusers torch accelerate`


## References

- [Open CV](https://opencv.org/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Computer Vision Datasets](https://public.roboflow.com/)
- [ChatGPT]()
- [Langchain](https://www.langchain.com)
- [Langchain Library]()


## Author

- Jonah De Smet ()


