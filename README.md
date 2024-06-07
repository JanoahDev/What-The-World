# What the world?! - Digital installation

In deze opdracht creëren jullie een interactieve kunstinstallatie die bezoekersgedrag in een museum analyseert en vertaalt naar dynamische kunstwerken. Je moet de camera niet echt in een museum te installeren, je kan je eigen omgeving of een foto van een museum gebruiken (ter demo).

Tijdens de lessen hebben we geleerd te werken met YOLO voor objectherkenning, Langchain voor het maken van LLM applicaties en agents, en diffusion modellen voor het genereren van afbeeldingen (Stable Diffusion). 

## Inference

The working model is based upon a couple of building blocks:

- Detecting objects from a webcam stream -> Turn them into a string
- Seind a string to an API which is ran on drop
- ...

## Training

Install the needed library in conda environment

`pip install ultralytics`

Start training the model via

`yolo detect train data=data.yaml model=yolov9c.pt epochs=10 imgsz=640 batch=8`


## Runpod requirements

1. Initiate a runpod with a graphics card of choice
2. In the workspace folder, upload the 2 files from the /building_blocks/runpod_diffusion_server/ Folder
3. From that folder, run the following commands:
4. `chmod +x run_app.sh`
5. `python3 -m venv venv`
6. `source venv/bin/activate`
7. `pip install flask transformers diffusers torch accelerate`


## References

- [Open CV](https://opencv.org/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Computer Vision Datasets](https://public.roboflow.com/)


## Author

- Jonah De Smet


