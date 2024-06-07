import time
import requests
import cv2
import numpy as np
import base64
import math
from ultralytics import YOLO

# Start the webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

# Define the model
model = YOLO("yolo-Weights/yolov8n.pt")

# Define the objects
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


# Insert API URL here
api_url = 'https://qbwvbikcp6d37t-5000.proxy.runpod.net/generate'


# Function that calls the api with a prompt and returns the image
def get_image_from_api(prompt):
    response = requests.post(api_url, json={'prompt': prompt})
    data = response.json()
    img_str = data['image_url'].split(",")[1]
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

last_update_time = 0
generated_image = None

while True:
    success, webcam = cap.read()
    results = model(webcam, stream=True)

    # Coordinates and object details
    object_details = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
            cv2.rectangle(webcam, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            object_details.append((confidence, classNames[cls]))
            org = [x1, y1]
            cv2.putText(webcam, classNames[cls], org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Count unique objects
    unique_objects = {}
    for confidence, name in object_details:
        if name in unique_objects:
            unique_objects[name] += 1
        else:
            unique_objects[name] = 1

    # Create a string which gives a summary of the objects detected
    output_string = (" and ".join([f"there are {count} {name}(s)" for name, count in unique_objects.items()]))
    print(output_string)

    # Update the image every 5 seconds
    current_time = time.time()
    if current_time - last_update_time >= 5:
        prompt = output_string if output_string else "No objects detected"
        generated_image = get_image_from_api(prompt)
        last_update_time = current_time


    # Resize the generated image to match the height of the webcam feed
    if generated_image is not None:
        height, width, _ = webcam.shape
        generated_image = cv2.resize(generated_image, (width, height))


    # Display the webcam feed and the generated image side by side
    if generated_image is not None:
        combined_image = np.hstack((webcam, generated_image))
    else:
        combined_image = webcam

    cv2.imshow('Webcam and Generated Image', combined_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()