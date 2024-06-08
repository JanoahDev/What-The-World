import time
import cv2
import numpy as np
import math
from ultralytics import YOLO

# Import the helper functions
from helpers import get_image_from_api, agent_that_describes, agent_that_creates


# Change the path to your local video file
video_path = "video/1.mp4"

# Start the video capture from the local video file
cap = cv2.VideoCapture(video_path)

# Set the frame dimensions (if necessary)
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


# Variables for timing
last_update_time = 0
generated_image = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)

    # Coordinates and object details
    object_details = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            object_details.append((confidence, classNames[cls]))
            org = [x1, y1]
            cv2.putText(frame, classNames[cls], org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Count unique objects
    unique_objects = {}
    for confidence, name in object_details:
        if name in unique_objects:
            unique_objects[name] += 1
        else:
            unique_objects[name] = 1

    # Create a string which gives a summary of the objects detected
    output_string = (" and ".join([f"there are {count} {name}(s)" for name, count in unique_objects.items()]))
    print("Detected object string: " + output_string)

    # Get the description from the agent
    output_string = agent_that_describes(output_string)
    print("Agent generated string: " + output_string)

    # Get the description from the agent
    output_string = agent_that_creates(output_string)
    print("Creative agent generated string: " + output_string)

    # Update the image every 5 seconds
    current_time = time.time()
    if current_time - last_update_time >= 5:
        prompt = output_string if output_string else "No objects detected"
        generated_image = get_image_from_api(prompt)
        last_update_time = current_time

    # Resize the generated image to match the height of the video frame
    if generated_image is not None:
        height, width, _ = frame.shape
        generated_image = cv2.resize(generated_image, (width, height))

    # Display the video frame and the generated image side by side
    if generated_image is not None:
        combined_image = np.hstack((frame, generated_image))
    else:
        combined_image = frame

    cv2.imshow('Video and Generated Image', combined_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()