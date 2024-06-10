import time
import cv2
import numpy as np
import math
from ultralytics import YOLO

from agents import knowledge_aggregator_agent, creative_agent

from helper import get_image_from_api

# Initialize the video capture
def initialize_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 640)
    cap.set(4, 480)
    return cap

# Initialize the YOLO model
def initialize_model(weights_path="yolo-Weights/yolov8n.pt"):
    return YOLO(weights_path)

# Define the detectable objects by YOLO
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

# Process the frame using the YOLO model
def process_frame(frame, model):
    results = model(frame, stream=True)
    return results

# Annotate the frame with the detected objects
def annotate_frame(frame, results, classNames, font_scale=2, thickness=3, font_color=(255, 255, 255)):
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
            cv2.putText(frame, classNames[cls], org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    return frame, object_details

# Summarize the detected objects
def summarize_objects(object_details):
    unique_objects = {}
    for confidence, name in object_details:
        if name in unique_objects:
            unique_objects[name] += 1
        else:
            unique_objects[name] = 1
    return unique_objects

# Generate a summary string from the unique objects
def generate_summary_string(unique_objects):
    return " and ".join([f"there are {count} {name}(s)" for name, count in unique_objects.items()])

# Main function to run the video object detection
def main(video_path):
    cap = initialize_capture(video_path)
    model = initialize_model()
    
    last_update_time = 0
    generated_image = None

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = process_frame(frame, model)
        frame, object_details = annotate_frame(frame, results, classNames)

        # Run function which turns detected objects into a somewhat readable string
        unique_objects = summarize_objects(object_details)
        output_string = generate_summary_string(unique_objects)

        # Feed the detected objects string to the Knowledge aggreator agent
        generated_string = knowledge_aggregator_agent(output_string)
        

        # Each 5 seconds, generate an image based on the generated string
        current_time = time.time()
        if current_time - last_update_time >= 5:
            # Check if generated string is not empty
            creative_prompt = generated_string if generated_string else "No objects detected"

            # Feed the generated narrative to the Vision agent
            generated_image = get_image_from_api(creative_prompt)
            last_update_time = current_time

        # Create the frame with the detected objects and the generated image
        if generated_image is not None:
            height, width, _ = frame.shape
            generated_image = cv2.resize(generated_image, (width, height))
            combined_image = np.hstack((frame, generated_image))
        else:
            combined_image = frame

        # Display the frame
        cv2.imshow('Video and Generated Image', combined_image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function with the path to your local video file
if __name__ == "__main__":
    video_path = "video/4.mp4"  # Update this with your video file path
    main(video_path)
