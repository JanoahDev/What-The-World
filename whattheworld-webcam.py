import time
import cv2
import numpy as np
import math
from ultralytics import YOLO

from helpers import get_image_from_api, agent_that_describes, agent_that_creates


#Vision Agent: Real-time beschrijving van bezoekersgedrag.
##Voorbeeld: "Twee bezoekers wijzen naar het schilderij."

#Commentator Agent: Biedt context en analyse van de waargenomen gedragingen. Gebruik hiervoor taalmodellen (GPT) met goeie prompting.
##Voorbeeld: "Deze interactie suggereert een hoge mate van interesse in het kunstwerk."

#Knowledge Aggregator Agent (Langchain): Consolideert de informatie van de Vision en Commentator Agents tot een samenhangend verhaal. Gebruik hiervoor ook taalmodellen.
##Voorbeeld: "Momenteel bekijken en bespreken twee bezoekers een schilderij, wat wijst op een sterke betrokkenheid bij het kunstwerk."

#Creative Agent: Gebruikt een diffusion model om kunst te genereren op basis van de geaggregeerde kennis.
##Voorbeeld: Een abstract digitaal kunstwerk dat de energie en focus van de bezoekersdiscussie symboliseert.


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



def initialize_capture(device_index=1):
    cap = cv2.VideoCapture(device_index)
    cap.set(3, 640)
    cap.set(4, 480)
    return cap


def initialize_model(weights_path="yolo-Weights/yolov8n.pt"):
    return YOLO(weights_path)


def process_frame(frame, model):
    results = model(frame, stream=True)
    return results


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


def summarize_objects(object_details):
    unique_objects = {}
    for confidence, name in object_details:
        if name in unique_objects:
            unique_objects[name] += 1
        else:
            unique_objects[name] = 1
    return unique_objects


def generate_summary_string(unique_objects):
    return " and ".join([f"there are {count} {name}(s)" for name, count in unique_objects.items()])


def main():
    cap = initialize_capture()
    model = initialize_model()
    
    last_update_time = 0
    generated_image = None

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = process_frame(frame, model)
        frame, object_details = annotate_frame(frame, results, classNames)

        unique_objects = summarize_objects(object_details)
        output_string = generate_summary_string(unique_objects)
        print("Detected object string: " + output_string)

        output_string = agent_that_describes(output_string)
        print("Agent generated string: " + output_string)

        output_string = agent_that_creates(output_string)
        print("Creative agent generated string: " + output_string)

        current_time = time.time()
        if current_time - last_update_time >= 5:
            prompt = output_string if output_string else "No objects detected"
            generated_image = get_image_from_api(prompt)
            last_update_time = current_time

        if generated_image is not None:
            height, width, _ = frame.shape
            generated_image = cv2.resize(generated_image, (width, height))
            combined_image = np.hstack((frame, generated_image))
        else:
            combined_image = frame

        cv2.imshow('Video and Generated Image', combined_image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
