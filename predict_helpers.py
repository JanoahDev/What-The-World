import cv2
import math

def draw_boxes(webcam, results, classNames):
    font_scale = 2  # Increased font scale
    thickness = 3  # Increased thickness for the text
    font_color = (255, 255, 255)  # Changed font color to black

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
            cv2.putText(webcam, classNames[cls], org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
    return object_details, webcam