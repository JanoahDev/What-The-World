from ultralytics import YOLO
import cv2
import math 

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
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, webcam = cap.read()
    results = model(webcam, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes
        object_details = []
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(webcam, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])
            
            # Append the object details to the list
            object_details.append((confidence, classNames[cls]))

           
            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(webcam, classNames[cls], org, font, fontScale, color, thickness)
    

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

    cv2.imshow('Webcam', webcam)
    if cv2.waitKey(1) == ord('q'):
        break

    

cap.release()
cv2.destroyAllWindows()