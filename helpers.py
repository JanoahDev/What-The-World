import cv2
import random

def draw_boxes(image, boxes, confidences, class_ids, classes, color_map):
    """
    Draw bounding boxes on the image.

    Parameters:
    - image: The image on which to draw.
    - boxes: List of bounding boxes, each represented as [x, y, width, height].
    - confidences: List of confidences for each bounding box.
    - class_ids: List of class IDs for each bounding box.
    - classes: List of class names corresponding to class IDs.
    """
    font_scale = 2  # Increased font scale
    thickness = 3  # Increased thickness for the text
    font_color = (255, 255, 255)  # Changed font color to black

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = color_map[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y), color, -1)
        cv2.putText(image, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)



def predict(chosen_model, img, classes=[], conf=0.5):
    """
    chosen_model: The trained model to use for prediction
    img: The image to make a prediction on
    classes: (Optional) A list of class names to filter predictions to
    conf: (Optional) The minimum confidence threshold for a prediction to be considered

    The conf argument is used to filter out predictions with a confidence score lower than the specified threshold. This is useful for removing false positives.

    The function returns a list of prediction results, where each result contains the following information:

    name: The name of the predicted class
    conf: The confidence score of the prediction
    box: The bounding box of the predicted object
    """
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def generate_color_map(num_classes):
    """
    Generate a color map for the given number of classes.

    Parameters:
    - num_classes: The number of classes.

    Returns:
    - A dictionary mapping class IDs to colors.
    """
    random.seed(42)  # For reproducibility
    color_map = {}
    for i in range(num_classes):
        color_map[i] = [random.randint(0, 255) for _ in range(3)]
    return color_map

# predict_and_detect function
def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
  """
  Predicts and detects objects in an image using a chosen model.

  Args:
    chosen_model: The model used for prediction and detection.
    img: The input image.
    classes: Optional list of class names. If provided, only objects from these classes will be detected.
    conf: Optional confidence threshold for object detection. Defaults to 0.5.

  Returns:
    img: The input image with bounding boxes drawn around detected objects.
    results: The results of the prediction and detection.
  """
  # Predict objects in the image
  results = predict(chosen_model, img, classes, conf=conf)

  # Prepare lists to pass to the draw_boxes function
  boxes = []
  confidences = []
  class_ids = []
  classes = chosen_model.names  # Get the actual class names from the model

  # Generate a color map for the classes
  color_map = generate_color_map(len(classes))

  # Extract bounding box coordinates, confidences, and class IDs from the results
  for result in results:
    #print(result)
    for box in result.boxes:
      x1, y1, x2, y2 = box.xyxy[0]  # Extract xyxy format
      x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
      boxes.append([x, y, w, h])
      confidences.append(box.conf[0].item())
      class_ids.append(int(box.cls[0].item()))

  # Draw bounding boxes on the image
  draw_boxes(img, boxes, confidences, class_ids, classes, color_map)

  return img, results
