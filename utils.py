import json
import random
import cvzone
import cv2 as cv


def generate_random_color():
    while True:
        color = [random.randint(0, 255) for i in range(3)]
        if color[0] < 150:  # ensure the red component is not dominant
            break
    return tuple(color)

def draw_bounding_box(frame, bbox, color, label, timer=None):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv.rectangle(frame, p1, p2, color, 2, 1)
    cvzone.putTextRect(frame, label, (p1[0], p2[1] - 10))
    if timer is not None:
        cvzone.putTextRect(frame, f"Timer: {timer:.1f}s", (10, 30))
    return frame

def load_yolo_classes(path="yolo_classes.json"):
    with open(path, "r") as file:
        classes = json.load(file)
    return classes['class']
