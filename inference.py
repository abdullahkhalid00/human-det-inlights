import numpy as np
from ultralytics import YOLO


def load_model(path="models/yolov8n.pt"):
    return YOLO(path)

def run_inference(model, frame):
    return model(source=frame)
