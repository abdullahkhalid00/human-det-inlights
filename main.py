import cv2 as cv
import time
import numpy as np
from inference import load_model, run_inference
from track import create_tracker, update_tracker
from utils import generate_random_color, draw_bounding_box, load_yolo_classes


# initialize the yolov8 model
model = load_model()

# initialize variables
persons = []
selected_person = None
start_time = None
timer_active = False
yolo_classes = load_yolo_classes()

def click_event(event, x, y, flags, param):
    global selected_person, start_time, timer_active, persons

    if event == cv.EVENT_LBUTTONDOWN:
        for i, (bbox, color, label, tracker) in enumerate(persons):
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            if p1[0] <= x <= p2[0] and p1[1] <= y <= p2[1]:
                if selected_person == i:
                    timer_active = False
                    persons[i][1] = generate_random_color()
                    selected_person = None
                else:
                    selected_person = i
                    start_time = time.time()
                    timer_active = True
                    persons[i][1] = (0, 0, 255)

# open webcam
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# create a window and set the mouse callback function
cv.namedWindow('Webcam')
cv.setMouseCallback('Webcam', click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run yolo inference
    results = run_inference(model, frame)

    # extract person detections and update persons list
    new_persons = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]
            cls = yolo_classes[str(int(box.cls[0]))]
            if cls == "person" and conf > 0.3:  # class 0 is person
                bbox = (x1, y1, x2 - x1, y2 - y1)
                color = generate_random_color()
                tracker = create_tracker()
                tracker.init(frame, bbox)
                new_persons.append([bbox, color, cls, tracker])

    persons = new_persons if not persons else persons

    # update trackers and draw bounding boxes
    for i, (bbox, color, label, tracker) in enumerate(persons):
        ret, updated_bbox = update_tracker(tracker, frame)
        if ret:
            persons[i][0] = updated_bbox
            if selected_person == i and timer_active:
                elapsed_time = time.time() - start_time
                draw_bounding_box(frame, updated_bbox, color, label, elapsed_time)
            else:
                draw_bounding_box(frame, updated_bbox, color, label)

    # display the frame
    cv.imshow('Webcam', frame)

    # break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv.destroyAllWindows()
