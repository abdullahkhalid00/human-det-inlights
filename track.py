import cv2 as cv


def create_tracker():
    return cv.TrackerKCF_create()

def update_tracker(tracker, frame):
    ret, box = tracker.update(frame)
    return ret, box
