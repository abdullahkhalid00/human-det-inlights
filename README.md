# Human Detection Using YOLOv8 and OpenCV

This repository contains an object detection and tracking project using OpenCV and the pretrained YOLOv8 weights. The project is part of a coding assignment from InLights.

## Project Breakdown

- Loading and Inference
- Creating Bounding Boxes
- Object Tracking using KCF Tracker
- Handling Click Events

### Challenges

Since I am a complete beginner to OpenCV, this project was a big challenge for me. Some of the areas where I struggled in are:

- Creating bounding boxes explicitly around objects marked as a 'person'.
- Handling click events like starting a timer each time a user clicks on a bounded box.

### Limitations

As of now the project has the following limitations:

- Runs inference on local webcam instead of an RSTP video stream.
- Does not create bounded boxes for multiple persons.

## Usage

Create a virtual environment using pip or conda and activate it.

```bash
python -m venv {environment_name}
{environment_name}\Scripts\activate
```

Install the required dependencies mentioned in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```

Run the [`main.py`](main.py) file for inference.

```bash
python main.py
```
