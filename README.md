# Gesture Controller

This program allows you to control your computer using hand gestures

# Features


<details>
<summary>Move Cursor</summary>
<video src='https://user-images.githubusercontent.com/40134444/152646039-e3917655-e7fc-4606-b438-d3aa4d269b36.mp4'/>
</details>

<details>
<summary>Left Click</summary>
<video src='https://user-images.githubusercontent.com/40134444/152646176-3ddee4dd-f387-4db2-915a-bdcff1baefae.mp4'/>
</details>

<details>
<summary>Right Click</summary>
<video src='https://user-images.githubusercontent.com/40134444/152646181-e239cadb-6134-4a81-bfb4-051736422336.mp4'/>
</details>

<details>
<summary>Double Click</summary>
<video src='https://user-images.githubusercontent.com/40134444/152646195-949c3680-d297-43c8-8838-2cfa8e0b9bd9.mp4'/>
</details>

<details>
<summary>Scroll</summary>
<video src='https://user-images.githubusercontent.com/40134444/152646511-f9238898-d665-46d1-8eba-c4fe48435259.mp4'/>
</details>

<details>
<summary>Switch window</summary>
<video src='https://user-images.githubusercontent.com/40134444/152646235-035dc75a-8957-4daa-8875-49ed9216a41d.mp4'/>
</details>

<details>
<summary>Volume Control</summary>
<video src='https://user-images.githubusercontent.com/40134444/152646248-96194e40-c8d4-4b13-a047-a5ad4c98e0b9.mp4'/>
</details>

<details>
<summary>Drag and Drop</summary>
<video src='https://user-images.githubusercontent.com/40134444/152646208-891d6872-6f8f-41e0-a57f-ea5b719b9b1e.mp4'/>
</details>


## Overview

`OpenCV` is used for getting image from webcam.  
`Mediapipe` is used to extract hand landmarks from this image.  
`GestureDetector` uses a deep learning model to classify gesture based on hand landmarks.  
`GestureController` executes the command corresponding to that gesture.

`model` folder contains the files of deep learning model used for gesture classification.  
 
## Getting Started

After cloning this repo run the following commands -

```bash
pip install -r requirements.txt
python main.py
```
