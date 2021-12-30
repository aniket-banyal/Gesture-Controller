import enum

import mediapipe as mp

from model import KeyPointClassifier

mp_hands = mp.solutions.hands


index_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP.value
middle_tip = mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value
ring_tip = mp_hands.HandLandmark.RING_FINGER_TIP.value
pinky_tip = mp_hands.HandLandmark.PINKY_TIP.value
thumb_tip = mp_hands.HandLandmark.THUMB_TIP.value

fingers = [index_tip, middle_tip, ring_tip, pinky_tip, thumb_tip]


class Finger(enum.Enum):
    INDEX = 0
    MIDDLE = 1
    RING = 2
    PINKY = 3
    THUMB = 4


class Gesture(enum.Enum):
    OPEN = enum.auto()
    FIST = enum.auto()
    INDEX_UP = enum.auto()
    FIRST_2 = enum.auto()
    FIRST_3 = enum.auto()
    V = enum.auto()
    FRIST2_THUMB = enum.auto()
    YO = enum.auto()
    THUMBS_UP = enum.auto()
    NONE = enum.auto()
    L = enum.auto()


class GestureDetector:
    def __init__(self, img_width, img_height) -> None:
        self.img_width = img_width
        self.img_height = img_height
        self.keypoint_classifier = KeyPointClassifier(img_width, img_height)

    def find_gesture(self, landmarks, handedness):
        label = self.keypoint_classifier(landmarks)

        if label == 'None':
            return Gesture.NONE
        if label == 'Open':
            return Gesture.OPEN
        elif label == 'Close':
            return Gesture.FIST
        elif label == 'Index':
            return Gesture.INDEX_UP
        elif label == 'First2_thumb':
            return Gesture.FRIST2_THUMB
        elif label == 'V':
            return Gesture.V
        elif label == 'Thumbs_up':
            return Gesture.THUMBS_UP
        elif label == 'First3':
            return Gesture.FIRST_3
        elif label == 'Yo':
            return Gesture.YO
        elif label == 'L':
            return Gesture.L

    def get_hand_label(self, handedness):
        return handedness.classification[0].label
