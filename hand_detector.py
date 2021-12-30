import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self,  model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> None:
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def find_hands(self, img, draw=True):
        with self.mp_hands.Hands(model_complexity=self.model_complexity, min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as hands:
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if draw:
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            return img, results
