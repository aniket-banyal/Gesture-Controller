import cv2

from gesture_controller import GestureController
from hand_detector import HandDetector

width, height = 640, 480

mouse_move_smoothening_factor = 4

hand_detector = HandDetector(min_detection_confidence=0.3, min_tracking_confidence=0.3)
gesture_controller = GestureController(width, height, mouse_move_smoothening_factor)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def main():
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image, results = hand_detector.find_hands(cv2.flip(image, 1))

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks = hand_landmarks.landmark
                gesture_controller.run(landmarks, handedness)

        cv2.imshow('Gesture Controller', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


main()
