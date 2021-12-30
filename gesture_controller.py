import enum

import mediapipe as mp
import numpy as np
import pyautogui

from gesture_detector import Finger, Gesture, GestureDetector, fingers
from point import Point
from ui_controls import (DoubleClickThread, DragThread, MouseClickThread,
                         MouseMoveThread, ScrollThread, VolumeThread,
                         WindowSwitchThread)

mp_hands = mp.solutions.hands


class Mode(enum.Enum):
    NORMAL = enum.auto()
    MOVE_MOUSE = enum.auto()
    SWITCH_WIN = enum.auto()
    SCROLL = enum.auto()
    VOLUME = enum.auto()
    DRAG = enum.auto()


class GestureController:
    def __init__(self, img_width, img_height, bound_rect=150, mouse_move_smoothening_factor=3, scroll_smoothening_factor=3, scroll_speed=500, scroll_threshold=15):
        self.img_width = img_width
        self.img_height = img_height
        self.bound_rect = bound_rect

        self.gesture_detector = GestureDetector(img_width, img_height)
        self.prev_gesture = None
        self.gesture_count = 0

        self.mouse_move_smoothening_factor = mouse_move_smoothening_factor
        self.prev_mouse_pos = Point()

        self.scroll_speed = scroll_speed
        self.scroll_threshold = scroll_threshold
        self.prev_scroll_y = None
        self.scroll_smoothening_factor = scroll_smoothening_factor

        self.mode = Mode.NORMAL

        self.switching_frames = 8

        self.prev_vol_y = None
        self.vol_threshold = 15

    def run(self, landmarks, handedness):
        curr_gesture = self.gesture_detector.find_gesture(landmarks, handedness)

        if self.mode == Mode.SWITCH_WIN:
            self.handle_switching_mode(curr_gesture)
            return

        if self.mode == Mode.DRAG:
            self.handle_drag_mode(curr_gesture, landmarks)
            return

        if self.mode == Mode.SCROLL:
            self.handle_scroll_mode(curr_gesture, landmarks)
            return

        if self.mode == Mode.MOVE_MOUSE:
            self.handle_move_mouse_mode(curr_gesture, landmarks)
            return

        if self.mode == Mode.VOLUME:
            self.handle_vol_mode(curr_gesture, landmarks)
            return

        if curr_gesture == self.prev_gesture:
            self.gesture_count += 1
        else:
            self.gesture_count = 0

        self.prev_gesture = curr_gesture

        if self.gesture_count != 1:
            return

        if curr_gesture == Gesture.INDEX_UP:
            self.mode = Mode.MOVE_MOUSE
            self.prev_mouse_pos = Point()

        elif curr_gesture == Gesture.V:
            thread = MouseClickThread(left_click=True)
            thread.start()

        elif curr_gesture == Gesture.FRIST2_THUMB:
            thread = DoubleClickThread()
            thread.start()

        elif curr_gesture == Gesture.FIRST_3:
            thread = MouseClickThread(left_click=False)
            thread.start()

        elif curr_gesture == Gesture.YO:
            self.mode = Mode.SWITCH_WIN

        elif curr_gesture == Gesture.FIST:
            self.mode = Mode.DRAG
            self.prev_mouse_pos = Point()

        elif curr_gesture == Gesture.THUMBS_UP:
            self.mode = Mode.SCROLL
            # when u scroll, and then in neutral gesture move hand to opp side then there will be sudden scroll if prev_scroll_y is not set to None
            self.prev_scroll_y = None

        elif curr_gesture == Gesture.L:
            self.mode = Mode.VOLUME
            self.prev_vol_y = None

    def get_finger_world_coords(self, finger, landmarks):
        coords = landmarks[fingers[finger.value]]
        return Point(int(coords.x * self.img_width), int(coords.y * self.img_height))

    def get_mouse_coords_from_landmark_coords_bound_rect(self, point):
        screen_width, screen_height = pyautogui.size()

        x = np.interp(point.x, (self.bound_rect, self.img_width - self.bound_rect), (0, screen_width))
        y = np.interp(point.y, (self.bound_rect, self.img_height - self.bound_rect), (0, screen_height))
        return Point(x, y)

    def get_mouse_coords_from_landmark_coords(self, point):
        screen_width, screen_height = pyautogui.size()

        x = np.interp(point.x, (0, self.img_width), (0, screen_width))
        y = np.interp(point.y, (0, self.img_height), (0, screen_height))
        return Point(x, y)

    def handle_switching_mode(self, curr_gesture):
        if curr_gesture != Gesture.YO:
            pyautogui.keyUp('alt')
            self.prev_gesture = curr_gesture
            self.mode = Mode.NORMAL
            self.gesture_count = 0
            return

        self.gesture_count += 1

        # first gesture will be detected and then switching mode will be activated and then again the gesture will be detected, so gesture_count == 2
        if self.gesture_count == 2:
            WindowSwitchThread(hold=True).start()
            # when gesture is first detected alt + tab will be pressed and one window would already be switched so for next switch start count from self.switching_frames not 2
            self.gesture_count = self.switching_frames

        elif self.gesture_count % self.switching_frames == 0:
            WindowSwitchThread(hold=False).start()

    def handle_drag_mode(self, curr_gesture, landmarks):
        if curr_gesture != Gesture.FIST:
            self.prev_gesture = curr_gesture
            self.mode = Mode.NORMAL
            self.gesture_count = 0
            self.prev_mouse_pos = Point()

            DragThread(False).start()
            return

        coords = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
        coords = Point(int(coords.x * self.img_width), int(coords.y * self.img_height))

        mouse_coords = self.get_mouse_coords_from_landmark_coords(coords)

        if self.prev_mouse_pos.x is None:
            self.prev_mouse_pos = Point(mouse_coords.x, mouse_coords.y)
            return

        prev_x, prev_y = self.prev_mouse_pos.x, self.prev_mouse_pos.y

        curr_x = prev_x + (mouse_coords.x - prev_x) / self.mouse_move_smoothening_factor
        curr_y = prev_y + (mouse_coords.y - prev_y) / self.mouse_move_smoothening_factor
        curr_mouse_pos = Point(curr_x, curr_y)
        self.prev_mouse_pos = curr_mouse_pos

        DragThread(True, Point(curr_x - prev_x, curr_y-prev_y)).start()

    def handle_scroll_mode(self, curr_gesture, landmarks):
        if curr_gesture != Gesture.THUMBS_UP:
            self.prev_gesture = curr_gesture
            self.mode = Mode.NORMAL
            self.prev_scroll_y = None
            self.gesture_count = 0
            return

        coords = landmarks[mp_hands.HandLandmark.THUMB_CMC.value]
        curr_y = int(coords.y * self.img_height)

        if self.prev_scroll_y is None:
            self.prev_scroll_y = curr_y
            return

        curr_y = self.get_smooth_value(curr_y, self.prev_scroll_y, self.scroll_smoothening_factor)

        scroll_by = int(np.interp(curr_y-self.prev_scroll_y, (-self.img_height/4, self.img_height/4), (-self.scroll_speed, self.scroll_speed)))

        if abs(scroll_by) < self.scroll_threshold:
            return

        thread = ScrollThread(scroll_by)
        thread.start()

        self.prev_scroll_y = curr_y

    def handle_vol_mode(self, curr_gesture, landmarks):
        if curr_gesture != Gesture.L:
            self.prev_gesture = curr_gesture
            self.mode = Mode.NORMAL
            self.prev_vol_y = None
            self.gesture_count = 0
            return

        coords = landmarks[mp_hands.HandLandmark.THUMB_CMC.value]
        curr_y = int(coords.y * self.img_height)

        if self.prev_vol_y is None:
            self.prev_vol_y = curr_y
            return

        delta_vol = int(np.interp(curr_y - self.prev_vol_y, (-self.img_height/4, self.img_height/4), (-self.scroll_speed, self.scroll_speed)))

        if abs(delta_vol) < self.vol_threshold:
            return

        thread = VolumeThread(delta_vol)
        thread.start()

        self.prev_vol_y = curr_y

    def handle_move_mouse_mode(self, curr_gesture, landmarks):
        if curr_gesture != Gesture.INDEX_UP:
            self.prev_gesture = curr_gesture
            self.mode = Mode.NORMAL
            self.gesture_count = 0
            self.prev_mouse_pos = Point()
            return

        coords = self.get_finger_world_coords(Finger.INDEX, landmarks)
        mouse_coords = self.get_mouse_coords_from_landmark_coords_bound_rect(coords)

        if self.prev_mouse_pos.x is None:
            self.prev_mouse_pos = Point(mouse_coords.x, mouse_coords.y)
            return

        prev_x, prev_y = self.prev_mouse_pos.x, self.prev_mouse_pos.y

        curr_x = self.get_smooth_value(mouse_coords.x, prev_x, self.mouse_move_smoothening_factor)
        curr_y = self.get_smooth_value(mouse_coords.y, prev_y, self.mouse_move_smoothening_factor)

        curr_mouse_pos = Point(curr_x, curr_y)
        self.prev_mouse_pos = curr_mouse_pos

        thread = MouseMoveThread(curr_mouse_pos)
        thread.start()

    def get_smooth_value(self, curr, prev, smoothening_factor):
        smooth_value = prev + (curr - prev) / smoothening_factor
        return smooth_value
