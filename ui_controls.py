import threading

import pyautogui

from point import Point


class ScrollThread(threading.Thread):
    def __init__(self, scroll_by):
        threading.Thread.__init__(self)
        self.scroll_by = scroll_by

    def run(self):
        pyautogui.scroll(self.scroll_by)


class MouseClickThread(threading.Thread):
    def __init__(self, left_click):
        threading.Thread.__init__(self)
        self.left_click = left_click

    def run(self):
        if self.left_click:
            pyautogui.click()
        else:
            pyautogui.click(button=pyautogui.SECONDARY)


class WindowSwitchThread(threading.Thread):
    def __init__(self, hold):
        threading.Thread.__init__(self)
        self.hold = hold

    def run(self):
        if self.hold:
            pyautogui.keyDown('alt')
            pyautogui.press('tab')
        else:
            pyautogui.press('tab')


class DoubleClickThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        pyautogui.doubleClick()


class MouseMoveThread(threading.Thread):
    def __init__(self, point):
        threading.Thread.__init__(self)
        self.point = point

    def run(self):
        pyautogui.moveTo(self.point.x, self.point.y)


class DragThread(threading.Thread):
    def __init__(self, start_drag, coords=Point()):
        threading.Thread.__init__(self)
        self.start_drag = start_drag
        self.coords = coords

    def run(self):
        if self.start_drag:
            pyautogui.mouseDown(button=pyautogui.PRIMARY)
            pyautogui.moveRel(self.coords.x, self.coords.y)
            # pyautogui.moveTo(self.coords.x, self.coords.y)
        else:
            pyautogui.mouseUp(button=pyautogui.PRIMARY)


class VolumeThread(threading.Thread):
    def __init__(self, delta_vol):
        threading.Thread.__init__(self)
        self.delta_vol = delta_vol

    def run(self):
        if self.delta_vol == 0:
            return

        if self.delta_vol < 0:
            pyautogui.press('volumeup')
        else:
            pyautogui.press('volumedown')
