import enum


class Command(enum.Enum):
    MOVE_MOUSE = enum.auto()
    SWITCH_WIN = enum.auto()
    SCROLL = enum.auto()
    VOLUME = enum.auto()
    DRAG = enum.auto()
    LEFT_CLICK = enum.auto()
    RIGHT_CLICK = enum.auto()
    DOUBLE_CLICK = enum.auto()
