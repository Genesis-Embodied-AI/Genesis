# import genesis as gs
from typing import override
from pyglet.event import EVENT_HANDLE_STATE
from genesis.engine.scene import Scene


# Notes:
# Viewer window is based on pyglet.window.Window, mouse events are defined in pyglet.window.BaseWindow

class ViewerInteractionBase():
    """Base class for handling pyglet.window.Window events.
    """

    log_events: bool

    def __init__(self, log_events: bool = False):
        self.log_events = log_events

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        if self.log_events: print(f"Mouse moved to {x}, {y}")

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events: print(f"Mouse dragged to {x}, {y}")

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events: print(f"Mouse buttons {button} pressed at {x}, {y}")

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events: print(f"Mouse buttons {button} released at {x}, {y}")

    def on_key_press(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events: print(f"Key pressed: {chr(symbol)}")

    def on_key_release(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events: print(f"Key released: {chr(symbol)}")
    

class ViewerInteraction(ViewerInteractionBase):
    """Functionalities to be implemented:
    - mouse picking
    - mouse dragging
    """

    scene: Scene

    def __init__(self, scene: Scene, log_events: bool = False):
        super().__init__(log_events)
        self.scene = scene

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        return super().on_mouse_press(x, y, button, modifiers)

