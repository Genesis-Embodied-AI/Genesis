from typing import Union, Literal

import genesis as gs


EVENT_HANDLE_STATE = Union[Literal[True], None]

# Note: Viewer window is based on pyglet.window.Window, mouse events are defined in pyglet.window.BaseWindow

class ViewerInteractionBase():
    """Base class for handling pyglet.window.Window events.
    """

    log_events: bool

    def __init__(self, log_events: bool = False):
        self.log_events = log_events

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        if self.log_events:
            gs.logger.info(f"Mouse moved to {x}, {y}")

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events:
            gs.logger.info(f"Mouse dragged to {x}, {y}")

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events:
            gs.logger.info(f"Mouse buttons {button} pressed at {x}, {y}")

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events:
            gs.logger.info(f"Mouse buttons {button} released at {x}, {y}")

    def on_key_press(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events:
            gs.logger.info(f"Key pressed: {chr(symbol)}")

    def on_key_release(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self.log_events:
            gs.logger.info(f"Key released: {chr(symbol)}")

    def on_draw(self) -> None:
        pass
