from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "GenesisCameraVideoTrack",
    "WebRTCStreamer",
    "add_streamer_cli_args",
    "build_stream_url",
    "collect_ice_servers_from_args",
    "resolve_stream_token",
    "video_bitrate_bps_from_args",
]


def __getattr__(name: str) -> Any:
    if name in ("GenesisCameraVideoTrack", "WebRTCStreamer"):
        module = import_module(".webrtc_aiortc", __name__)
        return getattr(module, name)
    if name in (
        "add_streamer_cli_args",
        "build_stream_url",
        "collect_ice_servers_from_args",
        "resolve_stream_token",
        "video_bitrate_bps_from_args",
    ):
        module = import_module(".common", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if TYPE_CHECKING:
    from .common import (
        add_streamer_cli_args,
        build_stream_url,
        collect_ice_servers_from_args,
        resolve_stream_token,
        video_bitrate_bps_from_args,
    )
    from .webrtc_aiortc import GenesisCameraVideoTrack, WebRTCStreamer
