import os
import socket
from argparse import ArgumentParser, Namespace

DEFAULT_ICE_SERVERS = ("stun:stun.l.google.com:19302",)


def add_streamer_cli_args(
    parser: ArgumentParser,
    *,
    default_host: str = "0.0.0.0",
    default_port: int = 8000,
    default_fps: int = 30,
    default_video_bitrate_mbps: float = 8.0,
) -> None:
    parser.add_argument("--host", default=default_host, help="HTTP signaling bind host.")
    parser.add_argument("--port", type=int, default=default_port, help="HTTP signaling bind port.")
    parser.add_argument("--fps", type=int, default=default_fps, help="Target camera streaming FPS.")
    parser.add_argument(
        "--video-bitrate-mbps",
        type=float,
        default=default_video_bitrate_mbps,
        help="Target max outgoing video bitrate in Mbps. Use <=0 to disable explicit bitrate cap.",
    )
    parser.add_argument(
        "--ice-server",
        action="append",
        default=[],
        help="Repeat to add ICE servers, e.g. --ice-server stun:stun.l.google.com:19302",
    )
    parser.add_argument(
        "--no-default-stun",
        action="store_true",
        help="Disable fallback default STUN server when no ICE servers are provided.",
    )
    parser.add_argument("--token", default=None, help="Optional shared token for simple access control.")
    parser.add_argument(
        "--public-host",
        default=None,
        help="Host/IP shown in printed browser URL. Defaults to detected non-loopback IP for 0.0.0.0/::.",
    )


def collect_ice_servers_from_args(args: Namespace) -> list[str]:
    servers = [server.strip() for server in args.ice_server if server.strip()]
    env_servers = os.environ.get("GS_WEBRTC_ICE_SERVERS", "")
    if env_servers:
        servers.extend(server.strip() for server in env_servers.split(",") if server.strip())
    if not servers and not args.no_default_stun:
        servers.extend(DEFAULT_ICE_SERVERS)
    return servers


def resolve_stream_token(args: Namespace) -> str | None:
    return args.token or os.environ.get("GS_WEBRTC_TOKEN")


def _detect_non_loopback_ipv4() -> str | None:
    # UDP connect does not send packets, but lets us discover the default outbound interface.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidate = sock.getsockname()[0]
            if candidate and not candidate.startswith("127."):
                return candidate
    except OSError:
        pass

    try:
        _, _, candidates = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        return None

    for candidate in candidates:
        if candidate and not candidate.startswith("127."):
            return candidate
    return None


def build_stream_url(host: str, port: int, token: str | None = None, public_host: str | None = None) -> str:
    """Build browser URL shown in CLI output for the signaling server.

    If `public_host` is omitted and the bind host is wildcard (`0.0.0.0`/`::`),
    the function attempts to use a non-loopback IPv4 address for easier remote
    access, then falls back to `127.0.0.1`.
    """
    if public_host is None:
        if host in {"0.0.0.0", "::"}:
            public_host = _detect_non_loopback_ipv4() or "127.0.0.1"
        else:
            public_host = host
    url = f"http://{public_host}:{port}"
    if token:
        url = f"{url}?token={token}"
    return url


def video_bitrate_bps_from_args(args: Namespace) -> int | None:
    if args.video_bitrate_mbps <= 0:
        return None
    return int(args.video_bitrate_mbps * 1_000_000)
