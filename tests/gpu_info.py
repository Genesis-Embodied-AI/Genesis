"""
Cross-vendor GPU information module.
Provides abstract interface for querying GPU device information.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import warnings
import re
import shutil


class GpuBackend(ABC):
    """Abstract base class for GPU backend implementations."""

    @abstractmethod
    def get_device_count(self) -> int:
        """Get the number of available GPU devices."""
        pass

    @abstractmethod
    def get_device_vram_mib(self) -> tuple[int, ...]:
        """Get VRAM in MiB for each GPU device."""
        pass

    @abstractmethod
    def get_device_index_from_uuid(self, device_uuid: str) -> int:
        """Get device index from UUID."""
        pass

    @abstractmethod
    def get_per_process_vram_mib(self) -> dict[int, int]:
        """Get per-process VRAM usage in MiB."""
        pass

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Report if the backend is available"""
        pass


class NvidiaBackend(GpuBackend):
    """NVIDIA GPU backend implementation."""

    NVIDIA_GPU_INTERFACE_PATH = Path("/proc/driver/nvidia/gpus/")
    NVIDIA_SMI = "nvidia-smi"

    UUID_RE_STR = "-".join((r"(?:[A-Fa-f0-9\-]{" f"{nr}" "})" for nr in (8, 4, 4, 4, 12)))

    def __init__(self):
        # allow to later warn the user something is off and per-process memory monitoring won't
        # supported.
        self.is_nvidia_gpu_interface_populated = self._check_nvidia_gpu_interface_populated()

    @classmethod
    def _check_nvidia_gpu_interface_populated(cls):
        return cls.NVIDIA_GPU_INTERFACE_PATH.is_dir() and bool(list(cls.NVIDIA_GPU_INTERFACE_PATH.iterdir()))

    @classmethod
    def is_available(cls) -> bool:
        nvidia_smi_avail = shutil.which(cls.NVIDIA_SMI) is not None

        if nvidia_smi_avail and not cls._check_nvidia_gpu_interface_populated():
            warnings.warn(
                f"{cls.NVIDIA_GPU_INTERFACE_PATH!s} is not populated. Fallback to using nvidia-smi to discover GPUs. "
                "If you're running your app in a container, beware that per-process monitoring fails as the nvidia driver "
                "lacks support for process namespace."
            )

        return nvidia_smi_avail

    def get_device_count(self) -> int:
        """Get NVIDIA GPU count from proc interface."""
        return (
            (self.NVIDIA_GPU_INTERFACE_PATH.is_dir() and len(list(self.NVIDIA_GPU_INTERFACE_PATH.iterdir())))
            or
            # fallback to nvidia-smi
            len(self._get_nvidia_smi_list_gpus())
        )

    @staticmethod
    def _parse_nvidia_smi_query_memory(output: str) -> tuple[int, ...]:
        vram_values = []
        for line in output.splitlines():
            line = line.strip()
            if line:
                # Convert from MB to MiB (they're the same for this purpose)
                vram_values.append(int(line))

        return tuple(vram_values)

    @classmethod
    def _call_nvidia_smi(cls, args: list[str] = []) -> str:
        return subprocess.check_output(
            [cls.NVIDIA_SMI] + args,
            encoding="utf-8",
        )

    def get_device_vram_mib(self) -> tuple[int, ...]:
        """Get VRAM in MiB for each NVIDIA GPU using nvidia-smi."""

        output = self._call_nvidia_smi(
            [
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ]
        )

        per_process_vram_mib = self._parse_nvidia_smi_query_memory(output)

        if not per_process_vram_mib and not self.is_nvidia_gpu_interface_populated:
            warnings.warn(
                "No memory usage per process returned by nvidia-smi. "
                "Your configuration seems to be broken as Nvidia GPU /proc interface is not populated, "
                "indicating an abnormal driver setup. "
                "Querying per-process metrics , especially within a container or a Kuberneted pod, "
                "is likely to return nothing (nvidia driver does not support process namespaces)."
            )

        return per_process_vram_mib

    @classmethod
    def _parse_nvidia_smi_list_gpus(cls, output: str) -> dict[int, str]:
        device_idx_to_uuid = {}
        for l in output.splitlines():
            m = re.match(
                rf"GPU (\d*): .*\(UUID: GPU-({cls.UUID_RE_STR})\)",
                l,
            )

            assert m is not None, "nvidia-smi has changed its format"

            device_idx = int(m.group(1))
            device_uuid = m.group(2)

            device_idx_to_uuid[device_idx] = device_uuid

        return device_idx_to_uuid

    def _get_nvidia_smi_list_gpus(self) -> dict[int, str]:
        # fallback to nvidia-smi
        output = self._call_nvidia_smi(["--list-gpus"])

        return self._parse_nvidia_smi_list_gpus(output)

    @classmethod
    def _extract_uuid_from_nvidia_proc_information(cls, device_info: str) -> str | None:
        m = re.match(rf"GPU UUID:\s+GPU-({cls.UUID_RE_STR})", device_info)

        return m.group(1) if m is not None else None

    def _get_proc_nvidia_list_gpus(self) -> dict[int, str]:
        device_idx_to_uuid = {}
        for device_idx, device_path in enumerate(self.NVIDIA_GPU_INTERFACE_PATH.iterdir()):
            device_info_path = device_path / "information"
            if not device_info_path.exists():
                continue

            with device_info_path.open() as f:
                device_info = f.read()

            device_uuid = self._extract_uuid_from_nvidia_proc_information(device_info)
            assert device_uuid is not None, "/proc/driver/nvidia/gpus/<idx>/information has changed its format"
            device_idx_to_uuid[device_idx] = device_uuid

        return device_idx_to_uuid

    def get_device_index_from_uuid(self, device_uuid: str) -> int:
        """Get NVIDIA device index from UUID."""

        for idx_to_uuid_fn in (
            self._get_proc_nvidia_list_gpus,
            self._get_nvidia_smi_list_gpus,  # fallback to nvidia-smi
        ):
            map_idx_to_uuid = idx_to_uuid_fn()

            device_idx = next(
                (map_idx for map_idx, map_uuid in map_idx_to_uuid.items() if device_uuid == map_uuid), None
            )

            if device_idx is not None:
                return device_idx

        return -1

    def get_per_process_vram_mib(self) -> dict[int, int]:
        """Get per-process VRAM usage for NVIDIA GPUs."""
        output = self._call_nvidia_smi()

        return self._parse_nvidia_smi_output(output)

    def _parse_nvidia_smi_output(self, output: str) -> dict[int, int]:
        """Parse nvidia-smi output for per-process memory usage."""
        section = 0
        subsec = 0
        res = {}
        for line in output.split("\n"):
            if line.startswith("|==========="):
                section += 1
                subsec = 0
                continue
            if line.startswith("+-------"):
                subsec += 1
                continue
            if section == 2 and subsec == 0:
                if "No running processes" in line:
                    continue
                split_line = line.split()
                if len(split_line) >= 5:
                    pid = int(split_line[4])
                    mem = int(split_line[-2].split("MiB")[0])
                    res[pid] = mem
        return res


class AmdBackend(GpuBackend):
    """AMD GPU backend implementation."""

    KFD_SYSFS_PATH = Path("/sys/devices/virtual/kfd/kfd/topology")
    ROCM_SMI = "rocm-smi"

    @classmethod
    def is_available(cls) -> bool:
        return cls.KFD_SYSFS_PATH.is_dir() and (shutil.which(cls.ROCM_SMI) is not None)

    @classmethod
    def _call_rocm_smi(cls, args: list[str] = []) -> str:
        return subprocess.check_output(
            [cls.ROCM_SMI] + args,
            encoding="utf-8",
        )

    def _get_kfd_gpu_nodes_properties(self) -> dict[int, dict[str, int]]:
        """Get KFD GPU node properties."""
        kfd_sysfs_path_nodes = self.KFD_SYSFS_PATH / "nodes"
        gpu_nodes_properties = {}

        if not kfd_sysfs_path_nodes.is_dir():
            return {}

        for node_path in kfd_sysfs_path_nodes.iterdir():
            with (node_path / "properties").open() as node_properties_f:
                properties_str = node_properties_f.read()
                node_props = self._parse_kfd_node_properties(properties_str)

                if node_props["cpu_cores_count"] == 0:
                    gpu_nodes_properties[int(node_path.name)] = node_props

        return gpu_nodes_properties

    @staticmethod
    def _parse_kfd_node_properties(kfd_properties_str: str) -> dict[str, int]:
        """Parse KFD node properties string."""
        props = {}
        for line in kfd_properties_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            name, value_str = line.split()
            props[name] = int(value_str)

        return props

    def get_device_count(self) -> int:
        """Get AMD GPU count from KFD sysfs."""
        gpu_nodes = self._get_kfd_gpu_nodes_properties()
        return len(gpu_nodes)

    @staticmethod
    def _parse_rocm_smi_showmeminfo(output: str) -> tuple[int, ...]:
        return tuple(int(m.group(1)) for m in re.finditer(r"VRAM Total:\s+(\d+)\s*MiB", output))

    def get_device_vram_mib(self) -> tuple[int, ...]:
        """Get VRAM in MiB for each AMD GPU using rocm-smi."""
        output = self._call_rocm_smi(["--showmeminfo", "vram", "-d", "0-255"])

        # Parse rocm-smi output for VRAM info
        return self._parse_rocm_smi_showmeminfo(output)

    def get_device_index_from_uuid(self, device_uuid: str) -> int:
        """Get AMD device index from UUID."""
        device_uuid = device_uuid.replace("-", "")
        hip_uuid = "".join([chr(int(device_uuid[i : i + 2], 16)) for i in range(0, len(device_uuid), 2)])
        unique_id = int(hip_uuid, 16)

        gpu_nodes_properties = self._get_kfd_gpu_nodes_properties()

        for node_rank, gpu_props in enumerate(gpu_nodes_properties.values()):
            if gpu_props.get("unique_id") == unique_id:
                return node_rank

        return -1

    def get_per_process_vram_mib(self) -> dict[int, int]:
        """Get per-process VRAM usage for AMD GPUs."""
        output = self._call_rocm_smi(["--showpids"])

        return self._parse_rocm_smi_showpids(output)

    @staticmethod
    def _parse_rocm_smi_showpids(output: str) -> dict[int, int]:
        """Parse rocm-smi output for per-process memory usage."""
        NO_PIDS = "No KFD PIDs currently running"

        if NO_PIDS in output:
            return {}

        PID = "PID"
        VRAM_USED = "VRAM USED"
        TABLE_BORDER_STR = "===="

        pid_idx = output.find(PID)
        if pid_idx == -1:
            return {}

        output = output[pid_idx:]
        lines = output.split("\n")
        if not lines:
            return {}

        header = lines[0]

        # Extract column spans from header
        header_to_span = {}
        for header_m in re.finditer(r"(\S+( |$))+ *", header):
            header_name = header_m.group(0).strip()
            header_to_span[header_name] = header_m.span()

        if PID not in header_to_span or VRAM_USED not in header_to_span:
            return {}

        pid_to_mem_use = {}
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith(TABLE_BORDER_STR):
                continue

            header_to_info_str = {h: line[start:end].strip() for h, (start, end) in header_to_span.items()}

            pid = int(header_to_info_str[PID])
            mem_bytes = int(header_to_info_str[VRAM_USED])
            # Convert bytes to MiB
            pid_to_mem_use[pid] = mem_bytes >> 20

        return pid_to_mem_use


def detect_gpu_backend() -> GpuBackend | None:
    """
    Detect available GPU backend.

    Returns:
        NvidiaBackend if NVIDIA GPUs are detected,
        AmdBackend if AMD GPUs are detected,
        None otherwise.
    """

    Backend_candidates = [
        NvidiaBackend,
        AmdBackend,
    ]

    available_Backends = [Backend for Backend in Backend_candidates if Backend.is_available()]

    if not available_Backends:
        return None

    if len(available_Backends) > 2:
        warnings.warn("Multiple backends were detected on the current system.")

    Backend = available_Backends[0]

    return Backend()
