"""Hardware detection and profile selection for CABINGUARD-ADI."""
from __future__ import annotations
import platform
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_hardware() -> dict:
    """Detect and return hardware capabilities."""
    import psutil
    import torch

    info = {
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "disk_free_gb": round(psutil.disk_usage(Path.cwd().anchor).free / 1e9, 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i,
                "name": props.name,
                "vram_gb": round(props.total_memory / 1e9, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    return info


def auto_select_profile(hw: dict) -> str:
    """Select training profile based on detected hardware."""
    if not hw["cuda_available"] or hw["gpu_count"] == 0:
        return "cpu"
    if hw["gpu_count"] > 1:
        return "multi_gpu"
    vram = hw["gpus"][0]["vram_gb"] if hw["gpus"] else 0
    if vram >= 20:
        return "single_gpu_24gb"
    elif vram >= 14:
        return "single_gpu_16gb"
    else:
        return "single_gpu_8gb"


def print_hardware_report(hw: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  CABINGUARD-ADI Hardware Report")
    print(sep)
    print(f"  Python        : {hw['python_version']}")
    print(f"  OS            : {hw['os']}")
    print(f"  CPU (logical) : {hw['cpu_count_logical']} cores")
    print(f"  RAM           : {hw['ram_gb']} GB")
    print(f"  Disk free     : {hw['disk_free_gb']} GB")
    print(f"  PyTorch       : {hw['torch_version']}")
    print(f"  CUDA          : {hw['cuda_version']}")
    print(f"  GPU count     : {hw['gpu_count']}")
    for g in hw["gpus"]:
        print(f"  GPU {g['index']}         : {g['name']} — {g['vram_gb']} GB VRAM")
    print(f"  Auto profile  : {auto_select_profile(hw)}")
    print(sep + "\n")


def save_environment_report(hw: dict, output_path: str) -> None:
    """Save environment info as JSON for reproducibility."""
    import sys, subprocess
    report = dict(hw)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30
        )
        report["pip_freeze"] = result.stdout.strip().split("\n")
    except Exception:
        report["pip_freeze"] = ["NOT_AVAILABLE"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Environment report saved to {output_path}")

