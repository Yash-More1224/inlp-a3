from __future__ import annotations

from pathlib import Path


def ensure_output_dirs(base_dir: str = "outputs") -> dict[str, str]:
    base = Path(base_dir)
    logs_dir = base / "logs"
    plots_dir = base / "plots"
    results_dir = base / "results"

    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base": str(base),
        "logs": str(logs_dir),
        "plots": str(plots_dir),
        "results": str(results_dir),
    }


def write_text(path: str, text: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
