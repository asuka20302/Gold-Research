"""Regenerate model outputs and QMT-compatible gold price signals."""

from pathlib import Path
import subprocess
import sys


QMT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QMT_DIR.parent


def run(script_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR,
        check=True,
    )


def main() -> None:
    run(PROJECT_DIR / "gold_model_comparison.py")
    run(QMT_DIR / "export_qmt_signals.py")
    print(
        "\nQMT inputs are ready:\n"
        f"  {PROJECT_DIR / 'outputs' / 'qmt_gold_signals.csv'}\n"
        f"  {QMT_DIR / 'qmt_gold_strategy.py'}"
    )


if __name__ == "__main__":
    main()
