import subprocess
import tempfile
from pathlib import Path

from data.prepare import main as prepare_main


def test_prepare_throughput(tmp_path: Path):
    # create synthetic 64MB raw data
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    text = ("hello world\n" * 1024)  # ~12 KB
    repeat = (64 * 1024 * 1024) // len(text)  # ~64MB
    (raw_dir / "data.txt").write_text(text * repeat)

    out_file = tmp_path / "out.parquet"

    # run via subprocess to capture exit code like CLI
    res = subprocess.run([
        "python3",
        str(Path(__file__).parents[1] / "data/prepare.py"),
        "--input",
        str(raw_dir),
        "--output",
        str(out_file),
    ])
    assert res.returncode == 0, "prepare.py failed performance threshold"
    assert out_file.exists()
