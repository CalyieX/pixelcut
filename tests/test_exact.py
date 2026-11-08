"""The parts must add up to the original over white (max deviation <= 1/255), for a drawing and for a real photo."""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "pixelcut.py"), *map(str, args)], capture_output=True, text=True, check=True).stdout


def deviation(out):
    return float(out.split("without slit:")[1].split("/")[0])


def test_drawing(tmp_path):
    out = run("split", ROOT / "examples" / "demo.png", tmp_path, "--angles", "32,148,214,327", "--only-inside", "--verify")
    assert deviation(out) <= 1.0


def test_photo(tmp_path):
    out = run("split", ROOT / "examples" / "photo_cat.png", tmp_path, "--opaque", "--verify")
    assert deviation(out) <= 1.0
    assert len(list(tmp_path.glob("part_*.png"))) == 4
    assert np.array(Image.open(tmp_path / "part_0.png")).shape[2] == 4


def test_join_rebuilds_photo(tmp_path):
    run("split", ROOT / "examples" / "photo_cat.png", tmp_path / "p", "--opaque", "--gap", "0", "--edge", "0")
    run("join", tmp_path / "p", tmp_path / "back.png")
    a = np.array(Image.open(ROOT / "examples" / "photo_cat.png").convert("RGB")).astype(int)
    b = np.array(Image.open(tmp_path / "back.png").convert("RGB")).astype(int)
    assert np.abs(a - b).max() == 0
