"""The core claim, checked on the SAVED files: with --gap 0 --edge 0 the parts join to the original exactly (drawing and photo,
straight and zig-zag). Plus the guard rails: RGBA, 16 bit, stale parts, bad input."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def cli(*args, check=True):
    return subprocess.run([sys.executable, str(ROOT / "pixelcut.py"), *map(str, args)], capture_output=True, text=True, check=check)


def rgb(path):
    return np.array(Image.open(path).convert("RGB")).astype(int)


@pytest.mark.parametrize("image,extra", [
    ("demo.png", ["--angles", "32,148,214,327"]),
    ("demo.png", ["--angles", "32,148,214,327", "--depth", "0"]),
    ("photo_cat.png", ["--opaque"]),
    ("photo_cat.png", ["--opaque", "--angles", "20,110,200,290", "--tooth", "11", "--depth", "5"]),
])
def test_roundtrip_is_exact(tmp_path, image, extra):
    cli("split", EX / image, tmp_path / "p", "--gap", "0", "--edge", "0", *extra)
    cli("join", tmp_path / "p", tmp_path / "back.png")
    assert np.abs(rgb(EX / image) - rgb(tmp_path / "back.png")).max() == 0


def test_alpha_over_white(tmp_path):
    out = cli("alpha", EX / "demo.png", tmp_path / "a.png", "--verify").stdout
    assert float(out.split(":")[1].split("/")[0]) <= 1.0


def test_default_slit_is_visible_and_reported(tmp_path):
    out = cli("split", EX / "demo.png", tmp_path, "--verify").stdout
    assert "saved files joined over white" in out
    assert float(out.split("saved files joined over white:")[1].split("/")[0]) > 1.0     # the slit changes pixels on purpose


def test_stale_parts_are_removed(tmp_path):
    cli("split", EX / "demo.png", tmp_path, "--gap", "0", "--edge", "0")                  # 4 parts
    cli("split", EX / "demo.png", tmp_path, "--angles", "90,270", "--gap", "0", "--edge", "0", "--depth", "0")   # 2 parts
    assert len(list(tmp_path.glob("part_*.png"))) == 2
    cli("join", tmp_path, tmp_path / "back.png")
    assert np.abs(rgb(EX / "demo.png") - rgb(tmp_path / "back.png")).max() == 0


def test_rgba_is_composited_over_white(tmp_path):
    Image.fromarray(np.zeros((40, 40, 4), np.uint8), "RGBA").save(tmp_path / "t.png")      # fully transparent
    r = cli("compare", tmp_path / "t.png", tmp_path / "t.png")
    assert "transparency" in r.stderr


def test_16bit_is_announced(tmp_path):
    Image.fromarray((np.arange(1600).reshape(40, 40) * 40).astype(np.uint16)).save(tmp_path / "g.png")
    r = cli("compare", tmp_path / "g.png", tmp_path / "g.png")
    assert "16" in r.stderr or "more than 8 bit" in r.stderr


@pytest.mark.parametrize("args", [
    ["split", "WHITE", "o"],
    ["split", "MISSING.png", "o"],
    ["split", "EX", "o", "--angles", "a,b"],
    ["split", "EX", "o", "--angles", "90"],
    ["split", "EX", "o", "--centre", "5"],
    ["split", "EX", "o", "--tooth", "0"],
])
def test_bad_input_gives_one_line(tmp_path, args):
    Image.new("RGB", (50, 50), "white").save(tmp_path / "w.png")
    args = [str(EX / "demo.png") if x == "EX" else str(tmp_path / "w.png") if x == "WHITE" else str(tmp_path / x) if x in ("MISSING.png", "o") else x for x in args]
    r = cli(*args, check=False)
    assert r.returncode != 0 and "Traceback" not in r.stderr


def test_compare_size_mismatch_and_tiny(tmp_path):
    Image.new("RGB", (8, 8), "red").save(tmp_path / "a.png")
    Image.new("RGB", (16, 16), "red").save(tmp_path / "b.png")
    assert cli("compare", tmp_path / "a.png", tmp_path / "b.png", check=False).returncode != 0
    assert cli("compare", tmp_path / "a.png", tmp_path / "b.png", "--resize").returncode == 0
    Image.new("RGB", (1, 1), "red").save(tmp_path / "c.png")
    assert cli("compare", tmp_path / "c.png", tmp_path / "c.png").returncode == 0
