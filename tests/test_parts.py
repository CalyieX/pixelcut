"""Beyond 'the parts add up': the pieces themselves must be right (polygons, angles, centre, sizes), and the other commands
must really do their job. Each test here kills a mutation that the sum-over-white check cannot see."""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def cli(*args, check=True):
    return subprocess.run([sys.executable, str(ROOT / "pixelcut.py"), *map(str, args)], capture_output=True, text=True, check=check)


def fails_cleanly(*args):
    r = cli(*args, check=False)
    assert r.returncode != 0, "expected failure: %s" % (args,)
    assert "Traceback" not in r.stderr, r.stderr
    assert r.stderr.strip(), "failure without a message"
    return r.stderr


def alpha_of(path):
    return np.array(Image.open(path).convert("RGBA"))[:, :, 3]


@pytest.fixture
def photo_split(tmp_path):
    """photo, 4 uneven sectors, no slit, straight lines: every pixel belongs to exactly one part."""
    out = tmp_path / "p"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "20,110,200,290", "--depth", "0", "--gap", "0", "--edge", "0",
        "--centre", "225,150")
    return out, json.loads((out / "parts.json").read_text())


def test_parts_tile_the_image_without_overlap(photo_split):
    out, m = photo_split
    covered = sum((alpha_of(out / f) > 0).astype(int) for f in m["files"])
    assert covered.min() == 1 and covered.max() == 1            # each pixel in exactly one part


def test_manifest_size_and_centre(photo_split):
    out, m = photo_split
    h, w = alpha_of(out / "part_0.png").shape
    assert m["size"] == [w, h]
    assert m["centre"] == [225.0, 150.0]
    assert m["files"] == ["part_0.png", "part_1.png", "part_2.png", "part_3.png"]


def test_starts_at_angle_matches_geometry(photo_split):
    out, m = photo_split
    cx, cy = m["centre"]
    for part in m["parts"]:
        a = math.radians(part["starts_at_angle"] + 3)           # 3 degrees after the dividing line lies inside this part
        x, y = int(round(cx + 80 * math.cos(a))), int(round(cy + 80 * math.sin(a)))
        owners = [k for k in range(4) if alpha_of(out / ("part_%d.png" % k))[y, x] > 0]
        assert owners == [part["part"]], (part["part"], owners)


def test_polygons_describe_their_own_part(photo_split):
    out, m = photo_split
    h, w = alpha_of(out / "part_0.png").shape
    for part in m["parts"]:
        solid = alpha_of(out / ("part_%d.png" % part["part"])) > 0
        filled = np.zeros((h, w), np.uint8)
        cv2.fillPoly(filled, [np.array(part["polygon"], np.int32)], 1)
        iou = (filled.astype(bool) & solid).sum() / (filled.astype(bool) | solid).sum()
        assert iou > 0.97, (part["part"], iou)                  # a polygon of another sector would be ~0


def test_fattest_point_is_inside_and_deepest(photo_split):
    out, m = photo_split
    for part in m["parts"]:
        solid = (alpha_of(out / ("part_%d.png" % part["part"])) > 8).astype(np.uint8)
        dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
        x, y = part["fattest_point"]
        assert solid[y, x] == 1
        assert dist[y, x] >= dist.max() - 1.0


def test_zigzag_changes_the_cut_but_not_the_ownership_far_from_the_lines(tmp_path):
    straight, zig = tmp_path / "s", tmp_path / "z"
    for o, depth in ((straight, "0"), (zig, "9")):
        cli("split", EX / "photo_cat.png", o, "--opaque", "--angles", "20,110,200,290", "--depth", depth, "--tooth", "14",
            "--gap", "0", "--edge", "0", "--centre", "225,150")
    a, b = alpha_of(straight / "part_0.png") > 0, alpha_of(zig / "part_0.png") > 0
    assert (a != b).sum() > 200                                  # the teeth really cut differently ...
    assert (a != b).sum() < 0.1 * a.size                         # ... but only along the lines


def test_zigzag_direction_is_right(tmp_path):
    """At along = 7.5 teeth the first dividing line (20 degrees) has swung +depth to the clockwise side, so a point 4.5 px past
    the straight line still belongs to the previous part (3), while with straight lines it belongs to part 0."""
    cx, cy, a = 225.0, 150.0, math.radians(20)
    x = cx + 105 * math.cos(a) - 4.5 * math.sin(a)
    y = cy + 105 * math.sin(a) + 4.5 * math.cos(a)
    owner = {}
    for name, depth in (("zig", "9"), ("straight", "0")):
        out = tmp_path / name
        cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "20,110,200,290", "--depth", depth, "--tooth", "14",
            "--gap", "0", "--edge", "0", "--centre", "225,150")
        owner[name] = [k for k in range(4) if alpha_of(out / ("part_%d.png" % k))[int(round(y)), int(round(x))] > 0]
    assert owner["straight"] == [0]
    assert owner["zig"] == [3]


def test_polygon_with_hole_and_island(tmp_path):
    im = np.full((200, 300, 3), 255, np.uint8)
    cv2.circle(im, (60, 100), 40, (200, 30, 30), -1)
    cv2.circle(im, (60, 100), 15, (255, 255, 255), -1)           # hole inside the ring
    cv2.circle(im, (120, 30), 8, (30, 30, 200), -1)              # separate island in the same (left) part
    Image.fromarray(im).save(tmp_path / "ring.png")
    cli("split", tmp_path / "ring.png", tmp_path / "o", "--angles", "90,270", "--depth", "0", "--gap", "0", "--edge", "0",
        "--centre", "200,100")
    m = json.loads((tmp_path / "o" / "parts.json").read_text())
    left = max(m["parts"], key=lambda p: len(p["polygons"]))
    assert len(left["polygons"]) == 2                            # ring and island, not only the largest outline
    assert sum(len(poly["holes"]) for poly in left["polygons"]) == 1
    assert len(left["polygon"]) > 0                              # the old single-outline field stays


def test_only_inside_keeps_satellite_whole(tmp_path):
    im = np.full((240, 400, 3), 255, np.uint8)
    cv2.circle(im, (140, 120), 90, (200, 30, 30), -1)
    cv2.circle(im, (330, 120), 14, (30, 30, 200), -1)            # satellite on the line y = 120 (angle 0 from the centre)
    Image.fromarray(im).save(tmp_path / "sat.png")
    cli("split", tmp_path / "sat.png", tmp_path / "o", "--angles", "0,180", "--depth", "0", "--gap", "0", "--edge", "0",
        "--centre", "140,120", "--only-inside")
    owners = 0
    for k in range(2):
        a = alpha_of(tmp_path / "o" / ("part_%d.png" % k))
        if (a[106:135, 316:345] > 0).any():
            owners += 1
    assert owners == 1                                            # the satellite is not cut in two


# --------------------------------------------------------------------------- compare
def test_compare_reports_real_numbers(tmp_path):
    a = np.full((40, 40, 3), 100, np.uint8)
    b = a.copy()
    b[5, 5] = 107
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    out = cli("compare", tmp_path / "a.png", tmp_path / "b.png").stdout
    assert "max |diff| 7" in out
    assert "mean |diff| 0.004" in out                             # 7 / 1600 pixels
    same = cli("compare", tmp_path / "a.png", tmp_path / "a.png").stdout
    assert "SSIM 1.0000" in same and "max |diff| 0" in same
    assert "SSIM 1.0000" not in out


# --------------------------------------------------------------------------- key
def test_key_lifts_the_element_and_heals_the_hole(tmp_path):
    im = np.full((100, 100, 3), 120, np.uint8)
    im[30:50, 30:50] = (220, 20, 20)
    Image.fromarray(im).save(tmp_path / "k.png")
    cli("key", tmp_path / "k.png", tmp_path / "o", "--box", "10,10,70,70", "--to", "75,75", "--scale", "1")
    base = np.array(Image.open(tmp_path / "o" / "base.png").convert("RGB")).astype(int)
    prev = np.array(Image.open(tmp_path / "o" / "preview.png").convert("RGB")).astype(int)
    el = alpha_of(tmp_path / "o" / "element.png")
    assert np.abs(base[40, 40] - 120).max() <= 3                  # hole healed with the flat colour
    assert np.abs(prev[75, 75] - (220, 20, 20)).max() <= 3        # element now sits at --to
    assert np.abs(prev[40, 40] - 120).max() <= 3                  # and is gone from its old place
    assert 300 <= (el > 128).sum() <= 500                         # about the 20x20 square
    assert np.abs(prev[0:10] - im[0:10]).max() == 0               # outside the box nothing changed


# --------------------------------------------------------------------------- join / split guard rails
def test_join_refuses_a_missing_part(photo_split, tmp_path):
    out, _ = photo_split
    (out / "part_2.png").unlink()
    assert "part_2.png" in fails_cleanly("join", out, tmp_path / "x.png")


def test_join_refuses_parts_of_different_size(photo_split, tmp_path):
    out, _ = photo_split
    Image.new("RGBA", (10, 10)).save(out / "part_1.png")
    fails_cleanly("join", out, tmp_path / "x.png")


def test_join_refuses_a_broken_manifest(photo_split, tmp_path):
    out, _ = photo_split
    (out / "parts.json").write_text("{}")
    fails_cleanly("join", out, tmp_path / "x.png")


def test_split_replaces_only_its_own_files(tmp_path):
    out = tmp_path / "o"
    cli("split", EX / "demo.png", out)
    (out / "part_notes.png").write_bytes(b"mine")
    (out / "readme.txt").write_text("mine")
    cli("split", EX / "demo.png", out, "--angles", "90,270")
    assert (out / "part_notes.png").read_bytes() == b"mine"
    assert (out / "readme.txt").exists()
    assert not (out / "part_2.png").exists() and not (out / "part_3.png").exists()


def test_alpha_file_over_white_is_the_original(tmp_path):
    cli("alpha", EX / "demo.png", tmp_path / "a.png")
    rgba = np.array(Image.open(tmp_path / "a.png").convert("RGBA")).astype(float)
    al = rgba[:, :, 3:4] / 255
    over = rgba[:, :, :3] * al + 255 * (1 - al)
    orig = np.array(Image.open(EX / "demo.png").convert("RGB")).astype(float)
    assert np.abs(over - orig).max() <= 1.0


# --------------------------------------------------------------------------- input hardening
@pytest.mark.parametrize("args", [
    ["split", "IMG", "o", "--centre", "nan,0"],
    ["split", "IMG", "o", "--angles", "nan,10"],
    ["split", "IMG", "o", "--angles", "inf,10"],
    ["split", "IMG", "o", "--simplify", "-1"],
    ["key", "IMG", "o", "--box", "5000,5000,6000,6000"],
    ["key", "IMG", "o", "--box", "1,2,3"],
    ["key", "IMG", "o", "--box", "50,50,10,10"],
    ["key", "IMG", "o", "--box", "0,0,50,50", "--to", "5000,5000"],
    ["key", "IMG", "o", "--box", "0,0,50,50", "--scale", "5"],
    ["key", "IMG", "o", "--box", "0,0,50,50", "--scale", "0"],
    ["trace", "MISSING", "o.svg"],
    ["join", "NODIR", "o.png"],
])
def test_bad_input_is_one_clear_line(tmp_path, args):
    mapping = {"IMG": str(EX / "demo.png"), "MISSING": str(tmp_path / "none.png"), "o": str(tmp_path / "o"),
               "o.svg": str(tmp_path / "o.svg"), "o.png": str(tmp_path / "o.png"), "NODIR": str(tmp_path / "nodir")}
    fails_cleanly(*[mapping.get(x, x) for x in args])


def test_key_flat_box_says_so(tmp_path):
    Image.new("RGB", (100, 100), (120, 120, 120)).save(tmp_path / "f.png")
    assert "nothing stands out" in fails_cleanly("key", tmp_path / "f.png", tmp_path / "o", "--box", "0,0,50,50")


def test_48bit_and_cmyk_are_announced(tmp_path):
    deep = (np.random.RandomState(0).rand(30, 30, 3) * 65535).astype(np.uint16)
    cv2.imwrite(str(tmp_path / "deep.png"), deep)
    assert "16 bit" in cli("compare", tmp_path / "deep.png", tmp_path / "deep.png").stderr
    Image.new("CMYK", (20, 20), (10, 20, 30, 40)).save(tmp_path / "c.tif")
    assert "CMYK" in cli("compare", tmp_path / "c.tif", tmp_path / "c.tif").stderr


def test_48bit_values_are_scaled_not_clipped(tmp_path):
    cv2.imwrite(str(tmp_path / "deep.png"), np.full((30, 30, 3), 30000, np.uint16))
    Image.fromarray(np.full((30, 30, 3), 117, np.uint8)).save(tmp_path / "ref.png")      # 30000 / 257 = 116.7
    assert "max |diff| 0" in cli("compare", tmp_path / "deep.png", tmp_path / "ref.png").stdout


def test_lanczos_fallback_keeps_transparency(tmp_path):
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[2:8, 2:8] = (200, 30, 30, 255)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "t.png")
    r = cli("upscale", tmp_path / "t.png", tmp_path / "u.png", "--bin", str(tmp_path / "nothing.exe"), "--scale", "2")
    assert "Lanczos" in r.stdout
    up = Image.open(tmp_path / "u.png")
    assert up.mode == "RGBA" and up.size == (20, 20)
    assert np.array(up)[0, 0, 3] == 0


def test_download_is_pinned_by_checksum():
    sys.path.insert(0, str(ROOT))
    import pixelcut
    assert set(pixelcut.ESRGAN_SHA256) == {"windows", "ubuntu", "macos"}
    assert all(len(v) == 64 for v in pixelcut.ESRGAN_SHA256.values())
