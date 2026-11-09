"""Second ring of tests: analytic geometry (close and odd angles), the numbers printed by --verify, gap/edge widths, input
conversions with known results, key/trace/upscale details, manifest handling and the download guard."""
import json
import os
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
EX = EX_DIR = ROOT / "examples"
sys.path.insert(0, str(ROOT))


def cli(*args, check=True):
    return subprocess.run([sys.executable, str(ROOT / "pixelcut.py"), *map(str, args)], capture_output=True, text=True, check=check)


def fails(*args):
    r = cli(*args, check=False)
    assert r.returncode != 0 and "Traceback" not in r.stderr and r.stderr.strip(), (args, r.stderr)
    return r.stderr


def rgba_of(path):
    return np.array(Image.open(path).convert("RGBA"))


def owners(out, n):
    stack = np.stack([rgba_of(out / ("part_%d.png" % k))[:, :, 3] > 0 for k in range(n)])
    return stack.argmax(0), stack.sum(0)


def over_white(parts):
    acc = np.full(parts[0].shape[:2] + (3,), 255.0)
    for p in parts:
        al = p[:, :, 3:4] / 255.0
        acc = acc * (1 - al) + p[:, :, :3] * al
    return acc


def printed(out, label):
    return float(out.split(label)[1].split("/")[0].replace(":", "").strip())


# --------------------------------------------------------------------------- geometry
def expected_owner(shape, centre, angles):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    ang = (np.degrees(np.arctan2(yy - centre[1], xx - centre[0])) + 360) % 360
    starts = sorted(v % 360 for v in angles)
    own = np.full((h, w), len(starts) - 1)
    for k, s in enumerate(starts):
        own[ang >= s] = k
    dist = np.min([np.abs(((ang - s) + 180) % 360 - 180) for s in starts], axis=0)
    return own, dist


@pytest.mark.parametrize("angles", ["10,30,200", "0,20,40,180", "0,10,20", "-90,90", "360,180,90", "5,15,25,35,45,55,65,75,85,95"])
def test_ownership_matches_the_analytic_geometry(tmp_path, angles):
    out = tmp_path / "o"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles=" + angles, "--depth", "0", "--gap", "0", "--edge", "0", "--centre", "200,150")
    n = len(angles.split(","))
    got, count = owners(out, n)
    assert count.min() == 1 and count.max() == 1
    want, dist = expected_owner(got.shape, (200, 150), [float(v) for v in angles.split(",")])
    far = dist > 1.0                                              # a pixel right on a line may go either way
    assert (got[far] != want[far]).sum() == 0


def test_negative_and_full_turn_angles_equal_their_normal_form(tmp_path):
    for a, b in (("-90,90", "270,90"), ("360,180", "0,180")):
        cli("split", EX / "demo.png", tmp_path / "a", "--angles=" + a, "--depth", "0", "--gap", "0", "--edge", "0", "--centre", "300,240")
        cli("split", EX / "demo.png", tmp_path / "b", "--angles=" + b, "--depth", "0", "--gap", "0", "--edge", "0", "--centre", "300,240")
        ma = json.loads((tmp_path / "a" / "parts.json").read_text())
        for k in range(2):
            assert (rgba_of(tmp_path / "a" / ("part_%d.png" % k)) == rgba_of(tmp_path / "b" / ("part_%d.png" % k))).all()
        assert [p["starts_at_angle"] for p in ma["parts"]] == sorted(p["starts_at_angle"] for p in ma["parts"])


def test_unsorted_angles_give_the_same_parts(tmp_path):
    for name, angles in (("s", "20,110,200,290"), ("u", "290,20,200,110")):
        cli("split", EX / "photo_cat.png", tmp_path / name, "--opaque", "--angles", angles, "--depth", "0", "--gap", "0", "--edge", "0", "--centre", "225,150")
    for k in range(4):
        assert (rgba_of(tmp_path / "s" / ("part_%d.png" % k)) == rgba_of(tmp_path / "u" / ("part_%d.png" % k))).all()
    assert [p["starts_at_angle"] for p in json.loads((tmp_path / "u" / "parts.json").read_text())["parts"]] == [20, 110, 200, 290]


def test_zigzag_phase_of_the_second_line(tmp_path):
    """Line 1 (110 degrees) has its teeth shifted by a quarter period: at along = 101.5 px it stands at +depth."""
    a = math.radians(110)
    x, y = 225 + 101.5 * math.cos(a) - 4.5 * math.sin(a), 150 + 101.5 * math.sin(a) + 4.5 * math.cos(a)
    res = {}
    for name, depth in (("zig", "9"), ("straight", "0")):
        cli("split", EX / "photo_cat.png", tmp_path / name, "--opaque", "--angles", "20,110,200,290", "--depth", depth, "--tooth", "14",
            "--gap", "0", "--edge", "0", "--centre", "225,150")
        res[name] = [k for k in range(4) if rgba_of(tmp_path / name / ("part_%d.png" % k))[int(round(y)), int(round(x)), 3] > 0]
    assert res["straight"] == [1] and res["zig"] == [0]


def test_auto_centre_is_the_centroid_of_the_non_white_area(tmp_path):
    cli("split", EX / "demo.png", tmp_path / "o")
    m = json.loads((tmp_path / "o" / "parts.json").read_text())
    rgb = np.array(Image.open(EX / "demo.png").convert("RGB"))
    ys, xs = np.nonzero(rgb.min(2) < 250)
    assert abs(m["centre"][0] - xs.mean()) < 0.01 and abs(m["centre"][1] - ys.mean()) < 0.01


def test_gap_and_edge_have_the_requested_width(tmp_path):
    cli("split", EX / "photo_cat.png", tmp_path / "g", "--opaque", "--angles", "0,180", "--depth", "0", "--gap", "8", "--edge", "0", "--centre", "225,150")
    parts = [rgba_of(tmp_path / "g" / ("part_%d.png" % k)) for k in range(2)]
    for y in (147, 150, 153):
        assert all(p[y, 300, 3] == 0 for p in parts), y            # inside the slit (|perp| < 4)
    for y in (145, 155):
        assert sum(int(p[y, 300, 3] > 0) for p in parts) == 1, y   # outside: owned by exactly one part
    cli("split", EX / "photo_cat.png", tmp_path / "e", "--opaque", "--angles", "0,180", "--depth", "0", "--gap", "4", "--edge", "3", "--centre", "225,150")
    parts = [rgba_of(tmp_path / "e" / ("part_%d.png" % k)) for k in range(2)]
    orig = np.array(Image.open(EX / "photo_cat.png").convert("RGB"))
    owner = [p for p in parts if p[153, 300, 3] > 0][0]
    assert tuple(owner[153, 300, :3]) == (20, 16, 14) and owner[153, 300, 3] == 255      # ink edge 2..5 px from the line
    owner = [p for p in parts if p[157, 300, 3] > 0][0]
    assert tuple(owner[157, 300, :3]) == tuple(orig[157, 300])                           # beyond the edge: original pixel


def test_verify_numbers_are_measured_not_made_up(tmp_path):
    out = cli("split", EX / "demo.png", tmp_path / "x", "--gap", "0", "--edge", "0", "--depth", "0", "--verify").stdout
    parts = [rgba_of(tmp_path / "x" / ("part_%d.png" % k)).astype(float) for k in range(4)]
    orig = np.array(Image.open(EX / "demo.png").convert("RGB")).astype(float)
    measured = np.abs(over_white(parts) - orig).max()
    assert measured > 0.1                                           # the colour-to-alpha rounding is real, so a constant 0 would be a lie
    assert abs(printed(out, "without slit:") - measured) < 0.02
    assert abs(printed(out, "saved files joined over white:") - measured) < 0.02
    out = cli("split", EX / "demo.png", tmp_path / "y", "--verify").stdout
    parts = [rgba_of(tmp_path / "y" / ("part_%d.png" % k)).astype(float) for k in range(4)]
    assert abs(printed(out, "saved files joined over white:") - np.abs(over_white(parts) - orig).max()) < 0.02
    assert printed(out, "saved files joined over white:") > 5      # with the default slit the files are not the original


def test_alpha_verify_is_measured(tmp_path):
    out = cli("alpha", EX / "demo.png", tmp_path / "a.png", "--verify").stdout
    rgba = rgba_of(tmp_path / "a.png").astype(float)
    orig = np.array(Image.open(EX / "demo.png").convert("RGB")).astype(float)
    assert abs(printed(out, "over white:") - np.abs(over_white([rgba]) - orig).max()) < 0.02


def test_simplify_changes_the_polygon(tmp_path):
    cli("split", EX / "photo_cat.png", tmp_path / "a", "--opaque", "--simplify", "0.5")
    cli("split", EX / "photo_cat.png", tmp_path / "b", "--opaque", "--simplify", "25")
    na = len(json.loads((tmp_path / "a" / "parts.json").read_text())["parts"][0]["polygon"])
    nb = len(json.loads((tmp_path / "b" / "parts.json").read_text())["parts"][0]["polygon"])
    assert na > nb >= 3


# --------------------------------------------------------------------------- input conversions with known results
def test_rgba_goes_over_white_not_black(tmp_path):
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[:] = (255, 0, 0, 128)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    Image.fromarray(np.full((20, 20, 3), (255, 127, 127), np.uint8)).save(tmp_path / "ref.png")      # red at 50 % over white
    assert "max |diff| 0" in cli("compare", tmp_path / "a.png", tmp_path / "ref.png").stdout or \
        int(cli("compare", tmp_path / "a.png", tmp_path / "ref.png").stdout.split("max |diff|")[1]) <= 1


def test_16bit_divides_by_257(tmp_path):
    cv2.imwrite(str(tmp_path / "d.png"), np.full((20, 20, 3), 65280, np.uint16))
    Image.fromarray(np.full((20, 20, 3), 254, np.uint8)).save(tmp_path / "ref.png")                # 65280 / 257 = 254.0 (/256 would give 255)
    assert "max |diff| 0" in cli("compare", tmp_path / "d.png", tmp_path / "ref.png").stdout


def test_16bit_rgba_is_composited_over_white(tmp_path):
    bgra = np.zeros((20, 20, 4), np.uint16)                                                       # fully transparent black
    cv2.imwrite(str(tmp_path / "d.png"), bgra)
    Image.fromarray(np.full((20, 20, 3), 255, np.uint8)).save(tmp_path / "white.png")
    assert "max |diff| 0" in cli("compare", tmp_path / "d.png", tmp_path / "white.png").stdout


# --------------------------------------------------------------------------- key
def _key_image(tmp_path):
    im = np.full((100, 100, 3), 120, np.uint8)
    im[20:40, 30:50] = (220, 20, 20)
    Image.fromarray(im).save(tmp_path / "k.png")
    return im


def test_key_scale_is_applied(tmp_path):
    _key_image(tmp_path)
    cli("key", tmp_path / "k.png", tmp_path / "half", "--box", "10,10,90,50", "--scale", "0.5", "--to", "70,70")
    cli("key", tmp_path / "k.png", tmp_path / "full", "--box", "10,10,90,50", "--scale", "1", "--to", "70,70")
    half, full = [int((rgba_of(tmp_path / n / "element.png")[:, :, 3] > 128).sum()) for n in ("half", "full")]
    assert 80 <= half <= 130 and 300 <= full <= 500


def test_key_default_target_is_the_box_centre(tmp_path):
    _key_image(tmp_path)
    cli("key", tmp_path / "k.png", tmp_path / "o", "--box", "10,10,90,50", "--scale", "1")
    prev = np.array(Image.open(tmp_path / "o" / "preview.png").convert("RGB")).astype(int)
    assert np.abs(prev[30, 55] - (220, 20, 20)).max() <= 3        # box centre is (50,30): the element sits 10 px right of where it was
    assert np.abs(prev[30, 33] - 120).max() <= 3                  # and its old left edge is healed


def test_key_rejects_bad_numbers(tmp_path):
    _key_image(tmp_path)
    for extra in (["--close", "0"], ["--close", "-3"], ["--threshold", "-1"]):
        fails("key", tmp_path / "k.png", tmp_path / "o", "--box", "10,10,90,50", *extra)


# --------------------------------------------------------------------------- trace / upscale
def test_trace_writes_an_svg(tmp_path):
    cli("trace", EX / "demo.png", tmp_path / "o.svg")
    svg = (tmp_path / "o.svg").read_text()
    assert svg.lstrip().startswith(("<?xml", "<svg")) and "<path" in svg and len(svg) > 500


@pytest.mark.parametrize("extra", [["--speckle", "-1"], ["--colors", "0"], ["--colors", "9"], ["--layers", "-2"]])
def test_trace_rejects_bad_numbers(tmp_path, extra):
    fails("trace", EX / "demo.png", tmp_path / "o.svg", *extra)


def test_trace_rejects_a_missing_output_folder(tmp_path):
    fails("trace", EX / "demo.png", tmp_path / "nix" / "o.svg")


@pytest.mark.parametrize("scale", ["0", "100000", "-2"])
def test_upscale_rejects_silly_scales(tmp_path, scale):
    fails("upscale", EX / "demo.png", tmp_path / "u.png", "--scale", scale, "--bin", str(tmp_path / "none.exe"))


# --------------------------------------------------------------------------- manifest / output folder
def _split(tmp_path):
    out = tmp_path / "p"
    cli("split", EX / "demo.png", out, "--gap", "0", "--edge", "0", "--depth", "0")
    return out


@pytest.mark.parametrize("bad", ['[1,2]', '{"files":5}', '{"files":[1,null],"parts":[{},{}],"size":[600,480]}', '{"parts":[{}],"size":[]}',
                                 '{"parts":[{}],"size":[100]}', '{"parts":[{}],"size":[600,480],"files":["../x.png"]}'])
def test_broken_manifest_never_crashes(tmp_path, bad):
    out = _split(tmp_path)
    (out / "parts.json").write_text(bad)
    fails("join", out, tmp_path / "x.png")
    r = cli("split", EX / "demo.png", out, check=False)                    # second split into the folder: clean message or clean success
    assert "Traceback" not in r.stderr


def test_split_refuses_to_overwrite_a_foreign_part(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "part_0.png").write_bytes(b"my own picture")
    assert "part_0.png" in fails("split", EX / "demo.png", out)
    assert (out / "part_0.png").read_bytes() == b"my own picture"


def test_split_leaves_no_staging_folder_and_old_parts_survive_a_failed_run(tmp_path):
    out = _split(tmp_path)
    before = (out / "part_0.png").read_bytes()
    fails("split", EX / "demo.png", out, "--gap", "nan")
    assert (out / "part_0.png").read_bytes() == before
    assert not (out / ".pixelcut_new").exists()


@pytest.mark.parametrize("flag", ["--gap", "--edge", "--depth", "--tooth"])
@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_numbers_must_be_finite_and_not_negative(tmp_path, flag, value):
    fails("split", EX / "demo.png", tmp_path / "o", flag + "=" + value)


def test_join_without_manifest_needs_every_part(tmp_path):
    out = _split(tmp_path)
    (out / "parts.json").unlink()
    (out / "part_1.png").unlink()
    assert "part_1.png" in fails("join", out, tmp_path / "x.png")


def test_join_checks_the_size_in_the_manifest(tmp_path):
    out = _split(tmp_path)
    m = json.loads((out / "parts.json").read_text())
    m["size"] = [10, 10]
    (out / "parts.json").write_text(json.dumps(m))
    assert "size" in fails("join", out, tmp_path / "x.png")


def test_animated_gif_is_announced(tmp_path):
    frames = [Image.new("RGB", (20, 20), c) for c in ("red", "blue")]
    frames[0].save(tmp_path / "a.gif", save_all=True, append_images=frames[1:])
    assert "2 frames" in cli("compare", tmp_path / "a.gif", tmp_path / "a.gif").stderr


# --------------------------------------------------------------------------- the download guard
def test_download_checksum_and_zip_slip(tmp_path, monkeypatch, capsys):
    import hashlib
    import platform
    import shutil
    import urllib.request
    import zipfile

    import pixelcut
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sys, "platform", "linux")
    cache = tmp_path / ".cache" / "pixelcut" / "realesrgan"

    def good_zip(url, dest, extra=()):
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("pkg/realesrgan-ncnn-vulkan", "#!/bin/sh")
            for name in extra:
                z.writestr(name, "x")

    # 1. a perfectly valid archive with the WRONG checksum must be rejected before anything is extracted
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: good_zip(url, dest))
    assert pixelcut.ensure_esrgan() is None
    assert "checksum does not match" in capsys.readouterr().out
    assert not list(cache.rglob("realesrgan-ncnn-vulkan*"))

    # 2. right checksum, but the archive tries to leave the folder -> refused, nothing extracted
    class AnyHash(dict):
        def __getitem__(self, key):
            return hashlib.sha256((cache / "release.zip").read_bytes()).hexdigest()
    monkeypatch.setattr(pixelcut, "ESRGAN_SHA256", AnyHash())
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: good_zip(url, dest, ("../evil.txt",)))
    assert pixelcut.ensure_esrgan() is None
    assert "unsafe path" in capsys.readouterr().out
    assert not list(cache.rglob("realesrgan-ncnn-vulkan*"))

    # 3. the same archive without the bad member works - so the refusal above was really about the path
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: good_zip(url, dest))
    assert pixelcut.ensure_esrgan() is not None


@pytest.mark.parametrize("angles", ["0,20,200", "10,25,40,190"])
def test_zigzag_does_not_disturb_close_lines(tmp_path, angles):
    out = tmp_path / "z"
    depth = 8
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles=" + angles, "--depth", str(depth), "--tooth", "40", "--gap", "0", "--edge", "0", "--centre", "200,150")
    vals = [float(v) for v in angles.split(",")]
    got, count = owners(out, len(vals))
    assert count.min() == 1 and count.max() == 1
    want, _ = expected_owner(got.shape, (200, 150), vals)
    h, w = got.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - 200, yy - 150)
    ang = (np.degrees(np.arctan2(yy - 150, xx - 200)) + 360) % 360
    ray = []                                                      # distance of every pixel to every line (a ray, not an endless line)
    for s in vals:
        d = np.abs(((ang - s) + 180) % 360 - 180)
        ray.append(np.where(d < 90, r * np.sin(np.radians(d)), r))
    far = np.min(ray, axis=0) > depth + 3
    assert far.sum() > 1000
    assert (got[far] != want[far]).sum() == 0


def test_colour_key_transparency_is_composited_over_white(tmp_path):
    im = Image.new("RGB", (4, 4), (0, 0, 0))
    im.save(tmp_path / "k.png", transparency=(0, 0, 0))
    r = cli("alpha", tmp_path / "k.png", tmp_path / "k_out.png")
    assert "transparency" in r.stderr
    assert rgba_of(tmp_path / "k_out.png")[:, :, 3].max() == 0     # black was the transparent key, white remains -> nothing left


def test_join_refuses_duplicate_part_names(tmp_path):
    out = tmp_path / "d"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,90,180,270", "--gap", "0", "--edge", "0")
    m = json.loads((out / "parts.json").read_text())
    m["files"] = ["part_0.png"] * 4
    (out / "parts.json").write_text(json.dumps(m))
    assert "damaged" in fails("join", out, tmp_path / "j.png")


def test_split_rolls_back_when_the_swap_is_interrupted(tmp_path, monkeypatch):
    import os
    import pixelcut
    out = tmp_path / "r"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        if Path(a).parent.name == ".pixelcut_new":
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk went away")
        return real(a, b)
    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(sys, "argv", ["pixelcut", "split", str(EX / "photo_cat.png"), str(out), "--opaque", "--angles", "0,90,180", "--gap", "0", "--edge", "0"])
    with pytest.raises(SystemExit):
        pixelcut.main()
    after = {f.name: f.read_bytes() for f in out.iterdir()}
    assert after == before                                         # old set intact, no staging/backup left over


def test_interrupted_swap_is_restored_not_deleted(tmp_path):
    out = tmp_path / "i"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    old = out / ".pixelcut_old"                                    # simulate a crash right after the old set was moved aside
    old.mkdir()
    for n in ("part_0.png", "part_1.png", "parts.json"):
        (out / n).replace(old / n)
    (out / "part_0.png").write_bytes(b"half written new file")
    r = cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    assert "rolled back" in r.stderr
    assert not old.exists() and not (out / ".pixelcut_new").exists()
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before


def test_bright_but_not_white_image_can_be_cut_opaque(tmp_path):
    Image.new("RGB", (40, 30), (254, 254, 254)).save(tmp_path / "w.png")
    cli("split", tmp_path / "w.png", tmp_path / "o", "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    assert (tmp_path / "o" / "parts.json").exists()
    cli("split", tmp_path / "w.png", tmp_path / "o3", "--angles", "0,180", "--gap", "0", "--edge", "0")   # faint alpha is still alpha
    Image.new("RGB", (40, 30), (255, 255, 255)).save(tmp_path / "pure.png")
    assert "completely white" in fails("split", tmp_path / "pure.png", tmp_path / "o2")


def test_upscale_no_ai_works_offline(tmp_path):
    Image.new("RGB", (10, 8), "red").save(tmp_path / "s.png")
    cli("upscale", tmp_path / "s.png", tmp_path / "b.png", "--scale", "3", "--no-ai")
    assert Image.open(tmp_path / "b.png").size == (30, 24)


def test_trace_render_never_overwrites_input(tmp_path):
    Image.new("RGB", (20, 20), "black").save(tmp_path / "x.png")
    assert "different file" in fails("trace", tmp_path / "x.png", tmp_path / "x.svg", "--render", tmp_path / "x.png")


def test_rollback_after_a_failed_backup_step_keeps_unmoved_originals(tmp_path, monkeypatch):
    import os
    import pixelcut
    out = tmp_path / "b"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,90,180", "--gap", "0", "--edge", "0")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        if Path(b).parent.name == ".pixelcut_old" and Path(b).name != ".journal.json":             # the SECOND backup move fails: one original is still in place
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk went away")
        return real(a, b)
    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(sys, "argv", ["pixelcut", "split", str(EX / "photo_cat.png"), str(out), "--opaque", "--angles", "0,90,180", "--gap", "0", "--edge", "0"])
    with pytest.raises(SystemExit):
        pixelcut.main()
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before


def test_recovery_leaves_foreign_files_alone(tmp_path):
    out = tmp_path / "f"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    old = out / ".pixelcut_old"
    old.mkdir()
    (old / "notes.txt").write_text("not pixelcut's")
    assert "did not put there" in fails("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180")
    assert (out / "notes.txt").read_text() == "mine" and (old / "notes.txt").exists()


def test_only_inside_on_a_blank_image(tmp_path):
    Image.new("RGB", (40, 30), (255, 255, 255)).save(tmp_path / "w.png")
    cli("split", tmp_path / "w.png", tmp_path / "o", "--opaque", "--only-inside", "--angles", "0,180", "--gap", "0", "--edge", "0")
    assert (tmp_path / "o" / "parts.json").exists()


def test_foreign_manifest_is_never_touched(tmp_path):
    out = tmp_path / "m"
    out.mkdir()
    (out / "parts.json").write_text("not json at all")
    assert "not a pixelcut manifest" in fails("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180")
    assert (out / "parts.json").read_text() == "not json at all"
    (out / "parts.json").write_text('{"mine": 1}')
    assert "not a pixelcut manifest" in fails("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180")
    assert (out / "parts.json").read_text() == '{"mine": 1}'


def test_crash_while_growing_the_set_restores_the_old_one(tmp_path):
    out = tmp_path / "g"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    old = out / ".pixelcut_old"                                    # crash while swapping 2 -> 4 parts: part_2 already installed
    old.mkdir()
    for n in ("part_0.png", "part_1.png", "parts.json"):
        (out / n).replace(old / n)
    (old / ".journal.json").write_text(json.dumps(["part_0.png", "part_2.png"]))
    (out / "part_0.png").write_bytes(b"new")
    (out / "part_2.png").write_bytes(b"new")
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,90,180,270", "--gap", "0", "--edge", "0")
    assert len(list(out.glob("part_*.png"))) == 4                  # the retry worked: part_2 was not mistaken for a foreign file


def test_crash_after_first_backup_step_keeps_the_unmoved_original(tmp_path):
    out = tmp_path / "u"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    old = out / ".pixelcut_old"                                    # crash after moving ONLY part_0 aside; part_1 + parts.json still original
    old.mkdir()
    (out / "part_0.png").replace(old / "part_0.png")
    (old / ".journal.json").write_text("[]")
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before


def test_finished_swap_with_interrupted_cleanup_is_not_rolled_back(tmp_path):
    out = tmp_path / "c"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    done = out / ".pixelcut_done"                                  # crash while deleting the committed backup: half-empty
    done.mkdir()
    (done / "part_1.png").write_bytes(b"old leftover")
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    assert not done.exists()
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before


def test_manifest_from_before_the_files_field_can_be_recut(tmp_path):
    out = tmp_path / "l"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    m = json.loads((out / "parts.json").read_text())
    del m["files"]
    (out / "parts.json").write_text(json.dumps(m))
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,90,180", "--gap", "0", "--edge", "0")
    assert len(list(out.glob("part_*.png"))) == 3


def test_foreign_done_folder_is_never_deleted(tmp_path):
    out = tmp_path / "x"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    (out / ".pixelcut_done").mkdir()
    (out / ".pixelcut_done" / "notes.txt").write_text("mine")
    assert "did not put there" in fails("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180")
    assert (out / ".pixelcut_done" / "notes.txt").read_text() == "mine"


@pytest.mark.parametrize("manifest", ['{"parts": []}', '{"parts": [{}], "files": ["part_0.png"]}',
                                      '{"parts": [{"part": 0, "starts_at_angle": 0}], "size": [10, 10], "centre": [1, 1], "files": ["part_0.png", "part_0.png"]}',
                                      '[1, 2]'])
def test_lookalike_manifests_are_not_ours(tmp_path, manifest):
    out = tmp_path / "n"
    out.mkdir()
    (out / "parts.json").write_text(manifest)
    (out / "part_0.png").write_bytes(b"someone elses picture")
    assert "not a pixelcut manifest" in fails("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180")
    assert (out / "part_0.png").read_bytes() == b"someone elses picture" and (out / "parts.json").read_text() == manifest


def test_failing_rollback_is_finished_by_the_next_run(tmp_path, monkeypatch):
    import os
    import pixelcut
    out = tmp_path / "d"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    real, st = os.replace, {"installs": 0}
    real_unlink = Path.unlink

    def flaky_replace(a, b):                                       # growing 2 -> 4 parts: the 3rd install fails ...
        if Path(a).parent.name == ".pixelcut_new":
            st["installs"] += 1
            if st["installs"] == 3:
                raise OSError("install failed")
        return real(a, b)

    def flaky_unlink(self, *a, **k):                               # ... and then taking the first new file back fails too
        if self.name == "part_0.png" and st["installs"] == 3 and self.parent == out:
            st["installs"] = 99
            raise OSError("rollback failed")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    monkeypatch.setattr(sys, "argv", ["pixelcut", "split", str(EX / "photo_cat.png"), str(out), "--opaque", "--angles", "0,90,180,270", "--gap", "0", "--edge", "0"])
    with pytest.raises(BaseException):
        pixelcut.main()
    monkeypatch.undo()
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")   # next run recovers, then works
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before


def test_empty_part_has_no_fattest_point(tmp_path):
    Image.new("RGB", (40, 30), (254, 254, 254)).save(tmp_path / "w.png")
    cli("split", tmp_path / "w.png", tmp_path / "o", "--angles", "0,180", "--gap", "0", "--edge", "0")
    m = json.loads((tmp_path / "o" / "parts.json").read_text())
    assert all(x["fattest_point"] is None and x["polygon"] == [] for x in m["parts"])


@pytest.mark.parametrize("scratch", [".pixelcut_new", ".pixelcut_done", ".pixelcut_old"])
def test_a_folder_named_like_a_part_is_foreign(tmp_path, scratch):
    out = tmp_path / "s"
    out.mkdir()
    sub = out / scratch / "part_0.png"                             # a DIRECTORY with a part name, holding someone's file
    sub.mkdir(parents=True)
    (sub / "foreign.txt").write_text("keep me")
    assert "pixelcut" in fails("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180")
    assert (sub / "foreign.txt").read_text() == "keep me"


def test_unprintable_characters_in_a_path_do_not_fail_the_job(tmp_path):
    import os
    target = tmp_path / "Prüfung 空 Rundlauf"
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    r = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "pixelcut.py"), "split", str(EX / "photo_cat.png"), str(target),
                        "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0"], capture_output=True, env=env)   # bytes: the console is cp1252 on purpose
    assert r.returncode == 0, r.stderr.decode("cp1252", "replace")
    assert (target / "parts.json").exists()


def test_abort_between_backup_move_and_bookkeeping_loses_nothing(tmp_path, monkeypatch):
    import os
    import pixelcut
    out = tmp_path / "k"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    real, st = os.replace, {"hit": False}

    def interrupt_after_first_backup_move(a, b):
        r = real(a, b)
        if Path(b).parent.name == ".pixelcut_old" and Path(b).name.startswith("part_") and not st["hit"]:
            st["hit"] = True
            raise KeyboardInterrupt                                # the file is already in the backup, the bookkeeping never ran
        return r
    monkeypatch.setattr(os, "replace", interrupt_after_first_backup_move)
    monkeypatch.setattr(sys, "argv", ["pixelcut", "split", str(EX / "photo_cat.png"), str(out), "--opaque", "--angles", "0,90,180", "--gap", "0", "--edge", "0"])
    with pytest.raises(KeyboardInterrupt):
        pixelcut.main()
    assert st["hit"]
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before


@pytest.mark.parametrize("scratch", [".pixelcut_old", ".pixelcut_new", ".pixelcut_done"])
def test_scratch_folder_that_is_a_link_is_refused(tmp_path, scratch):
    out = tmp_path / "j"
    out.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (out / "part_0.png").write_bytes(b"not pixelcut's")             # a foreign file the run must not move
    if sys.platform == "win32":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(out / scratch), str(elsewhere)], capture_output=True)
        if r.returncode != 0:
            pytest.skip("cannot create a junction here")
    else:
        os.symlink(elsewhere, out / scratch)
    err = fails("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180")
    assert "link" in err or "pixelcut" in err
    assert (out / "part_0.png").read_bytes() == b"not pixelcut's"
    assert not list(elsewhere.iterdir())


def test_upscale_abort_keeps_the_existing_result(tmp_path, monkeypatch):
    import pixelcut
    Image.new("RGB", (10, 8), "red").save(tmp_path / "s.png")
    good = tmp_path / "b.png"
    Image.new("RGB", (30, 24), "blue").save(good)
    keep = good.read_bytes()
    real = Image.Image.save

    def partial(self, fp, *a, **k):
        if Path(str(fp)).parent.name.startswith(".pixelcut_"):                # the private scratch folder beside the target
            Path(fp).write_bytes(b"PARTIAL PNG")
            raise KeyboardInterrupt
        return real(self, fp, *a, **k)
    monkeypatch.setattr(Image.Image, "save", partial)
    monkeypatch.setattr(sys, "argv", ["pixelcut", "upscale", str(tmp_path / "s.png"), str(good), "--scale", "3", "--no-ai"])
    with pytest.raises(KeyboardInterrupt):
        pixelcut.main()
    assert good.read_bytes() == keep
    assert not list(tmp_path.glob(".pixelcut_*"))


def test_trace_refuses_same_input_and_output(tmp_path):
    Image.new("RGB", (20, 20), "black").save(tmp_path / "x.png")
    keep = (tmp_path / "x.png").read_bytes()
    assert "different file" in fails("trace", tmp_path / "x.png", tmp_path / "x.png")
    assert (tmp_path / "x.png").read_bytes() == keep
    assert "different file" in fails("trace", tmp_path / "x.png", tmp_path / "x.svg", "--render", tmp_path / "x.svg")


def test_a_file_named_like_the_old_temp_name_survives(tmp_path):
    Image.new("RGB", (8, 8), "white").save(tmp_path / ".pixelcut_tmp_r.png")
    keep = (tmp_path / ".pixelcut_tmp_r.png").read_bytes()
    cli("alpha", tmp_path / ".pixelcut_tmp_r.png", tmp_path / "r.png")
    assert (tmp_path / ".pixelcut_tmp_r.png").read_bytes() == keep


def _key_input(tmp_path):
    img = np.full((60, 80, 3), 235, np.uint8)
    img[20:40, 30:50] = (200, 30, 30)
    Image.fromarray(img).save(tmp_path / "k.png")
    return tmp_path / "k.png"


def test_key_refuses_read_only_results_before_touching_anything(tmp_path):
    import stat
    k = _key_input(tmp_path)
    out = tmp_path / "ko"
    cli("key", k, out, "--box", "10,10,70,50")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    (out / "element.png").chmod(stat.S_IREAD)
    try:
        err = fails("key", k, out, "--box", "10,10,70,50", "--scale", "0.8")
        assert "read-only" in err
        assert {f.name: f.read_bytes() for f in out.iterdir()} == before
    finally:
        (out / "element.png").chmod(stat.S_IWRITE | stat.S_IREAD)


def test_key_interrupt_during_publish_restores_the_old_set(tmp_path, monkeypatch):
    import pixelcut
    k = _key_input(tmp_path)
    out = tmp_path / "ko"
    cli("key", k, out, "--box", "10,10,70,50")
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    real, st = os.replace, {"n": 0}

    def flaky(a, b):
        if Path(b).parent.resolve() == out.resolve() and Path(a).parent.resolve() != out.resolve():
            st["n"] += 1
            if st["n"] == 2:                                       # second NEW file going into place
                raise KeyboardInterrupt
        return real(a, b)
    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(sys, "argv", ["pixelcut", "key", str(k), str(out), "--box", "10,10,70,50", "--scale", "0.8"])
    with pytest.raises(KeyboardInterrupt):
        pixelcut.main()
    monkeypatch.undo()
    assert st["n"] >= 2
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before


def test_web_copy_is_the_same_file():
    root = Path(__file__).parent.parent
    assert (root / "docs" / "pixelcut.py").read_bytes() == (root / "pixelcut.py").read_bytes(), "run: copy pixelcut.py docs/pixelcut.py"


def test_key_never_deletes_a_folder_that_has_a_result_name(tmp_path):
    k = _key_input(tmp_path)
    out = tmp_path / "ko"
    (out / "base.png").mkdir(parents=True)
    (out / "base.png" / "mine.txt").write_text("keep me")
    err = fails("key", k, out, "--box", "10,10,70,50")
    assert "not a plain file" in err
    assert (out / "base.png" / "mine.txt").read_text() == "keep me"
    assert not (out / "element.png").exists()


def test_publish_interrupt_right_after_moving_an_old_file_aside_loses_nothing(tmp_path, monkeypatch):
    import pixelcut
    d = tmp_path / "p"
    d.mkdir()
    names = ["a", "b", "c"]
    for n in names:
        (d / n).write_text("old " + n)
        (tmp_path / (n + ".new")).write_text("new " + n)
    pairs = [(tmp_path / (n + ".new"), d / n) for n in names]
    real = os.replace
    for hit in range(1, 7):                                         # an interrupt AFTER the n-th real move, wherever it falls
        for n in names:
            (d / n).write_text("old " + n)
            (tmp_path / (n + ".new")).write_text("new " + n)
        calls = {"n": 0}

        def flaky(a, b, calls=calls, hit=hit):
            r = real(a, b)
            calls["n"] += 1
            if calls["n"] == hit:
                raise KeyboardInterrupt
            return r
        monkeypatch.setattr(os, "replace", flaky)
        try:
            pixelcut.publish(pairs)
        except KeyboardInterrupt:
            pass
        monkeypatch.undo()
        state = {n: (d / n).read_text() if (d / n).exists() else None for n in names}
        assert set(state.values()) in ({"old a", "old b", "old c"}, {"new a", "new b", "new c"}), (hit, state)
        assert not list(d.glob(".pixelcut_*")) and not list(tmp_path.glob(".pixelcut_*")), hit


def test_publish_keeps_the_old_files_when_the_rollback_itself_fails(tmp_path, monkeypatch, capsys):
    import pixelcut
    d = tmp_path / "q"
    d.mkdir()
    for n in ("a", "b"):
        (d / n).write_text("old " + n)
        (tmp_path / (n + ".new")).write_text("new " + n)
    pairs = [(tmp_path / (n + ".new"), d / n) for n in ("a", "b")]
    real, st = os.replace, {"n": 0}

    def flaky(a, b):
        st["n"] += 1
        if st["n"] == 4:                                            # 2 moves aside, 1st install fine, 2nd install fails
            raise OSError("install failed")
        if st["n"] >= 5:                                            # ... and every restore fails (antivirus lock)
            raise PermissionError("locked")
        return real(a, b)
    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError):
        pixelcut.publish(pairs)
    monkeypatch.undo()
    kept = list(d.glob(".pixelcut_*/*"))
    assert {f.read_text() for f in kept} == {"old a", "old b"}      # the only copies of the old files still exist
    assert "previous files are kept in" in capsys.readouterr().err


def test_outputs_may_never_be_inputs(tmp_path):
    Image.new("RGB", (10, 8), "red").save(tmp_path / "o.png")
    keep = (tmp_path / "o.png").read_bytes()
    assert "also an input" in fails("upscale", tmp_path / "o.png", tmp_path / "o.png", "--no-ai")
    assert "also an input" in fails("alpha", tmp_path / "o.png", tmp_path / "o.png")
    assert (tmp_path / "o.png").read_bytes() == keep
    out = tmp_path / "pp"
    cli("split", EX / "photo_cat.png", out, "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0")
    assert "also an input" in fails("join", out, out / "part_0.png")
    assert (out / "part_0.png").exists() and json.loads((out / "parts.json").read_text())["files"]
    assert "does not exist" in fails("alpha", tmp_path / "o.png", tmp_path / "nope" / "x.png")


def test_exif_rotation_is_applied_with_a_notice(tmp_path):
    im = Image.new("RGB", (100, 40), "white")
    ex = im.getexif()
    ex[274] = 6                                                     # camera held upright: stored sideways, viewers turn it
    im.save(tmp_path / "phone.jpg", exif=ex)
    r = cli("alpha", tmp_path / "phone.jpg", tmp_path / "up.png")
    assert "EXIF orientation" in r.stderr
    assert Image.open(tmp_path / "up.png").size == (40, 100)


def test_python_api_roundtrip_is_exact_and_quiet(tmp_path, capsys):
    import pixelcut
    src = EX / "photo_cat.png"
    parts = tmp_path / "pp"
    info = pixelcut.split(src, parts, opaque=True, angles=[0, 120, 240], gap=0, edge=0)
    assert len(info["files"]) == 3 and info["size"]
    out = pixelcut.join(parts, tmp_path / "back.png")
    r = pixelcut.compare(src, out)
    assert r["max_diff"] == 0 and r["ssim"] == 1.0
    assert capsys.readouterr().out == ""                            # a library does not print


def test_python_api_raises_instead_of_exiting(tmp_path):
    import pixelcut
    with pytest.raises(pixelcut.PixelcutError, match="angles"):
        pixelcut.split(EX / "photo_cat.png", tmp_path / "x", angles="5")
    with pytest.raises(pixelcut.PixelcutError, match="unrecognized"):
        pixelcut.split(EX / "photo_cat.png", tmp_path / "a", nonsense=1)
    with pytest.raises(pixelcut.PixelcutError, match="different sizes"):
        Image.new("RGB", (5, 5)).save(tmp_path / "s.png")
        pixelcut.compare(EX / "photo_cat.png", tmp_path / "s.png")


def test_grid_and_strips_are_exact_for_any_size(tmp_path):
    import pixelcut
    rng = np.random.RandomState(5)
    for (w, h, grid) in [(97, 61, "4x3"), (50, 40, "1x7"), (50, 40, "9x1"), (13, 11, "13x11"), (200, 120, "10x10")]:
        Image.fromarray((rng.rand(h, w, 3) * 255).astype("uint8")).save(tmp_path / "g.png")
        out = tmp_path / ("g_" + grid)
        info = pixelcut.split(tmp_path / "g.png", out, grid=grid)
        cols, rows = (int(v) for v in grid.split("x"))
        assert len(info["files"]) == cols * rows and info["grid"] == [cols, rows]
        masks = [np.array(Image.open(out / f).getchannel("A")) > 0 for f in info["files"]]
        assert (np.sum(masks, axis=0) == 1).all()                   # every pixel in exactly one cell
        assert all(m.any() for m in masks)                          # no empty cell
        pixelcut.join(out, tmp_path / "j.png")
        assert pixelcut.compare(tmp_path / "g.png", tmp_path / "j.png")["max_diff"] == 0


def test_grid_rejects_nonsense(tmp_path):
    import pixelcut
    Image.new("RGB", (20, 10), "red").save(tmp_path / "g.png")
    for bad, word in [("1x1", "two cells"), ("abc", "must look like"), ("30x1", "does not fit"), ("0x5", "two cells"), ("40x40", "at most")]:
        with pytest.raises(pixelcut.PixelcutError, match=word):
            pixelcut.split(tmp_path / "g.png", tmp_path / "o", grid=bad)


def test_label_map_roundtrip_and_arbitrary_shapes(tmp_path):
    import pixelcut
    rng = np.random.RandomState(9)
    w, h = 71, 53
    Image.fromarray((rng.rand(h, w, 3) * 255).astype("uint8")).save(tmp_path / "o.png")
    # 1. the map of a grid, then split along it: identical to --grid
    pixelcut.labels(tmp_path / "o.png", tmp_path / "map.png", grid="5x4")
    assert np.array(Image.open(tmp_path / "map.png")).max() == 19
    a = pixelcut.split(tmp_path / "o.png", tmp_path / "a", from_labels=tmp_path / "map.png")
    b = pixelcut.split(tmp_path / "o.png", tmp_path / "b", grid="5x4")
    assert len(a["files"]) == len(b["files"]) == 20 and a["from_labels"] is True
    for f in a["files"]:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    # 2. a free-form map (random blobs, non-consecutive numbers, 16 bit): exact
    blobs = (rng.rand(h, w) * 7).astype(np.uint16) * 300 + 5
    Image.fromarray(blobs).save(tmp_path / "blobs.png")
    info = pixelcut.split(tmp_path / "o.png", tmp_path / "c", from_labels=tmp_path / "blobs.png")
    masks = [np.array(Image.open(tmp_path / "c" / f).getchannel("A")) > 0 for f in info["files"]]
    assert len(masks) == 7 and (np.sum(masks, axis=0) == 1).all()
    pixelcut.join(tmp_path / "c", tmp_path / "j.png")
    assert pixelcut.compare(tmp_path / "o.png", tmp_path / "j.png")["max_diff"] == 0


def test_label_map_rejects_nonsense(tmp_path):
    import pixelcut
    Image.new("RGB", (20, 10), "red").save(tmp_path / "o.png")
    Image.new("RGB", (20, 10), "blue").save(tmp_path / "colour.png")
    Image.new("L", (19, 10), 0).save(tmp_path / "wrongsize.png")
    Image.new("L", (20, 10), 3).save(tmp_path / "one.png")
    (tmp_path / "junk.png").write_text("no picture")
    for m, word in [("colour.png", "grey picture"), ("wrongsize.png", "must match"), ("one.png", "only one value"), ("junk.png", "cannot read")]:
        with pytest.raises(pixelcut.PixelcutError, match=word):
            pixelcut.split(tmp_path / "o.png", tmp_path / "x", from_labels=tmp_path / m)
    with pytest.raises(pixelcut.PixelcutError, match="not both"):
        pixelcut.split(tmp_path / "o.png", tmp_path / "x", grid="2x2", from_labels=tmp_path / "one.png")
    with pytest.raises(pixelcut.PixelcutError, match="is also an input"):
        pixelcut.labels(tmp_path / "o.png", tmp_path / "o.png", grid="2x2")


def test_input_inside_a_work_folder_is_refused_and_kept(tmp_path):
    import pixelcut
    for work in (".pixelcut_new", ".pixelcut_done", ".pixelcut_old"):
        out = tmp_path / ("o" + work)
        (out / work).mkdir(parents=True)
        Image.new("RGB", (30, 20), "red").save(out / work / "part_0.png")
        with pytest.raises(pixelcut.PixelcutError, match="work folder"):
            pixelcut.split(out / work / "part_0.png", out, grid="2x2")
        assert (out / work / "part_0.png").exists(), work


def test_join_and_verify_hold_one_picture_not_all_parts(tmp_path):
    import pixelcut
    Image.fromarray((np.random.RandomState(3).rand(40, 60, 3) * 255).astype("uint8")).save(tmp_path / "o.png")
    pixelcut.split(tmp_path / "o.png", tmp_path / "p", grid="8x5", opaque=True, verify=True)
    calls = []
    real = pixelcut.over_white

    def watch(it):
        assert not isinstance(it, list)                             # a generator: parts are produced one at a time
        calls.append(1)
        return real(it)
    pixelcut.over_white = watch
    try:
        pixelcut.join(tmp_path / "p", tmp_path / "j.png")
    finally:
        pixelcut.over_white = real
    assert calls and pixelcut.compare(tmp_path / "o.png", tmp_path / "j.png")["max_diff"] == 0


def test_labels_refuses_lossy_formats(tmp_path):
    import pixelcut
    Image.new("RGB", (32, 32), "red").save(tmp_path / "o.png")
    with pytest.raises(pixelcut.PixelcutError, match="lossless"):
        pixelcut.labels(tmp_path / "o.png", tmp_path / "m.jpg", grid="8x8")
    assert not (tmp_path / "m.jpg").exists()


def _png_with_exif(path, arr16, orient):
    """Real 16-bit RGB PNG with an EXIF orientation: Pillow cannot write 16-bit RGB, so the eXIf chunk is inserted by hand."""
    import struct
    import zlib
    cv2.imencode(".png", arr16)[1].tofile(str(path))
    ex = Image.Exif()
    ex[274] = orient
    payload = ex.tobytes()
    payload = payload[6:] if payload.startswith(b"Exif\x00\x00") else payload
    chunk = struct.pack(">I", len(payload)) + b"eXIf" + payload + struct.pack(">I", zlib.crc32(b"eXIf" + payload) & 0xffffffff)
    raw = Path(path).read_bytes()
    cut = 8 + 4 + 4 + 13 + 4                                      # signature + IHDR chunk
    Path(path).write_bytes(raw[:cut] + chunk + raw[cut:])


def test_exif_rotation_also_for_16_bit_png(tmp_path):
    import pixelcut
    for orient in (2, 3, 4, 5, 6, 7, 8):
        a16 = (np.arange(6 * 10 * 3).reshape(6, 10, 3) * 300).astype(np.uint16)
        _png_with_exif(tmp_path / "e.png", a16, orient)
        assert cv2.imdecode(np.fromfile(str(tmp_path / "e.png"), np.uint8), cv2.IMREAD_UNCHANGED).dtype == np.uint16   # really 16 bit
        want = np.array(ImageOps.exif_transpose(Image.open(tmp_path / "e.png")).convert("RGB"))
        got = pixelcut.load_rgb(tmp_path / "e.png")
        assert got.shape == want.shape, orient
        pixelcut.labels(tmp_path / "e.png", tmp_path / "m.png", grid="2x2")
        pixelcut.split(tmp_path / "e.png", tmp_path / ("s%d" % orient), from_labels=tmp_path / "m.png", opaque=True)


def test_part_names_beyond_1023_are_protected_and_bmp_labels_refused(tmp_path):
    import pixelcut
    out = tmp_path / "o"
    out.mkdir()
    Image.new("RGB", (30, 20), "red").save(out / "part_1024.png")
    with pytest.raises(pixelcut.PixelcutError, match="split writes"):
        pixelcut.split(out / "part_1024.png", out, grid="2x2")
    assert (out / "part_1024.png").exists()
    Image.new("RGB", (40, 40), "red").save(tmp_path / "o.png")
    with pytest.raises(pixelcut.PixelcutError, match="lossless"):
        pixelcut.labels(tmp_path / "o.png", tmp_path / "m.bmp", grid="2x2")
    with pytest.raises(pixelcut.PixelcutError, match="at most"):
        pixelcut.split(tmp_path / "o.png", tmp_path / "w", angles=list(range(1025)), opaque=True)


def test_compare_bomb_is_a_pixelcut_error(tmp_path):
    import pixelcut
    Image.new("RGB", (20, 20), "red").save(tmp_path / "o.png")
    old = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 10
    try:
        with pytest.raises(pixelcut.PixelcutError):
            pixelcut.compare(tmp_path / "o.png", tmp_path / "o.png")
    finally:
        Image.MAX_IMAGE_PIXELS = old


def test_api_negative_angles_quiet_and_clean_errors(tmp_path, capsys):
    import pixelcut
    Image.new("RGBA", (40, 30), (255, 0, 0, 128)).save(tmp_path / "t.png")
    info = pixelcut.split(tmp_path / "t.png", tmp_path / "n", angles=[-90, 90], gap=0, edge=0, opaque=True)
    assert len(info["files"]) == 2
    pixelcut.alpha(tmp_path / "t.png", tmp_path / "a.png")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""                # not even the transparency note
    with pytest.raises(pixelcut.PixelcutError):
        pixelcut.compare(tmp_path / "missing.png", tmp_path / "t.png")
    assert type(pixelcut.compare(tmp_path / "t.png", tmp_path / "t.png")["ssim"]) is float


def test_join_refuses_lossy_output_and_foreign_zero_padded_parts(tmp_path):
    import pixelcut
    Image.fromarray((np.random.RandomState(4).rand(20, 30, 3) * 255).astype("uint8")).save(tmp_path / "o.png")
    pixelcut.split(tmp_path / "o.png", tmp_path / "p", grid="2x1", opaque=True)
    for bad in ("back.jpg", "back.webp", "back"):
        with pytest.raises(pixelcut.PixelcutError, match="lossless"):
            pixelcut.join(tmp_path / "p", tmp_path / bad)
        assert not (tmp_path / bad).exists()
    Image.new("RGB", (30, 20), "black").save(tmp_path / "p" / "part_01.png")      # not a name split writes
    (tmp_path / "p" / "parts.json").unlink()
    pixelcut.join(tmp_path / "p", tmp_path / "ok.png")
    assert pixelcut.compare(tmp_path / "o.png", tmp_path / "ok.png")["max_diff"] == 0


def test_lossy_label_map_is_refused(tmp_path):
    import pixelcut
    Image.new("RGB", (40, 40), "red").save(tmp_path / "o.png")
    Image.new("L", (40, 40), 0).save(tmp_path / "m.jpg", quality=90)
    with pytest.raises(pixelcut.PixelcutError, match="lossy"):
        pixelcut.split(tmp_path / "o.png", tmp_path / "x", from_labels=tmp_path / "m.jpg")


def test_version_flag_matches_pyproject_and_ignored_flags_are_announced(tmp_path):
    import pixelcut
    root = Path(__file__).resolve().parent.parent
    assert 'version = "%s"' % pixelcut.__version__ in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert pixelcut.__version__ in cli("--version").stdout
    Image.new("RGB", (40, 40), "red").save(tmp_path / "o.png")
    r = cli("split", tmp_path / "o.png", tmp_path / "g", "--grid", "2x2", "--gap", "20", "--centre", "5,5")
    assert "--centre, --gap ignored with --grid" in r.stderr


def test_fully_opaque_rgba_gives_no_transparency_note(tmp_path):
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(tmp_path / "solid.png")
    Image.new("RGBA", (10, 10), (255, 0, 0, 100)).save(tmp_path / "see.png")
    assert "transparency" not in cli("alpha", tmp_path / "solid.png", tmp_path / "a.png").stderr
    assert "transparency" in cli("alpha", tmp_path / "see.png", tmp_path / "b.png").stderr
