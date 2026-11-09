"""Second ring of tests: analytic geometry (close and odd angles), the numbers printed by --verify, gap/edge widths, input
conversions with known results, key/trace/upscale details, manifest handling and the download guard."""
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
                        "--opaque", "--angles", "0,180", "--gap", "0", "--edge", "0"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert (target / "parts.json").exists()
