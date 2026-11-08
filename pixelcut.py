#!/usr/bin/env python3
"""K&K pixelcut - cut a picture into parts without changing a single pixel.

Most "recreate this image" tools redraw, trace or upscale, so the result is only similar. pixelcut does the opposite: it keeps
the original pixels and only decides which part each pixel belongs to. Put the parts back together over white and you get the
original back (max. deviation <= 1/255, checked by `--verify`).

Commands
  alpha   white background -> transparent (colour-to-alpha), exact over white
  split   cut into wedges around a centre, dividing lines at free angles, straight or zig-zag, with hit-area polygons
  join    put the parts back together over white (exact)
  compare SSIM + pixel difference of two images
  trace   (vtracer) redraw as SVG - approximate
  upscale (Real-ESRGAN, auto-download) AI upscale - invents pixels
  key     lift a foreground element (e.g. a word) out of a flat area, shrink it, put it in the middle, heal the hole

Run `python pixelcut.py COMMAND -h` for the options.
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PART_NAME = re.compile(r"part_\d+\.png")


# ----------------------------------------------------------------------------- helpers
def load_rgb(path):
    """8-bit RGB. Transparency is composited over white and 16-bit data is scaled to 8 bit - both with a notice, never silently."""
    im = Image.open(path)
    if im.width * im.height > 150_000_000:
        sys.exit("image too large (%d x %d = %d MP, limit 150 MP)" % (im.width, im.height, im.width * im.height // 1_000_000))
    if getattr(im, "n_frames", 1) > 1:
        print("note: %s has %d frames - only the first one is used" % (path, im.n_frames), file=sys.stderr)
    deep = None
    if Path(path).suffix.lower() in (".png", ".tif", ".tiff"):                            # Pillow cuts 48/64-bit PNG silently
        deep = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if deep is not None and deep.dtype == np.uint16 and deep.ndim == 3:
        print("note: %s is 16 bit per channel - scaled to 8 bit, so the result is not exact for the full range" % path, file=sys.stderr)
        rgb16 = deep[:, :, :3][:, :, ::-1].astype(np.float64) if deep.shape[2] >= 3 else np.dstack([deep[:, :, 0]] * 3).astype(np.float64)
        if deep.shape[2] == 4:                                                              # transparency over white, as for 8 bit
            al = deep[:, :, 3:4].astype(np.float64) / 65535.0
            rgb16 = rgb16 * al + 65535.0 * (1 - al)
        return np.clip(rgb16 / 257.0 + 0.5, 0, 255).astype(np.uint8)
    if im.mode == "CMYK":
        print("note: %s is CMYK - converted to RGB, colours are not exact" % path, file=sys.stderr)
    if im.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        print("note: %s is %s (more than 8 bit) - scaled to 8 bit, so the result is not exact for the full range" % (path, im.mode), file=sys.stderr)
        arr = np.array(im).astype(np.float64)
        top = 65535.0 if im.mode.startswith("I;16") else max(arr.max(), 1.0)
        return np.dstack([np.clip(arr / top * 255 + 0.5, 0, 255).astype(np.uint8)] * 3)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        print("note: %s has transparency - composited over white" % path, file=sys.stderr)
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.array(im.convert("RGB"))


def finite(values, what):
    if not all(np.isfinite(values)):
        sys.exit("%s must be finite numbers" % what)
    return values


def color_to_alpha(rgb):
    """RGBA such that compositing over white gives `rgb` back (standard colour-to-alpha against white)."""
    f = rgb.astype(np.float32) / 255.0
    a = np.clip(1.0 - f.min(2), 0, 1)
    col = np.clip((f - (1 - a[:, :, None])) / np.maximum(a, 1e-4)[:, :, None], 0, 1)
    col[a < 1e-3] = 0
    return np.dstack([col * 255, a * 255]).round().astype(np.uint8)


def over_white(rgba_list):
    h, w = rgba_list[0].shape[:2]
    acc = np.full((h, w, 3), 255.0, np.float32)
    for im in rgba_list:
        al = im[:, :, 3:4].astype(np.float32) / 255.0
        acc = acc * (1 - al) + im[:, :, :3].astype(np.float32) * al
    return acc


def tri(t):
    """Triangle wave -1..1 with sharp teeth, period 1."""
    t = np.mod(t, 1.0)
    return np.where(t < .5, 4 * t - 1, 3 - 4 * t)


def contour_polygon(mask, eps):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, eps, True)[:, 0, :]
    return [[int(x), int(y)] for x, y in ap]


def contour_polygons(mask, eps, min_area=4):
    """Every connected area as {outer, holes}: rings and islands are kept, unlike contour_polygon (largest outline only)."""
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    hier = hier[0]

    def ring(c):
        return [[int(x), int(y)] for x, y in cv2.approxPolyDP(c, eps, True)[:, 0, :]]
    for i, c in enumerate(cs):
        if hier[i][3] == -1 and cv2.contourArea(c) >= min_area:                       # outer ring; holes are its children
            holes = [ring(cs[j]) for j in range(len(cs)) if hier[j][3] == i and cv2.contourArea(cs[j]) >= min_area]
            out.append({"outer": ring(c), "holes": holes})
    return out


def fattest_point(mask):
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return [int(x), int(y)]


# ----------------------------------------------------------------------------- alpha
def cmd_alpha(a):
    rgb = load_rgb(a.input)
    rgba = color_to_alpha(rgb)
    Image.fromarray(rgba, "RGBA").save(a.output)
    if a.verify:
        err = np.abs(over_white([rgba]) - rgb.astype(np.float32)).max()
        print("max deviation over white: %.2f / 255" % err)


# ----------------------------------------------------------------------------- split
def split_sectors(shape, centre, angles, tooth, depth, gap, edge, region=None):
    """Sector id per pixel (0..n-1, sector i starts at angles[i]), slit mask and ink-edge mask.

    The dividing lines run from `centre` outwards at `angles` (degrees, screen angle: 0 = right, 90 = down). With depth > 0 each
    line is a zig-zag (teeth of period `tooth`, amplitude `depth`) instead of straight. Lines may be uneven (any angles).
    """
    h, w = shape
    cx, cy = centre
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    ang = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    order = sorted(range(len(angles)), key=lambda i: angles[i])
    ang_sorted = [angles[i] for i in order]
    n = len(ang_sorted)
    sector = np.zeros((h, w), np.int16)                       # sector k = from ang_sorted[k] to ang_sorted[k+1]
    for k in range(n):
        lo, hi = ang_sorted[k], ang_sorted[(k + 1) % n]
        m = (ang >= lo) & (ang < hi) if hi > lo else (ang >= lo) | (ang < hi)
        sector[m] = k
    slit_dist = np.full((h, w), 1e9, np.float32)
    for k in range(n):                                        # line k separates sector k-1 and k
        bang = ang_sorted[k]
        d = np.radians(((ang - bang + 180) % 360) - 180)
        along, perp = r * np.cos(d), r * np.sin(d)
        zig = depth * tri(along / tooth + .25 * k) if depth > 0 else 0.0
        near = (np.abs(perp) <= depth + gap / 2 + edge + 2) & (along > 0)      # a narrow band along the line, never a wide wedge
        if depth > 0:                                                          # straight lines: membership by angle is already exact
            sector = np.where(near, np.where(perp > zig, k, (k - 1) % n), sector)
        slit_dist = np.where(near, np.minimum(slit_dist, np.abs(perp - zig)), slit_dist)
    limit = region if region is not None else np.ones((h, w), bool)
    slit = (slit_dist < gap / 2) & limit
    edge_zone = (slit_dist < gap / 2 + edge) & ~slit & limit
    return sector, slit, edge_zone, order


def cmd_split(a):
    rgb = load_rgb(a.input)
    h, w = rgb.shape[:2]
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]) if a.opaque else color_to_alpha(rgb)
    if rgb.min(2).max() >= 250 and (rgb.min(2) < 250).sum() == 0:
        sys.exit("the image is completely white - nothing to cut")
    if a.centre == "auto":                                    # centroid of everything that is not white
        m = (rgb.min(2) < 250).astype(np.uint8)
        M = cv2.moments(m, True)
        centre = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    else:
        try:
            centre = tuple(float(v) for v in a.centre.split(","))
            assert len(centre) == 2
        except (ValueError, AssertionError):
            sys.exit("--centre must look like 300,240 (x,y) or 'auto'")
    finite(list(centre), "--centre")
    if a.simplify < 0 or not np.isfinite(a.simplify):
        sys.exit("--simplify must be 0 or greater")
    try:
        angles = [float(v) for v in a.angles.split(",")]
        finite(angles, "--angles")
        angles = [v % 360.0 for v in angles]                  # 360 means 0, -90 means 270
    except ValueError:
        sys.exit("--angles must be numbers separated by commas, e.g. 32,148,214,327")
    if len(angles) < 2 or len({round(v % 360, 6) for v in angles}) != len(angles):
        sys.exit("--angles needs at least two different angles")
    for flag, value in (("--gap", a.gap), ("--edge", a.edge), ("--depth", a.depth), ("--tooth", a.tooth)):
        if not np.isfinite(value) or value < 0:
            sys.exit("%s must be a finite number, 0 or greater" % flag)
    if a.gap > 100 or a.edge > 100 or a.depth > 1000:
        sys.exit("--gap/--edge/--depth are in pixels; these values are far too large")
    if a.depth > 0 and a.tooth <= 0:
        sys.exit("--tooth must be greater than 0 (or use --depth 0 for straight lines)")
    region = None
    if a.only_inside:                                         # cut only inside the largest non-white blob, leave satellites whole
        m = (rgb.min(2) < 250).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        region = cv2.dilate((lab == big).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    sector, slit, edge_zone, order = split_sectors((h, w), centre, angles, a.tooth, a.depth, a.gap, a.edge, region)
    if region is not None:                                    # satellites stay whole: one sector per connected area outside the main blob
        content = cv2.morphologyEx((rgb.min(2) < 250).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        nl, lab, _, _ = cv2.connectedComponentsWithStats(np.where(region, 0, content).astype(np.uint8))
        for c in range(1, nl):
            cc = lab == c
            sector[cc] = np.bincount(sector[cc].ravel()).argmax()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    old_manifest = out / "parts.json"                        # only files that an earlier pixelcut run listed may be replaced
    ours = set()
    if old_manifest.exists():
        try:
            listed = json.loads(old_manifest.read_text()).get("files", [])
            ours = {n for n in listed if isinstance(n, str) and PART_NAME.fullmatch(n)}
        except Exception:                                    # damaged manifest: treat the folder as foreign
            ours = set()
    clash = [("part_%d.png" % k) for k in range(len(angles)) if (out / ("part_%d.png" % k)).exists() and ("part_%d.png" % k) not in ours]
    if clash:
        sys.exit("%s already exists in %s and was not written by pixelcut - use an empty folder" % (", ".join(clash), out))
    staging = out / ".pixelcut_new"                           # everything is written here first, replacing happens only when all parts worked
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    ink = np.array([20, 16, 14], np.uint8)
    parts, info = [], []
    for k in range(len(angles)):
        img = rgba.copy()
        img[(sector != k) | slit] = 0
        if a.edge > 0:
            e = edge_zone & (sector == k) & (img[:, :, 3] > 0)
            img[e, :3] = ink
            img[e, 3] = 255
        parts.append(img)
        Image.fromarray(img, "RGBA").save(staging / ("part_%d.png" % k))
        solid = ((sector == k) & ~slit & (rgba[:, :, 3] > 8)).astype(np.uint8)
        info.append({"part": k, "starts_at_angle": sorted(angles)[k],
                     "polygon": contour_polygon(solid, a.simplify), "polygons": contour_polygons(solid, a.simplify),
                     "fattest_point": fattest_point(solid)})
    files = ["part_%d.png" % k for k in range(len(parts))]
    (staging / "parts.json").write_text(json.dumps({"size": [w, h], "centre": list(centre), "files": files, "parts": info}, indent=1))
    for name in ours:                                         # all parts exist - now swap old for new
        if (out / name).is_file():
            (out / name).unlink()
    for name in files + ["parts.json"]:
        os.replace(staging / name, out / name)
    shutil.rmtree(staging)
    print("wrote %d parts + parts.json to %s" % (len(parts), out))
    if a.verify:
        # without slit and edge the parts must add up to the original over white
        flat = [rgba.copy() for _ in parts]
        for k, im in enumerate(flat):
            im[sector != k] = 0
        err = np.abs(over_white(flat) - rgb.astype(np.float32)).max()
        print("max deviation over white without slit: %.2f / 255" % err)
        real = over_white([np.array(Image.open(out / ("part_%d.png" % k)).convert("RGBA")) for k in range(len(parts))])
        real_err = np.abs(real - rgb.astype(np.float32)).max()
        print("saved files joined over white:        %.2f / 255 (with --gap %.1f --edge %.1f the slit is visible on purpose; 0 slit = exact)" % (real_err, a.gap, a.edge))


# ----------------------------------------------------------------------------- key
def cmd_key(a):
    rgb = load_rgb(a.input)
    h, w = rgb.shape[:2]
    try:
        x0, y0, x1, y1 = (int(v) for v in a.box.split(","))
    except ValueError:
        sys.exit("--box must look like x0,y0,x1,y1 (four whole numbers)")
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        sys.exit("--box %s does not lie inside the image (%dx%d) or is empty/reversed" % (a.box, w, h))
    if not (np.isfinite(a.scale) and 0 < a.scale <= 10):
        sys.exit("--scale must be greater than 0 and at most 10")
    if a.close < 1 or a.close > 99 or not np.isfinite(a.threshold) or a.threshold < 0:
        sys.exit("--close must be 1..99 and --threshold 0 or greater")
    roi = rgb[y0:y1, x0:x1].astype(np.float32)
    ring = np.concatenate([roi[:3].reshape(-1, 3), roi[-3:].reshape(-1, 3), roi[:, :3].reshape(-1, 3), roi[:, -3:].reshape(-1, 3)])
    fill = np.median(ring, 0)                                  # the flat colour the element sits on
    diff = np.abs(roi - fill).max(2)
    fg = (diff > a.threshold).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close, a.close)))
    cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fg = np.zeros_like(fg)
    cv2.drawContours(fg, cs, -1, 1, -1)                        # holes (counters, highlights) filled
    ys, xs = np.nonzero(fg)
    if len(xs) == 0:
        sys.exit("nothing stands out from the background inside --box (try a lower --threshold)")
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    # heal the hole with the flat colour, softly blended
    soft = cv2.GaussianBlur(cv2.dilate(fg, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.2)[:, :, None]
    healed = rgb.astype(np.float32).copy()
    healed[y0:y1, x0:x1] = roi * (1 - soft) + fill * soft
    # the element: exact pixels, shrunk, centred on (cx, cy)
    try:
        cx, cy = (float(v) for v in a.to.split(",")) if a.to else ((x0 + x1) / 2, (y0 + y1) / 2)
    except ValueError:
        sys.exit("--to must look like x,y")
    finite([cx, cy], "--to")
    el = rgb[y0:y1, x0:x1][by0:by1 + 1, bx0:bx1 + 1]
    al = cv2.GaussianBlur(fg[by0:by1 + 1, bx0:bx1 + 1].astype(np.float32), (0, 0), .8)
    kw, kh = max(1, int(round(el.shape[1] * a.scale))), max(1, int(round(el.shape[0] * a.scale)))
    el = cv2.resize(el, (kw, kh), interpolation=cv2.INTER_AREA)
    al = cv2.resize(al, (kw, kh), interpolation=cv2.INTER_AREA)
    layer = np.zeros((h, w, 4), np.uint8)
    px, py = int(round(cx - kw / 2)), int(round(cy - kh / 2))
    if px < 0 or py < 0 or px + kw > w or py + kh > h:
        sys.exit("the element would leave the image at --to %s (needs %dx%d px around that point)" % (a.to, kw, kh))
    layer[py:py + kh, px:px + kw, :3] = el
    layer[py:py + kh, px:px + kw, 3] = np.clip(al * 255, 0, 255).astype(np.uint8)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(healed, 0, 255).round().astype(np.uint8)).save(out / "base.png")
    Image.fromarray(layer, "RGBA").save(out / "element.png")
    comp = healed.copy()
    al3 = layer[:, :, 3:4].astype(np.float32) / 255
    comp = comp * (1 - al3) + layer[:, :, :3].astype(np.float32) * al3
    Image.fromarray(np.clip(comp, 0, 255).round().astype(np.uint8)).save(out / "preview.png")
    print("wrote base.png (hole healed), element.png (exact pixels, scale %.2f), preview.png" % a.scale)



# ----------------------------------------------------------------------------- join / compare
def cmd_join(a):
    """Put parts back together (over white or a colour) - the way to 'rebuild' a picture from its exact pieces."""
    manifest = Path(a.parts) / "parts.json"
    size = None
    if manifest.exists():                                     # the manifest says which parts belong together - all of them are needed
        try:
            m = json.loads(manifest.read_text())
            n = len(m["parts"])
            size = tuple(m["size"])
            assert len(size) == 2 and all(isinstance(v, int) and v > 0 for v in size) and n > 0
            names = m.get("files") or ["part_%d.png" % k for k in range(n)]
            assert len(names) == n and all(isinstance(x, str) and PART_NAME.fullmatch(x) for x in names)
        except (ValueError, KeyError, TypeError, AssertionError, AttributeError):
            sys.exit("parts.json is damaged (needs 'size' [w,h], 'parts' and matching 'files')")
        files = [Path(a.parts) / x for x in names]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            sys.exit("parts.json lists %d parts but these are missing: %s" % (n, ", ".join(missing)))
    else:
        files = sorted(f for f in Path(a.parts).glob("part_*.png") if PART_NAME.fullmatch(f.name))
        numbers = sorted(int(f.stem.split("_")[1]) for f in files)
        gaps = [k for k in range(numbers[-1] + 1) if k not in numbers] if numbers else []
        if gaps:
            sys.exit("part numbering has a gap (missing part_%d.png) - without parts.json every part is needed" % gaps[0])
    if not files:
        sys.exit("no part_*.png in %s" % a.parts)
    ims = [np.array(Image.open(f).convert("RGBA")) for f in files]
    shapes = {im.shape[:2] for im in ims}
    if len(shapes) != 1 or (size and (size[1], size[0]) not in shapes):
        sys.exit("the parts do not all have the same size%s" % (" as parts.json says" if size else ""))
    out = over_white(ims)
    Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8)).save(a.output)
    print("joined %d parts -> %s" % (len(ims), a.output))


def measure(x, y):
    """(SSIM on grey, mean abs diff, max abs diff) of two same-size RGB arrays."""
    from skimage.metrics import structural_similarity
    gx = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    gy = cv2.cvtColor(y, cv2.COLOR_RGB2GRAY)
    d = np.abs(x.astype(np.float32) - y.astype(np.float32))
    win = min(7, min(gx.shape) // 2 * 2 - 1 if min(gx.shape) % 2 == 0 else min(gx.shape))
    ss = structural_similarity(gx, gy, win_size=win) if win >= 3 else float("nan")   # tiny images: no SSIM window
    return ss, float(d.mean()), float(d.max())


def cmd_compare(a):
    x, y = load_rgb(a.first), load_rgb(a.second)
    if x.shape != y.shape:
        if not a.resize:
            sys.exit("the images have different sizes (%dx%d vs %dx%d) - use --resize to scale the second one" % (x.shape[1], x.shape[0], y.shape[1], y.shape[0]))
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
    ss, mean, mx = measure(x, y)
    print("SSIM %.4f   mean |diff| %.3f   max |diff| %d" % (ss, mean, mx))


# ----------------------------------------------------------------------------- trace / upscale (optional external engines)
def find_chrome():
    import shutil
    for c in ("chrome", "google-chrome", "chromium", "msedge", "brave"):
        w = shutil.which(c)
        if w:
            return w
    for c in (r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if Path(c).exists():
            return c
    return None


def cmd_trace(a):
    """Redraw the picture as vector shapes (vtracer). Approximate by nature: use `compare` to see how far it is from the original."""
    import vtracer
    if not Path(a.input).is_file():
        sys.exit("error: %s does not exist" % a.input)
    if not (0 <= a.speckle <= 128 and 1 <= a.colors <= 8 and 0 <= a.layers <= 255):
        sys.exit("--speckle must be 0..128, --colors 1..8, --layers 0..255")
    if not Path(a.output).resolve().parent.is_dir():
        sys.exit("error: the folder for %s does not exist" % a.output)
    vtracer.convert_image_to_svg_py(a.input, a.output, colormode="color", hierarchical="stacked", mode="spline",
                                    filter_speckle=a.speckle, color_precision=a.colors, layer_difference=a.layers,
                                    corner_threshold=60, length_threshold=3.5, splice_threshold=45, path_precision=3)
    print("wrote", a.output)
    if a.render:
        import subprocess
        import tempfile
        chrome = find_chrome()
        if not chrome:
            sys.exit("--render needs Chrome/Edge/Brave")
        w, h = Image.open(a.input).size
        t = Path(tempfile.mkdtemp())
        (t / "p.html").write_text('<html><body style="margin:0;background:#fff"><img src="%s" width=%d height=%d></body></html>'
                                  % (Path(a.output).resolve().as_uri(), w, h))
        render_target = Path(a.render).resolve()
        if render_target.exists():
            render_target.unlink()
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                        "--user-data-dir=" + str(t / "prof"), "--window-size=%d,%d" % (w, h), "--screenshot=" + str(Path(a.render).resolve()),
                        (t / "p.html").as_uri()], capture_output=True, timeout=120)
        if not Path(a.render).exists():
            sys.exit("error: Chrome did not produce %s" % a.render)
        print("rendered", a.render)


ESRGAN_SHA256 = {"windows": "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d",
                 "ubuntu": "e5aa6eb131234b87c0c51f82b89390f5e3e642b7b70f2b9bbe95b6a285a40c96",
                 "macos": "e0ad05580abfeb25f8d8fb55aaf7bedf552c375b5b4d9bd3c8d59764d2cc333a"}
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-%s.zip"


def ensure_esrgan():
    """Find realesrgan-ncnn-vulkan; if it is missing, download the official release once into ~/.cache/pixelcut.
    Returns None when this machine cannot run it (e.g. ARM Linux)."""
    import platform
    import shutil
    import urllib.request
    import zipfile
    found = shutil.which("realesrgan-ncnn-vulkan")
    if found:
        return found
    osname = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "ubuntu")
    if platform.machine().lower() not in ("amd64", "x86_64") and osname != "macos":
        return None
    exename = "realesrgan-ncnn-vulkan" + (".exe" if osname == "windows" else "")
    cache = Path.home() / ".cache" / "pixelcut" / "realesrgan"
    exe = next(cache.rglob(exename), None) if cache.exists() else None
    if exe:
        return str(exe)
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / "release.zip"
    print("realesrgan-ncnn-vulkan not found - downloading the official release (~50 MB, once) ...")
    try:
        import hashlib
        urllib.request.urlretrieve(ESRGAN_URL % osname, zpath)
        if hashlib.sha256(zpath.read_bytes()).hexdigest() != ESRGAN_SHA256[osname]:
            zpath.unlink()
            print("download rejected: checksum does not match the known Real-ESRGAN release")
            return None
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():                       # no path may leave the cache folder
                if not (cache / member).resolve().is_relative_to(cache.resolve()):
                    raise ValueError("unsafe path in archive: %s" % member)
            z.extractall(cache)
        zpath.unlink()
    except Exception as e:
        print("download failed (%s)" % e)
        return None
    exe = next(cache.rglob(exename), None)
    if exe and osname != "windows":
        exe.chmod(0o755)
    return str(exe) if exe else None


def lanczos_upscale(a, why):
    """Fallback without AI: plain Lanczos resize. Smooth, but no invented detail."""
    print("Real-ESRGAN not usable here (%s) - using plain Lanczos resize instead (no AI detail)." % why)
    im = Image.open(a.input)
    im = im.convert("RGBA" if (im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info) else "RGB")   # keep transparency
    im.resize((im.width * a.scale, im.height * a.scale), Image.LANCZOS).save(a.output)
    print("wrote", a.output)


def cmd_upscale(a):
    """AI upscale with Real-ESRGAN (realesrgan-ncnn-vulkan). New pixels are invented: sharper, but not exact any more.
    Falls back to Lanczos where the engine cannot run (ARM, no Vulkan GPU)."""
    import subprocess
    if not 1 <= a.scale <= 8:
        sys.exit("--scale must be a whole number from 1 to 8")
    exe = a.bin or ensure_esrgan()
    if not exe:
        return lanczos_upscale(a, "no build for this machine")
    try:
        subprocess.run([exe, "-i", str(Path(a.input).resolve()), "-o", str(Path(a.output).resolve()), "-n", a.model, "-s", str(a.scale)],
                       check=True, cwd=str(Path(exe).parent), capture_output=True)
    except OSError as e:
        return lanczos_upscale(a, "cannot start: %s" % e)
    except subprocess.CalledProcessError:
        return lanczos_upscale(a, "needs a Vulkan GPU/driver")
    print("wrote", a.output)


# ----------------------------------------------------------------------------- cli
def main(argv=None):
    p = argparse.ArgumentParser(prog="pixelcut", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("alpha", help="white background -> transparent, exact over white")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--verify", action="store_true", help="print the max deviation over white")
    s.set_defaults(fn=cmd_alpha)

    s = sub.add_parser("split", help="cut into wedges with straight or zig-zag dividing lines")
    s.add_argument("input"); s.add_argument("outdir")
    s.add_argument("--centre", default="auto", help="x,y of the centre or 'auto' (centroid of the non-white area)")
    s.add_argument("--angles", default="45,135,225,315", help="angles of the dividing lines in degrees (0 = right, 90 = down); free, e.g. 32,148,214,327")
    s.add_argument("--tooth", type=float, default=18.0, help="zig-zag period in px")
    s.add_argument("--depth", type=float, default=9.0, help="zig-zag amplitude in px, 0 = straight lines")
    s.add_argument("--gap", type=float, default=2.6, help="width of the slit in px")
    s.add_argument("--edge", type=float, default=3.2, help="thin ink edge along the slit in px, 0 = none")
    s.add_argument("--only-inside", action="store_true", help="cut only the biggest blob, keep satellites (clouds, dots) whole")
    s.add_argument("--simplify", type=float, default=2.0, help="polygon simplification in px (for hit areas)")
    s.add_argument("--opaque", action="store_true", help="photos: keep every pixel opaque, no white-to-alpha")
    s.add_argument("--verify", action="store_true")
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("key", help="lift a foreground element out of a flat area, shrink and centre it, heal the hole")
    s.add_argument("input"); s.add_argument("outdir")
    s.add_argument("--box", required=True, help="x0,y0,x1,y1 box around the element; its border must be flat background")
    s.add_argument("--scale", type=float, default=.55)
    s.add_argument("--to", help="x,y where the element is centred (default: centre of the box)")
    s.add_argument("--threshold", type=float, default=40, help="colour distance from the flat background that counts as element")
    s.add_argument("--close", type=int, default=9, help="closing kernel px, joins letters and fills gaps")
    s.set_defaults(fn=cmd_key)

    s = sub.add_parser("join", help="put parts_dir/part_*.png back together over white")
    s.add_argument("parts"); s.add_argument("output")
    s.set_defaults(fn=cmd_join)

    s = sub.add_parser("compare", help="SSIM and pixel difference of two images")
    s.add_argument("first"); s.add_argument("second")
    s.add_argument("--resize", action="store_true", help="scale the second image to the size of the first (otherwise different sizes are an error)")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("trace", help="redraw as SVG (vtracer): approximate")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--render", help="also render the SVG to this PNG (needs Chrome/Edge/Brave)")
    s.add_argument("--speckle", type=int, default=2); s.add_argument("--colors", type=int, default=7); s.add_argument("--layers", type=int, default=8)
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("upscale", help="AI upscale (Real-ESRGAN, downloaded automatically on first use): invents pixels")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--scale", type=int, default=4); s.add_argument("--model", default="realesrgan-x4plus-anime")
    s.add_argument("--bin", help="path to realesrgan-ncnn-vulkan (default: automatic)")
    s.set_defaults(fn=cmd_upscale)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as e:                                # includes pyo3 PanicException, which is not an Exception
        if os.environ.get("PIXELCUT_DEBUG"):
            raise
        sys.exit("error: %s (set PIXELCUT_DEBUG=1 for the full trace)" % (str(e).strip().splitlines() or [type(e).__name__])[0])


if __name__ == "__main__":
    sys.exit(main())
