#!/usr/bin/env python3
"""K&K pixelcut - cut a picture into parts without changing a single pixel.

Most "recreate this image" tools redraw, trace or upscale, so the result is only similar. pixelcut does the opposite: it keeps
the original pixels and only decides which part each pixel belongs to. Put the parts back together over white and you get the
original back (max. deviation <= 1/255, checked by `--verify`).

Commands
  alpha   white background -> transparent (colour-to-alpha), exact over white
  split   cut into wedges around a centre (free angles, straight or zig-zag), into a grid / strips (--grid 4x3) or along a label map
          (--from-labels, any shape), with hit-area polygons
  labels  write a label map: ONE picture in which every pixel carries the number of its part (--grid 4x3)
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
import stat
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


__version__ = "1.6.19"
PART_NAME = re.compile(r"part_(0|[1-9]\d*)\.png")                  # exactly the names split writes: no leading zeros
LOSSLESS = (".png", ".tif", ".tiff")


def manifest_files(m):
    """Validate a pixelcut manifest and return its part file names. Raises ValueError for anything that is not one.
    Manifests from v1.0/1.1 have no 'files' list: the names are then part_0..part_N-1."""
    if not isinstance(m, dict) or not isinstance(m.get("parts"), list) or not m["parts"]:
        raise ValueError("no parts")
    size = m.get("size")
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)):
        raise ValueError("bad size")
    c = m.get("centre")
    if not (isinstance(c, list) and len(c) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in c)):
        raise ValueError("bad centre")
    if not all(isinstance(x, dict) and isinstance(x.get("part"), int) and isinstance(x.get("starts_at_angle"), (int, float)) for x in m["parts"]):
        raise ValueError("bad part entries")
    names = m.get("files") or ["part_%d.png" % k for k in range(len(m["parts"]))]
    if not (isinstance(names, list) and len(names) == len(m["parts"]) == len(set(names))
            and all(isinstance(x, str) and PART_NAME.fullmatch(x) for x in names)):
        raise ValueError("bad file list")
    return names


JOURNAL = ".journal.json"


def link_like(path):
    """Symlink or Windows junction/reparse point: it points somewhere else, so whatever we do 'in' it happens elsewhere."""
    st = os.lstat(path)
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def real_folder(folder, what):
    if folder.exists() and (link_like(folder) or not folder.is_dir()):
        sys.exit("%s is a link or not a folder - pixelcut will not work inside it (%s)" % (folder, what))


def scratch_beside(path):
    """A fresh private folder next to the target (same disk, so the final rename is atomic). Returns (folder, file inside it
    with the target's name). Nothing that already exists can be hit: the folder name is unique and created by us."""
    import tempfile
    path = Path(path).resolve()
    d = Path(tempfile.mkdtemp(prefix=".pixelcut_", dir=path.parent))
    return d, d / path.name


def _lock(f):
    """Try to take an exclusive, non-blocking lock on byte 0 of an open file: 'ok', 'busy' (somebody else holds it) or
    'unavailable' (this system cannot lock, e.g. no fcntl/msvcrt)."""
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return "ok"
    except ImportError:
        return "unavailable"
    except OSError as e:
        import errno
        return "busy" if e.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK, getattr(errno, "EWOULDBLOCK", errno.EAGAIN)) else "unavailable"


def lock_folder(folder, held, quiet=False, busy=None):
    """One pixelcut run at a time may change the files of a folder. The lock lives as long as the process holds the open file, so
    a crashed run never leaves a stale lock behind. Returns True if the lock is held; False where locking does not work (then
    nothing can be told about other runs). Another live run: exit with a clear message. `held` collects the handle to close."""
    path = Path(folder) / ".pixelcut_lock"
    try:
        f = open(path, "a+b")
    except OSError:                                           # a lock file we cannot open (read-only folder/file): work without it, and say so
        if quiet:
            return False
        print("note: cannot create the lock file %s - two runs at once in this folder would not be noticed" % path, file=sys.stderr)
        return False
    held.append(f)
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b"1")
        f.flush()
        if os.name == "nt":                                   # not a file the person should see or delete by accident
            try:
                import ctypes
                ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x2)
            except Exception:
                pass
    state = _lock(f)
    if state == "busy":
        sys.exit(busy or "error: another pixelcut run is working in %s right now - wait until it has finished" % folder)
    if state == "unavailable" and sys.platform != "emscripten" and not quiet:
        print("note: this file system cannot lock folders - two pixelcut runs at once in %s would not be noticed" % folder, file=sys.stderr)
    return state == "ok"


def folder_is_busy(folder):
    """True if a live pixelcut run holds the folder lock. Creates nothing and needs no write access."""
    path = Path(folder) / ".pixelcut_lock"
    if not path.is_file():
        return False
    try:
        with open(path, "rb") as probe:
            return _lock(probe) == "busy"
    except OSError:
        return False


def _write_atomic_text(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def publish(pairs):
    """Replace several finished files at once: [(finished_file, destination), ...]. Existing destinations are moved aside first and
    put back if anything goes wrong, so either all new files are in place or all old ones are. The aside folder is the truth about
    what has to be restored and it is deleted only when nothing is left to restore. A folder lock keeps two runs apart."""
    held = []
    try:
        _publish(pairs, held)
    finally:
        for f in held:
            f.close()


def _publish(pairs, held):
    import tempfile
    folder = Path(pairs[0][1]).resolve().parent
    locked = lock_folder(folder, held)
    recover_publish(folder, locked)                           # a hard abort of an earlier run is finished first
    for _, dest in pairs:
        if os.path.lexists(dest) and (link_like(dest) or not Path(dest).is_file()):
            sys.exit("error: %s exists and is not a plain file (folder or link) - nothing was changed" % dest)
        if Path(dest).exists() and not os.access(dest, os.W_OK):
            sys.exit("error: %s is read-only - nothing was changed" % dest)
    aside = Path(tempfile.mkdtemp(prefix=".pixelcut_", dir=folder))
    installed = []
    try:
        _write_atomic_text(aside / "publish.json", json.dumps({"dests": [str(Path(d).resolve()) for _, d in pairs]}))   # lets a later run undo an abort
        for i, (_, dest) in enumerate(pairs):
            if Path(dest).exists():
                os.replace(dest, aside / str(i))
        (aside / "installing").write_text("1")                # from here on every destination that exists is a new file
        for src, dest in pairs:
            installed.append(dest)                            # before the move: an abort in between must still clean up
            os.replace(src, dest)
        (aside / "committed").write_text("1")                 # the new set is complete: a later run only removes the backup
    except BaseException:
        stuck = []
        for dest in installed:
            try:
                Path(dest).unlink(missing_ok=True)
            except OSError as e:
                stuck.append("%s (%s)" % (dest, e))
        for f in [f for f in aside.iterdir() if f.name.isdigit()]:
            try:
                os.replace(f, pairs[int(f.name)][1])
            except OSError as e:
                stuck.append("%s (%s)" % (pairs[int(f.name)][1], e))
        if stuck:                                             # never destroy the only copy: say where it is
            print("error: could not restore everything - your previous files are kept in %s\n  problem with: %s" % (aside, "; ".join(stuck)), file=sys.stderr)
        else:
            shutil.rmtree(aside, ignore_errors=True)
        raise
    shutil.rmtree(aside, ignore_errors=True)


def recover_publish(folder, locked=True):
    """Finish what a hard abort (power cut, kill) left behind in publish(): a backup folder with publish.json beside the files.
    Called under the folder lock, so no live run can own such a folder. Without a working lock (`locked` False) nothing is
    touched - a live run could own it - and the person is told. Only files directly inside `folder` are ever written.
    'committed' = the new set is complete, drop the backup; otherwise go back to the old set (delete the new files if the
    install phase had started, then put the backups back). A backup that was not restored is never thrown away."""
    folder = Path(folder).resolve()
    for d in sorted(folder.glob(".pixelcut_*")):
        meta = d / "publish.json"
        if link_like(d) or not d.is_dir():
            continue
        if not meta.is_file():
            if locked and [f.name for f in d.iterdir()] == ["publish.json.tmp"]:     # died while writing its note: nothing in here to restore
                shutil.rmtree(d, ignore_errors=True)
            continue
        if not locked:
            print("note: %s may be the backup of an interrupted run, but this system cannot tell whether it is still running - left untouched" % d, file=sys.stderr)
            continue
        try:
            dests = json.loads(meta.read_text(encoding="utf-8"))["dests"]
            assert isinstance(dests, list) and all(isinstance(x, str) for x in dests)
            assert all(Path(x).resolve().parent == folder for x in dests)      # a renamed folder must not make us write elsewhere
        except (OSError, ValueError, KeyError, AssertionError):
            if not any(f.name.isdigit() for f in d.iterdir()):                 # nothing to restore: a half-written note only
                shutil.rmtree(d, ignore_errors=True)
            else:
                print("note: %s looks like a backup of an interrupted run but does not belong here - left untouched" % d, file=sys.stderr)
            continue
        left = False
        if not (d / "committed").exists():
            if (d / "installing").exists():
                for x in dests:
                    Path(x).unlink(missing_ok=True)
            restored = 0
            for f in [f for f in d.iterdir() if f.name.isdigit()]:
                if int(f.name) < len(dests):
                    os.replace(f, dests[int(f.name)])
                    restored += 1
                else:
                    left = True
            if restored:
                print("note: finished an interrupted earlier run in %s - the previous files are back" % folder, file=sys.stderr)
            if left:
                print("note: %s still holds backups that belong to no file in the list - kept" % d, file=sys.stderr)
        if not left:
            shutil.rmtree(d, ignore_errors=True)


def refuse_inputs(inputs, outputs):
    """An output must never be one of the input files - writing it would destroy what is being read."""
    ins = {Path(x).resolve() for x in inputs}
    for o in outputs:
        if Path(o).resolve() in ins:
            sys.exit("error: %s is also an input - choose a different output file" % o)


def refuse_inputs_in_workfolders(inputs, outdir):
    """split keeps its scratch/backup folders inside outdir and clears them during recovery: an input that lives there would be
    read first and then deleted or moved. Refuse that before anything is touched."""
    work = [(Path(outdir) / n).resolve() for n in (DONE, ".pixelcut_old", ".pixelcut_new")]
    for x in inputs:
        rx = Path(x).resolve()
        if rx.parent == Path(outdir).resolve() and (PART_NAME.fullmatch(rx.name) or rx.name == "parts.json"):
            sys.exit("error: %s is one of the files split writes into %s - choose a different input" % (x, outdir))
        for wf in work:
            if rx == wf or wf in rx.parents:
                sys.exit("error: %s lies in %s, a work folder of pixelcut - move the picture elsewhere" % (x, wf))


def rmtree_force(folder):
    """Delete a folder of ours even if a file in it is read-only (a read-only part from an earlier run blocks rmtree on Windows,
    and this runs after the commit point, where a failure would leave the folder unusable)."""
    for root, _, files in os.walk(folder):
        for f in files:
            try:
                os.chmod(os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
    shutil.rmtree(folder)


def need_picture(path):
    """An input must be an existing file: say so in plain words instead of an operating-system error."""
    if Path(path).is_dir():
        sys.exit("error: %s is a folder, not a picture" % path)
    if not Path(path).exists():
        sys.exit("error: %s does not exist" % path)


def need_folder_of(path):
    """An output must have a folder to go to and, if it already exists, be writable - before anything is touched."""
    if not Path(path).resolve().parent.is_dir():
        sys.exit("error: the folder for %s does not exist" % path)
    if Path(path).is_file() and not os.access(path, os.W_OK):
        sys.exit("error: %s is read-only - nothing was changed" % path)


def oserror_message(e):
    """Plain wording for the operating-system errors people actually meet; None for anything else."""
    names = [n for n in (getattr(e, "filename2", None), getattr(e, "filename", None)) if n and ".pixelcut" not in str(n)]
    name = names[0] if names else (e.filename or "a file")
    if isinstance(e, FileNotFoundError):
        return "error: cannot find %s" % name
    if isinstance(e, IsADirectoryError):
        return "error: %s is a folder, not a picture" % name
    if isinstance(e, (FileExistsError, NotADirectoryError)):
        return "error: %s is a file, not a folder (or the other way round) - choose another path" % name
    if isinstance(e, PermissionError):
        if Path(str(name)).is_dir():
            return "error: %s is a folder, not a file - choose another path" % name
        return "error: cannot write %s (read-only, or in use by another program)" % name
    return None


def flatten_white(img):
    flat = Image.new("RGB", img.size, "white")
    flat.paste(img.convert("RGBA"), mask=img.convert("RGBA").getchannel("A"))
    return flat


def save_atomic(img, path):
    """Write beside the target, replace only when complete: an abort never leaves half a picture in place of a good one."""
    need_folder_of(path)
    ext = Path(path).suffix.lower()
    Image.init()
    if Image.registered_extensions().get(ext) not in Image.SAVE:
        sys.exit("error: cannot write %s - use .png, .tif, .jpg, .webp or .bmp" % (ext or "a file without an ending"))
    if ext in (".jpg", ".jpeg", ".bmp") and img.mode in ("RGBA", "LA", "PA"):  # these have no (usable) transparency: flatten over white, say so
        print("note: %s cannot hold transparency - flattened over white" % path, file=sys.stderr)
        img = flatten_white(img)
    d, tmp = scratch_beside(path)
    try:
        img.save(tmp)
        os.replace(tmp, Path(path).resolve())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def foreign_entries(folder, extra=()):
    """Names in a pixelcut scratch/backup folder that pixelcut did not put there. Only plain files with pixelcut's own
    names count as ours - a directory or a link called part_0.png is foreign, whatever it is named."""
    bad = []
    for f in folder.iterdir():
        own = (PART_NAME.fullmatch(f.name) or f.name in ("parts.json", JOURNAL, JOURNAL + ".tmp") or f.name in extra)
        if not own or link_like(f) or not f.is_file():
            bad.append(f.name)
    return bad
DONE = ".pixelcut_done"


def write_journal(folder, names):
    """Atomic: a crash leaves either the old or the new journal, never half a file."""
    tmp = folder / (JOURNAL + ".tmp")
    tmp.write_text(json.dumps(names))
    os.replace(tmp, folder / JOURNAL)


# ----------------------------------------------------------------------------- helpers
def open_upright(path):
    """Image.open plus the camera's rotation: a phone photo is stored sideways with an 'Orientation' tag, viewers turn it - so do we
    (with a notice), otherwise the result would not look like what the person sees."""
    need_picture(path)
    im = Image.open(path)
    try:
        orient = im.getexif().get(274, 1)
    except Exception:
        orient = 1
    if orient not in (1, None, 0):
        print("note: %s is rotated by its EXIF orientation tag (%s) - turned upright like a viewer shows it" % (path, orient), file=sys.stderr)
        im = ImageOps.exif_transpose(im)
    im.pixelcut_orientation = orient if orient in (2, 3, 4, 5, 6, 7, 8) else 1
    return im


def decoded_8bit(path):
    """The picture as 8-bit pixels the way every other command sees it (upright, 16 bit scaled with a notice, limits checked).
    Keeps transparency (RGBA) where the file has some; otherwise RGB. For tools that read a file themselves (vtracer, the upscale engine):
    they get exactly these pixels instead of reading the original a second time with their own, different decoder."""
    need_picture(path)
    with Image.open(path) as probe:
        has_alpha = ("A" in probe.getbands() or "transparency" in probe.info) and probe.mode not in ("I;16", "I;16L", "I;16B", "I", "F")
    if not has_alpha:
        return Image.fromarray(load_rgb(path))
    pic = open_upright(path)
    if pic.width * pic.height > 150_000_000:
        sys.exit("image too large (%d x %d = %d MP, limit 150 MP)" % (pic.width, pic.height, pic.width * pic.height // 1_000_000))
    if getattr(pic, "n_frames", 1) > 1:
        print("note: %s has %d frames - only the first one is used" % (path, pic.n_frames), file=sys.stderr)
    if Path(path).suffix.lower() in (".png", ".tif", ".tiff"):                            # Pillow cuts 16 bit per channel with transparency
        deep = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
        if deep is not None and deep.dtype == np.uint16 and deep.ndim == 3 and deep.shape[2] == 4:
            print("note: %s is 16 bit per channel - scaled to 8 bit, so the result is not exact for the full range" % path, file=sys.stderr)
            deep = upright_array(deep, getattr(pic, "pixelcut_orientation", 1))
            rgba = np.clip(deep.astype(np.float64) / 257.0 + 0.5, 0, 255).astype(np.uint8)[:, :, [2, 1, 0, 3]]
            return Image.fromarray(np.ascontiguousarray(rgba), "RGBA")
    return pic.convert("RGBA")


def upright_array(arr, orient):
    """The same turn as ImageOps.exif_transpose, for a numpy array that was decoded without it (16 bit)."""
    if orient == 2:
        return arr[:, ::-1]
    if orient == 3:
        return np.rot90(arr, 2)
    if orient == 4:
        return arr[::-1]
    if orient == 5:
        return arr.swapaxes(0, 1)
    if orient == 6:
        return np.rot90(arr, -1)
    if orient == 7:
        return np.rot90(arr.swapaxes(0, 1), 2)
    if orient == 8:
        return np.rot90(arr, 1)
    return arr


def load_rgb(path):
    """8-bit RGB. Transparency is composited over white and 16-bit data is scaled to 8 bit - both with a notice, never silently."""
    im = open_upright(path)
    if im.width * im.height > 150_000_000:
        sys.exit("image too large (%d x %d = %d MP, limit 150 MP)" % (im.width, im.height, im.width * im.height // 1_000_000))
    if getattr(im, "n_frames", 1) > 1:
        print("note: %s has %d frames - only the first one is used" % (path, im.n_frames), file=sys.stderr)
    deep = None
    if Path(path).suffix.lower() in (".png", ".tif", ".tiff"):                            # Pillow cuts 48/64-bit PNG silently
        deep = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if deep is not None and deep.dtype == np.uint16 and deep.ndim == 3:
        print("note: %s is 16 bit per channel - scaled to 8 bit, so the result is not exact for the full range" % path, file=sys.stderr)
        deep = upright_array(deep, getattr(im, "pixelcut_orientation", 1))
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
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:                      # includes L/RGB with a PNG colour key
        im = im.convert("RGBA")
        if im.getchannel("A").getextrema()[0] < 255:                                      # a fully opaque RGBA is no news
            print("note: %s has transparency - composited over white" % path, file=sys.stderr)
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


def over_white(rgba_iter):
    """Composite parts over white, one at a time (a list or a generator): memory stays one picture, not one per part."""
    acc = None
    for im in rgba_iter:
        if acc is None:
            acc = np.full(im.shape[:2] + (3,), 255.0, np.float32)
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
    if not mask.any():
        return None                                           # nothing solid in this part (e.g. only faint pixels): no honest answer
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return [int(x), int(y)]


# ----------------------------------------------------------------------------- alpha
def cmd_alpha(a):
    refuse_inputs([a.input], [a.output])
    if Path(a.output).suffix.lower() not in (".png", ".tif", ".tiff"):
        sys.exit("error: alpha must write .png or .tif - %s would lose or change the transparency, the result is only exact lossless" % (Path(a.output).suffix or "this format"))
    rgb = load_rgb(a.input)
    rgba = color_to_alpha(rgb)
    save_atomic(Image.fromarray(rgba, "RGBA"), a.output)
    if a.verify:                                              # measure the SAVED file, not the buffer in memory
        with Image.open(a.output) as saved:
            back = np.array(saved.convert("RGBA"))
        err = np.abs(over_white([back]) - rgb.astype(np.float32)).max()
        print("max deviation over white: %.2f / 255 (measured on the saved file)" % err)


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


MAX_PARTS = 1024


def parse_grid(text, w, h):
    """'4x3' -> (columns, rows). 1xN = horizontal strips, Nx1 = vertical strips."""
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError:
        sys.exit("--grid must look like 4x3 (columns x rows); 1x5 = five horizontal strips, 5x1 = five vertical strips")
    if cols < 1 or rows < 1 or cols * rows < 2:
        sys.exit("--grid needs at least two cells, e.g. 2x1")
    if cols * rows > MAX_PARTS:
        sys.exit("--grid: at most %d cells" % MAX_PARTS)
    if cols > w or rows > h:
        sys.exit("--grid %s does not fit a %dx%d picture (a cell needs at least one pixel)" % (text, w, h))
    return cols, rows


def grid_cells(shape, cols, rows):
    """Cell id per pixel (col + row * cols). The borders are whole pixels, so every pixel belongs to exactly one cell."""
    h, w = shape
    xs = np.minimum(np.arange(w) * cols // w, cols - 1)
    ys = np.minimum(np.arange(h) * rows // h, rows - 1)
    return (ys[:, None] * cols + xs[None, :]).astype(np.int32)


def read_label_map(path, w, h):
    """Label picture -> (part id 0..n-1 per pixel, n). Grey values are the part numbers (8 or 16 bit, indexed also fine); the
    numbers need not be consecutive, parts are ordered by value. Every pixel belongs to exactly one part."""
    need_picture(path)
    try:
        im = Image.open(path)
        im.load()
    except Exception as e:
        sys.exit("--from-labels: cannot read %s (%s)" % (path, (str(e).strip().splitlines() or [type(e).__name__])[0]))
    if im.format in ("JPEG", "MPO", "WEBP", "JPEG2000") or str(im.info.get("compression", "")).lower() in ("jpeg", "tiff_jpeg", "webp", "tiff_webp", "jpeg2000"):
        sys.exit("--from-labels: %s is lossy (%s) - its grey values are not the part numbers any more; save the map as PNG" % (path, im.format))
    if im.mode in ("RGB", "RGBA", "LA", "F", "CMYK", "YCbCr", "LAB", "HSV"):
        sys.exit("--from-labels needs a grey picture (one number per pixel), not a colour one - %s is %s" % (path, im.mode))
    if im.size != (w, h):
        sys.exit("--from-labels: the label picture is %dx%d but the image is %dx%d - they must match" % (im.width, im.height, w, h))
    if im.mode == "1":
        im = im.convert("L")
    raw = np.array(im).astype(np.int64)                       # 'P' keeps its palette indices, which are the numbers
    values, ids = np.unique(raw, return_inverse=True)
    if len(values) < 2:
        sys.exit("--from-labels: the label picture has only one value - at least two parts are needed")
    if len(values) > MAX_PARTS:
        sys.exit("--from-labels: %d different values, at most %d parts" % (len(values), MAX_PARTS))
    return ids.reshape(h, w).astype(np.int32), len(values)


def cmd_labels(a):
    refuse_inputs([a.input], [a.output])
    need_folder_of(a.output)
    if Path(a.output).suffix.lower() not in LOSSLESS:
        sys.exit("error: a label map must be saved lossless as .png or .tif - JPEG would change the part numbers, BMP cannot hold more than 256 parts")
    w, h = open_upright(a.input).size
    cols, rows = parse_grid(a.grid, w, h)
    ids = grid_cells((h, w), cols, rows)
    save_atomic(Image.fromarray(ids.astype(np.uint8)) if cols * rows <= 256 else Image.fromarray(ids.astype(np.uint16)), a.output)
    print("wrote label map %s: %d parts, value = part number" % (a.output, cols * rows))


def cmd_split(a):
    held = []
    try:
        _cmd_split(a, held)
    finally:
        for f in held:                                        # the folder lock ends with the run
            f.close()


def _cmd_split(a, held):
    if folder_is_busy(a.outdir):
        sys.exit("error: another pixelcut run is working in %s right now - wait until it has finished" % a.outdir)
    if a.grid and a.from_labels:
        sys.exit("use either --grid or --from-labels, not both")
    refuse_inputs([a.input] + ([a.from_labels] if a.from_labels else []),
                  [Path(a.outdir) / n for n in ["parts.json"] + ["part_%d.png" % k for k in range(MAX_PARTS)]])
    refuse_inputs_in_workfolders([a.input] + ([a.from_labels] if a.from_labels else []), a.outdir)
    rgb = load_rgb(a.input)
    h, w = rgb.shape[:2]
    grid = parse_grid(a.grid, w, h) if a.grid else None
    labelled = read_label_map(a.from_labels, w, h) if a.from_labels else None
    exact = bool(grid or labelled)                            # cells / label parts are exact by construction: no slit, no ink edge
    nparts = grid[0] * grid[1] if grid else labelled[1] if labelled else None
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]) if a.opaque else color_to_alpha(rgb)
    if rgba[:, :, 3].max() == 0:                              # no pixel has any alpha: there is literally nothing to cut
        sys.exit("the image is completely white - nothing to cut (use --opaque to cut it anyway)")
    if exact:
        centre = (w / 2.0, h / 2.0)
    elif a.centre == "auto":                                  # centroid of everything that is not white
        m = (rgb.min(2) < 250).astype(np.uint8)
        M = cv2.moments(m, True)
        centre = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] > 0 else (w / 2.0, h / 2.0)
    else:
        try:
            centre = tuple(float(v) for v in a.centre.split(","))
            assert len(centre) == 2
        except (ValueError, AssertionError):
            sys.exit("--centre must look like 300,240 (x,y) or 'auto'")
    finite(list(centre), "--centre")
    if a.simplify < 0 or not np.isfinite(a.simplify):
        sys.exit("--simplify must be 0 or greater")
    if exact:
        angles = [0.0] * nparts                               # placeholder: one entry per part, only the count is used
    else:
        try:
            angles = [float(v) for v in a.angles.split(",")]
            finite(angles, "--angles")
            angles = [v % 360.0 for v in angles]              # 360 means 0, -90 means 270
        except ValueError:
            sys.exit("--angles must be numbers separated by commas, e.g. 32,148,214,327")
        if len(angles) > MAX_PARTS:
            sys.exit("--angles: at most %d parts" % MAX_PARTS)
        if len(angles) < 2 or len({round(v % 360, 6) for v in angles}) != len(angles):
            sys.exit("--angles needs at least two different angles")
    for flag, value in (("--gap", a.gap), ("--edge", a.edge), ("--depth", a.depth), ("--tooth", a.tooth)):
        if not np.isfinite(value) or value < 0:
            sys.exit("%s must be a finite number, 0 or greater" % flag)
    if a.gap > 100 or a.edge > 100 or a.depth > 1000:
        sys.exit("--gap/--edge/--depth are in pixels; these values are far too large")
    if a.depth > 0 and a.tooth <= 0:
        sys.exit("--tooth must be greater than 0 (or use --depth 0 for straight lines)")
    if exact:                                                 # said only after every option was accepted: no note and then an error
        ignored = [f for f, v, d in (("--centre", a.centre, "auto"), ("--angles", a.angles, "45,135,225,315"), ("--gap", a.gap, 2.6),
                                     ("--edge", a.edge, 3.2), ("--depth", a.depth, 9.0), ("--tooth", a.tooth, 18.0),
                                     ("--only-inside", a.only_inside, False)) if v != d]
        if ignored:
            print("note: %s ignored with %s (the parts are cut exactly, without slit)" % (", ".join(ignored), "--grid" if grid else "--from-labels"), file=sys.stderr)
    region = None
    if a.only_inside:                                         # cut only inside the largest non-white blob, leave satellites whole
        m = (rgb.min(2) < 250).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:                                             # an all-background image has no blob: cut the whole picture
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            region = cv2.dilate((lab == big).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if exact:
        sector = grid_cells((h, w), *grid) if grid else labelled[0]
        slit, edge_zone = np.zeros((h, w), bool), np.zeros((h, w), bool)
    else:
        sector, slit, edge_zone, order = split_sectors((h, w), centre, angles, a.tooth, a.depth, a.gap, a.edge, region)
    if region is not None and not exact:                       # satellites stay whole: one sector per connected area outside the main blob
        content = cv2.morphologyEx((rgb.min(2) < 250).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        nl, lab, _, _ = cv2.connectedComponentsWithStats(np.where(region, 0, content).astype(np.uint8))
        for c in range(1, nl):
            cc = lab == c
            sector[cc] = np.bincount(sector[cc].ravel()).argmax()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    locked = lock_folder(out, held)                           # recovery below must never meet a live run of ours
    if not locked:                                            # no working lock: we cannot tell a dead run from a live one - touch nothing of theirs
        for left in (".pixelcut_old", ".pixelcut_new", DONE):
            if (out / left).exists():
                sys.exit("error: %s exists - it may belong to a run that is still working, and this system cannot check that. "
                         "Make sure no other pixelcut is running in %s, then delete that folder" % (out / left, out))
    for scratch in (DONE, ".pixelcut_old", ".pixelcut_new"):
        real_folder(out / scratch, "scratch folder")
    if (out / DONE).is_dir():                                 # the swap had finished, only the clean-up was interrupted
        if foreign_entries(out / DONE):
            sys.exit("%s holds files pixelcut did not put there - check and remove it first" % (out / DONE))
        rmtree_force(out / DONE)
    leftover = out / ".pixelcut_old"                          # an earlier run was interrupted while swapping: its old set is the only full copy
    if leftover.is_dir():
        strange = foreign_entries(leftover)
        if strange:
            sys.exit("%s holds files pixelcut did not put there (%s) - check and remove it first" % (leftover, ", ".join(strange)))
        journal = leftover / JOURNAL
        if journal.is_file():                                 # files the interrupted run had already put in place: take them out again
            try:
                for name in json.loads(journal.read_text()):
                    if isinstance(name, str) and (PART_NAME.fullmatch(name) or name == "parts.json") and (out / name).is_file():
                        (out / name).unlink()
            except ValueError:
                pass
            journal.unlink()
        if (leftover / (JOURNAL + ".tmp")).is_file():
            (leftover / (JOURNAL + ".tmp")).unlink()
        for f in leftover.iterdir():                          # only pixelcut's own names are ever restored
            os.replace(f, out / f.name)
        shutil.rmtree(leftover)
        print("note: an interrupted earlier run was rolled back first (previous parts restored)", file=sys.stderr)
    old_manifest = out / "parts.json"                        # only files that an earlier pixelcut run listed may be replaced
    ours = set()
    if old_manifest.exists():
        try:
            ours = set(manifest_files(json.loads(old_manifest.read_text())))
        except Exception:                                    # a parts.json that is not ours is never touched
            sys.exit("%s exists but is not a pixelcut manifest - use an empty folder" % old_manifest)
    clash = [("part_%d.png" % k) for k in range(len(angles)) if (out / ("part_%d.png" % k)).exists() and ("part_%d.png" % k) not in ours]
    if clash:
        sys.exit("%s already exists in %s and was not written by pixelcut - use an empty folder" % (", ".join(clash), out))
    staging = out / ".pixelcut_new"                           # everything is written here first, replacing happens only when all parts worked
    if staging.exists():
        if foreign_entries(staging):
            sys.exit("%s exists and does not look like pixelcut's own scratch folder - remove it first" % staging)
        if not locked:                                        # without a working lock it may be a live run's folder: never delete it
            sys.exit("error: %s exists - it may belong to a run that is still working, and this system cannot check that. "
                     "Make sure no other pixelcut is running in %s, then delete that folder" % (staging, out))
        shutil.rmtree(staging)
    try:
        staging.mkdir()
    except FileExistsError:
        sys.exit("error: another pixelcut run is working in %s right now - wait until it has finished" % out)
    ink = np.array([20, 16, 14], np.uint8)
    info = []
    for k in range(len(angles)):
        img = rgba.copy()
        img[(sector != k) | slit] = 0
        if a.edge > 0 and not exact:
            e = edge_zone & (sector == k) & (img[:, :, 3] > 0)
            img[e, :3] = ink
            img[e, 3] = 255
        Image.fromarray(img, "RGBA").save(staging / ("part_%d.png" % k))
        solid = ((sector == k) & ~slit & (rgba[:, :, 3] > 8)).astype(np.uint8)
        info.append({"part": k, "starts_at_angle": 0 if exact else sorted(angles)[k], **({"cell": [k % grid[0], k // grid[0]]} if grid else {}),
                     "polygon": contour_polygon(solid, a.simplify), "polygons": contour_polygons(solid, a.simplify),
                     "fattest_point": fattest_point(solid)})
    files = ["part_%d.png" % k for k in range(len(angles))]
    (staging / "parts.json").write_text(json.dumps({"size": [w, h], "centre": list(centre), "files": files, "parts": info,
                                                    **({"grid": list(grid)} if grid else {}), **({"from_labels": True} if labelled else {})}, indent=1))
    backup = out / ".pixelcut_old"                            # old files are moved aside, not deleted - an abort rolls back
    try:
        backup.mkdir()                                        # (a leftover one was restored above, so this is always new)
    except FileExistsError:
        sys.exit("error: another pixelcut run is working in %s right now - wait until it has finished" % out)
    write_journal(backup, [])                       # names this run is installing, written BEFORE each install (all old files are aside by then)
    moved, installed = [], []
    try:
        for name in sorted(ours | {"parts.json"}):
            if (out / name).is_file():
                os.replace(out / name, backup / name)
                moved.append(name)
        for name in files + ["parts.json"]:
            installed.append(name)
            write_journal(backup, installed)
            os.replace(staging / name, out / name)
    except BaseException:
        for name in installed:                                # 1. take back exactly what this run put in place (journal stays: an interrupted
            if (out / name).is_file():                        #    roll-back is finished by the next run from the same list)
                (out / name).unlink()
        (backup / JOURNAL).unlink(missing_ok=True)            # 2. only now may the journal go - the files restored next must not be taken out again
        for f in list(backup.iterdir()):                      # 3. restore whatever is really in the backup (an abort between the move and the
            if f.name != JOURNAL + ".tmp":                    #    bookkeeping must not hide a file), and never delete a backup that is not empty
                os.replace(f, out / f.name)
        (backup / (JOURNAL + ".tmp")).unlink(missing_ok=True)
        backup.rmdir()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    os.replace(backup, out / DONE)                            # commit point: from here on the new set is the truth
    rmtree_force(out / DONE)
    shutil.rmtree(staging)
    print("wrote %d parts + parts.json to %s" % (len(angles), out))
    if a.verify:
        # without slit and edge the parts must add up to the original over white
        def masked(k):
            im = rgba.copy()
            im[sector != k] = 0
            return im
        err = np.abs(over_white(masked(k) for k in range(len(angles))) - rgb.astype(np.float32)).max()
        print("max deviation over white without slit: %.2f / 255" % err)
        real = over_white(np.array(Image.open(out / ("part_%d.png" % k)).convert("RGBA")) for k in range(len(angles)))
        real_err = np.abs(real - rgb.astype(np.float32)).max()
        print("saved files joined over white:        %.2f / 255 (with --gap %.1f --edge %.1f the slit is visible on purpose; 0 slit = exact)" % (real_err, a.gap, a.edge))


# ----------------------------------------------------------------------------- key
def cmd_key(a):
    if folder_is_busy(a.outdir):
        sys.exit("error: another pixelcut run is working in %s right now - wait until it has finished" % a.outdir)
    refuse_inputs([a.input], [Path(a.outdir) / n for n in ("base.png", "element.png", "preview.png")])
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
    comp = healed.copy()
    al3 = layer[:, :, 3:4].astype(np.float32) / 255
    comp = comp * (1 - al3) + layer[:, :, :3].astype(np.float32) * al3
    work, _ = scratch_beside(out / "base.png")                # all three are finished first, then published together
    try:
        Image.fromarray(np.clip(healed, 0, 255).round().astype(np.uint8)).save(work / "base.png")
        Image.fromarray(layer, "RGBA").save(work / "element.png")
        Image.fromarray(np.clip(comp, 0, 255).round().astype(np.uint8)).save(work / "preview.png")
        publish([(work / n, out / n) for n in ("base.png", "element.png", "preview.png")])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("wrote base.png (hole healed), element.png (exact pixels, scale %.2f), preview.png" % a.scale)



# ----------------------------------------------------------------------------- join / compare
def cmd_join(a):
    """Put parts back together (over white or a colour) - the way to 'rebuild' a picture from its exact pieces."""
    held = []
    try:
        _cmd_join(a, held)
    finally:
        for f in held:
            f.close()


def _cmd_join(a, held):
    if not Path(a.parts).is_dir():
        sys.exit("error: %s is not a folder of parts (does not exist, or is a file)" % a.parts)
    for work in (".pixelcut_old", ".pixelcut_new", DONE):     # a split is running here or died in the middle: the set is not complete
        if (Path(a.parts) / work).exists():
            sys.exit("error: %s is being written (or a split was interrupted) - run split again, or wait until it has finished" % a.parts)
    if (Path(a.parts) / ".pixelcut_lock").exists():           # parts made by pixelcut have a lock; a hand-made folder gets none from a read
        lock_folder(a.parts, held, quiet=True,                    # a split that starts while we read must wait: no mixed set, even if it rolls back
                    busy="error: a pixelcut run is writing in %s right now - try again when it has finished" % a.parts)
    for work in (".pixelcut_old", ".pixelcut_new", DONE):     # looked at again under the lock: a split may have started in between
        if (Path(a.parts) / work).exists():
            sys.exit("error: %s is being written (or a split was interrupted) - run split again, or wait until it has finished" % a.parts)
    manifest = Path(a.parts) / "parts.json"
    size = None
    manifest_bytes = None
    if manifest.exists():                                     # the manifest says which parts belong together - all of them are needed
        try:
            manifest_bytes = manifest.read_bytes()            # kept: it is checked again after reading, like the parts
            m = json.loads(manifest_bytes.decode("utf-8"))
            size = tuple(m["size"])                           # join needs only these: hand-made and older manifests may lack centre/angles
            assert isinstance(m["parts"], list) and m["parts"] and len(size) == 2                 and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)
            n = len(m["parts"])
            names = m.get("files") or ["part_%d.png" % k for k in range(n)]
            assert isinstance(names, list) and len(names) == n == len(set(names))                 and all(isinstance(x, str) and PART_NAME.fullmatch(x) for x in names)
            assert all(isinstance(x, dict) for x in m["parts"])
        except (ValueError, KeyError, TypeError, AssertionError, AttributeError):
            sys.exit("parts.json is damaged (needs 'size' [w,h], 'parts' and matching 'files')")
        files = [Path(a.parts) / x for x in names]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            sys.exit("parts.json lists %d parts but these are missing: %s" % (n, ", ".join(missing)))
    else:
        files = sorted(f for f in Path(a.parts).glob("part_*.png") if PART_NAME.fullmatch(f.name))
        numbers = sorted(int(f.stem.split("_")[1]) for f in files)
        gap = next((k for k, v in enumerate(numbers) if v != k), None)      # sorted and unique: the first number that is not its own index
        if gap is not None:
            sys.exit("part numbering has a gap (missing part_%d.png) - without parts.json every part is needed" % gap)
    if not files:
        sys.exit("no part_*.png in %s" % a.parts)
    shapes = set()
    for f in files:                                           # sizes come from the file headers: nothing is decoded yet
        with Image.open(f) as pim:
            shapes.add((pim.height, pim.width))
    if len(shapes) != 1 or (size and (size[1], size[0]) not in shapes):
        sys.exit("the parts do not all have the same size%s" % (" as parts.json says" if size else ""))
    refuse_inputs(files + [Path(a.parts) / "parts.json"], [a.output])
    if Path(a.output).suffix.lower() not in LOSSLESS + (".bmp",):
        sys.exit("error: join must write a lossless picture (.png, .tif or .bmp) - %s would change pixels, the rebuild is only exact lossless" % (Path(a.output).suffix or "this format"))
    import hashlib, io
    def digest(f):
        return hashlib.sha256(Path(f).read_bytes()).hexdigest()
    data = [f.read_bytes() for f in files]                    # every part is read once, into memory
    before = [hashlib.sha256(d).hexdigest() for d in data] + [hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None]
    def decode_part(d):
        if d[:4] == bytes([0x89, 0x50, 0x4E, 0x47]) and len(d) > 24 and d[24] == 16:
            sys.exit("error: join works on 8-bit parts (as split writes them) - this part is 16 bit per channel and would be cut off")
        im = Image.open(io.BytesIO(d))
        if any(v > 8 for v in (getattr(im, "tag_v2", None) or {}).get(258, ())):             # TIFF: BitsPerSample
            sys.exit("error: join works on 8-bit parts (as split writes them) - this part is more than 8 bit per channel and would be cut off")
        if im.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
            sys.exit("error: join works on 8-bit parts (as split writes them) - %s-bit grey parts would be cut off" % "16")
        return np.array(im.convert("RGBA"))
    out = over_white(decode_part(d) for d in data)
    try:                                                      # what is on disk now must be exactly what we used: a split that swapped parts and
        after = [digest(f) for f in files] + [digest(manifest) if manifest_bytes is not None and manifest.exists() else None]   # (rolled back: same names, sizes, times)
    except OSError:
        after = None
    if after != before or any((Path(a.parts) / w).exists() for w in (".pixelcut_old", ".pixelcut_new", DONE)):
        sys.exit("error: the parts in %s changed while they were being read (a split ran at the same time) - nothing was written, try again" % a.parts)
    save_atomic(Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8)), a.output)
    print("joined %d parts -> %s" % (len(files), a.output))


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
    need_picture(a.input)
    if not (0 <= a.speckle <= 128 and 1 <= a.colors <= 8 and 0 <= a.layers <= 255):
        sys.exit("--speckle must be 0..128, --colors 1..8, --layers 0..255")
    if Path(a.output).resolve() == Path(a.input).resolve():
        sys.exit("error: the SVG must be a different file than the input")
    if a.render and Path(a.render).resolve() in (Path(a.input).resolve(), Path(a.output).resolve()):
        sys.exit("error: --render must be a different file than the input and the SVG")
    if Path(a.output).suffix.lower() != ".svg":
        sys.exit("error: trace writes an SVG - name the output *.svg (not %s)" % (Path(a.output).suffix or "without an ending"))
    if a.render and Path(a.render).suffix.lower() != ".png":
        sys.exit("error: --render writes a PNG - name it *.png (not %s)" % (Path(a.render).suffix or "without an ending"))
    need_folder_of(a.output)
    if a.render:
        need_folder_of(a.render)
    here = Path(a.input).resolve()
    if Path(a.output).resolve() == here:
        sys.exit("error: the SVG must be a different file than the input")
    if a.render and Path(a.render).resolve() in (here, Path(a.output).resolve()):
        sys.exit("error: --render must be a different file than the input and the SVG")
    pic = decoded_8bit(a.input)                               # upright, 16 bit scaled with a notice - like compare/alpha/labels
    w, h = pic.size
    work, tmp_svg = scratch_beside(a.output)
    try:
        src = work / "input.png"                              # vtracer gets exactly the pixels we decoded, not a second reading of the file
        pic.save(src)
        vtracer.convert_image_to_svg_py(str(src), str(tmp_svg), colormode="color", hierarchical="stacked", mode="spline",
                                        filter_speckle=a.speckle, color_precision=a.colors, layer_difference=a.layers,
                                        corner_threshold=60, length_threshold=3.5, splice_threshold=45, path_precision=3)
        os.replace(tmp_svg, Path(a.output).resolve())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("wrote", a.output)
    if a.render:
        import subprocess
        import tempfile
        render_target = Path(a.render).resolve()
        if render_target in (Path(a.input).resolve(), Path(a.output).resolve()):
            sys.exit("error: --render must be a different file than the input and the SVG")
        chrome = find_chrome()
        if not chrome:
            sys.exit("--render needs Chrome/Edge/Brave")
        tmp = tempfile.TemporaryDirectory()
        t = Path(tmp.name)
        (t / "p.html").write_text('<html><body style="margin:0;background:#fff"><img src="%s" width=%d height=%d></body></html>'
                                  % (Path(a.output).resolve().as_uri(), w, h))
        shot = t / "shot.png"                                 # render aside, replace the target only when it worked
        try:
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                            "--user-data-dir=" + str(t / "prof"), "--window-size=%d,%d" % (w, h), "--screenshot=" + str(shot),
                            (t / "p.html").as_uri()], capture_output=True, timeout=120)
            if not shot.exists():
                sys.exit("error: Chrome did not produce a picture - %s was left as it was" % a.render)
            wd, staged = scratch_beside(render_target)        # copy beside the target first (may be another disk), then one atomic rename
            try:
                shutil.copyfile(shot, staged)
                os.replace(staged, render_target)
            finally:
                shutil.rmtree(wd, ignore_errors=True)
        finally:
            shutil.rmtree(tmp.name, ignore_errors=True)       # browser profile included
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
        import socket
        socket.setdefaulttimeout(60)                          # a dead connection must end in the Lanczos fallback, not hang
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


class BadEngineResult(Exception):
    pass


def read_engine_result(path):
    """The picture the engine made, as a BGR(A) array. It is 16 times the input, far over Pillow's bomb limit for a normal photo - and
    our own scratch file - so it is read with OpenCV: nothing global is switched off, nothing has to be put back."""
    arr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None or arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise BadEngineResult("the engine's picture could not be read")
    return arr


def bleed_colour(im):
    """RGBA picture whose fully transparent pixels carry the colour of their opaque neighbours (alpha untouched), so that a resampler
    working on the colour channels does not pull black into the edge of a shape."""
    arr = np.array(im)
    solid = arr[:, :, 3] > 0
    if solid.all() or not solid.any():
        return im
    rgb = arr[:, :, :3].astype(np.float32)
    m = solid.astype(np.float32)
    num = cv2.blur(rgb * m[:, :, None], (15, 15))
    den = cv2.blur(m, (15, 15))
    fill = num / np.maximum(den, 1e-6)[:, :, None]
    far = den <= 1e-6                                          # farther than the kernel from any solid pixel: the mean of the solid ones
    fill[far] = rgb[solid].mean(axis=0)
    arr[~solid, :3] = np.clip(fill[~solid] + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def check_upscaled_size(im, scale):
    if im.width * im.height * scale * scale > 600_000_000:
        sys.exit("error: the result would be %d x %d = %d MP (limit 600 MP) - use a smaller --scale or a smaller picture"
                 % (im.width * scale, im.height * scale, im.width * im.height * scale * scale // 1_000_000))


def lanczos_upscale(a, why):
    """Fallback without AI: plain Lanczos resize. Smooth, but no invented detail."""
    im = decoded_8bit(a.input)                                # the same pixels every other command works on (16 bit, EXIF, CMYK ...)
    check_upscaled_size(im, a.scale)
    print("Real-ESRGAN not usable here (%s) - using plain Lanczos resize instead (no AI detail)." % why)
    save_atomic(im.resize((im.width * a.scale, im.height * a.scale), Image.LANCZOS), a.output)
    print("wrote", a.output)


def cmd_upscale(a):
    """AI upscale with Real-ESRGAN (realesrgan-ncnn-vulkan). New pixels are invented: sharper, but not exact any more.
    Falls back to Lanczos where the engine cannot run (ARM, no Vulkan GPU)."""
    import subprocess
    if not 1 <= a.scale <= 8:
        sys.exit("--scale must be a whole number from 1 to 8")
    need_picture(a.input)
    refuse_inputs([a.input], [a.output])
    need_folder_of(a.output)
    if a.no_ai:
        return lanczos_upscale(a, "--no-ai")
    exe = a.bin or ensure_esrgan()
    if not exe:
        return lanczos_upscale(a, "no build for this machine")
    target = Path(a.output).resolve()
    work, tmp = scratch_beside(target)                        # the engine writes into a private folder beside the target; only a finished file replaces it
    engine_ok = target.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")   # the engine writes only these
    if not engine_ok:
        tmp = work / "engine.png"                             # other endings (.tif, .bmp): engine -> png, then converted
    try:
        src = work / "input.png"                              # the engine ignores EXIF and 16 bit: it gets the pixels we decoded
        pic = decoded_8bit(a.input)
        check_upscaled_size(pic, a.scale)
        if pic.mode == "RGBA" and target.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
            print("note: %s cannot hold transparency - flattened over white" % a.output, file=sys.stderr)
            pic = flatten_white(pic)
        needs_decode = a.scale != 4 or not engine_ok          # then the 4x picture (16 x the input) is held in memory once
        if needs_decode and pic.width * pic.height * 16 > 400_000_000:
            return lanczos_upscale(a, "the AI picture at this factor would need too much memory (%d MP)" % (pic.width * pic.height * 16 // 1_000_000))
        if pic.mode == "RGBA":
            pic = bleed_colour(pic)                           # the engine blurs across edges: colour under transparent pixels must not be black
        pic.save(src)
        subprocess.run([exe, "-i", str(src), "-o", str(tmp), "-n", a.model, "-s", "4"],      # the models are 4x models: other factors give a wrong picture
                       check=True, cwd=str(Path(exe).parent), capture_output=True, timeout=1800)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise OSError("the engine produced no picture")
        if not needs_decode:
            os.replace(tmp, target)
        else:
            arr = read_engine_result(tmp)
            if a.scale != 4:                                  # the wanted factor: resize the 4x result
                arr = cv2.resize(arr, (pic.width * a.scale, pic.height * a.scale), interpolation=cv2.INTER_AREA if a.scale < 4 else cv2.INTER_LANCZOS4)
            made = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB if arr.shape[2] == 3 else cv2.COLOR_BGRA2RGBA))
            del arr
            if engine_ok:                                     # written by us here (the engine's own file was resized): keep it as lossless as the format allows
                ext = target.suffix.lower()
                made.save(tmp, **({"quality": 95, "subsampling": 0} if ext in (".jpg", ".jpeg") else {"lossless": True} if ext == ".webp" else {}))
                os.replace(tmp, target)
            else:
                save_atomic(made, target)
    except (BadEngineResult, cv2.error) as e:
        return lanczos_upscale(a, str(e) if isinstance(e, BadEngineResult) else "OpenCV could not handle the AI picture")
    except MemoryError:
        return lanczos_upscale(a, "not enough memory for the AI result")
    except subprocess.TimeoutExpired:
        return lanczos_upscale(a, "the engine did not finish in 30 minutes")
    except OSError as e:
        return lanczos_upscale(a, "cannot start: %s" % e)
    except subprocess.CalledProcessError:
        return lanczos_upscale(a, "needs a Vulkan GPU/driver")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("wrote", a.output)


# ----------------------------------------------------------------------------- cli
class PixelcutError(Exception):
    """Raised by the Python API (never by the command line, which exits with the same message)."""


class _ApiParser(argparse.ArgumentParser):
    def error(self, message):
        raise PixelcutError(message)


def build_parser(parser_class=argparse.ArgumentParser):
    p = parser_class(prog="pixelcut", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version="pixelcut " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("alpha", help="white background -> transparent, exact over white")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--verify", action="store_true", help="print the max deviation over white")
    s.set_defaults(fn=cmd_alpha)

    s = sub.add_parser("split", help="cut into wedges with straight or zig-zag dividing lines")
    s.add_argument("input"); s.add_argument("outdir")
    s.add_argument("--grid", help="instead of wedges: COLSxROWS cells, e.g. 4x3; 1x5 = five horizontal strips, 5x1 = five vertical strips (always exact, no slit)")
    s.add_argument("--from-labels", help="instead of wedges: a label map (grey picture, same size, value = part number) - any shape, e.g. a segmentation mask; exact, no slit")
    s.add_argument("--centre", default="auto", help="x,y of the centre or 'auto' (centroid of the non-white area)")
    s.add_argument("--angles", default="45,135,225,315", help="angles of the dividing lines in degrees (0 = right, 90 = down); free, e.g. 32,148,214,327")
    s.add_argument("--tooth", type=float, default=18.0, help="zig-zag period in px")
    s.add_argument("--depth", type=float, default=9.0, help="zig-zag amplitude in px, 0 = straight lines")
    s.add_argument("--gap", type=float, default=2.6, help="width of the slit in px")
    s.add_argument("--edge", type=float, default=3.2, help="thin ink edge along the slit in px, 0 = none")
    s.add_argument("--only-inside", action="store_true", help="cut only the biggest blob, keep satellites (clouds, dots) whole")
    s.add_argument("--simplify", type=float, default=2.0, help="polygon simplification in px (for hit areas)")
    s.add_argument("--opaque", action="store_true", help="photos: keep every pixel opaque, no white-to-alpha")
    s.add_argument("--verify", action="store_true", help="after writing, re-join the saved parts and print the max deviation")
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("labels", help="write a label map: one picture, every pixel = number of its part")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--grid", required=True, help="COLSxROWS cells, e.g. 4x3 (1x5 = horizontal strips, 5x1 = vertical strips)")
    s.set_defaults(fn=cmd_labels)

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
    s.add_argument("--no-ai", action="store_true", help="plain Lanczos resize: no download, no engine, works offline")
    s.set_defaults(fn=cmd_upscale)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):                   # a path with characters the console cannot show must not turn a finished job into an error
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as e:                                # includes pyo3 PanicException, which is not an Exception
        if os.environ.get("PIXELCUT_DEBUG"):
            raise
        if isinstance(e, OSError) and oserror_message(e):
            sys.exit(oserror_message(e))
        sys.exit("error: %s (set PIXELCUT_DEBUG=1 for the full trace)" % (str(e).strip().splitlines() or [type(e).__name__])[0])


# ----------------------------------------------------------------------------- python api
# The functions below run exactly the command-line commands (same options, same defaults, same checks, same exactness), so
# there is one truth. Options are keyword arguments with the flag names, dashes as underscores: split(.., only_inside=True).
# Problems raise PixelcutError instead of exiting. Progress lines the command line prints are not shown.
def _flags(opts):
    argv = []
    for key, val in opts.items():
        flag = "--" + key.replace("_", "-")
        if val is None or val is False:
            continue
        if val is True:
            argv.append(flag)
        else:
            if isinstance(val, (list, tuple)):
                val = ",".join(str(v) for v in val)
            argv.append("%s=%s" % (flag, val))                # '--angles=-90,90': a value that starts with '-' must stay attached
    return argv


def _warn_about(err):
    """Notes the command line prints that a program using the API must not miss (a backup left behind, a lock that did not work)."""
    import warnings
    for ln in err.getvalue().splitlines():
        if ln.startswith("note:") and any(w in ln for w in ("left untouched", "still holds", "cannot tell", "cannot create the lock", "cannot lock folders", "kept")):
            warnings.warn(ln[5:].strip(), UserWarning, stacklevel=4)


def _run(cmd, positional, opts):
    import contextlib
    import io
    out, err = io.StringIO(), io.StringIO()

    def hint(message):                                        # "your previous files are kept in ..." must not be swallowed
        extra = [ln for ln in err.getvalue().splitlines() if ln.startswith("error:") or "kept in" in ln or "still holds" in ln]
        return message + ("\n" + "\n".join(extra) if extra else "")
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):   # the API prints nothing, notes included
            a = build_parser(_ApiParser).parse_args([cmd] + [str(x) for x in positional] + _flags(opts))
            a.fn(a)
    except SystemExit as e:
        if e.code in (None, 0):
            _warn_about(err)
            return out.getvalue()
        raise PixelcutError(hint(str(e.code).removeprefix("error: "))) from None
    except (KeyboardInterrupt, PixelcutError):
        raise
    except BaseException as e:                                # same net as the command line (pyo3 panics are not Exceptions)
        if isinstance(e, OSError) and oserror_message(e):
            raise PixelcutError(hint(oserror_message(e).removeprefix("error: "))) from e
        raise PixelcutError(hint((str(e).strip().splitlines() or [type(e).__name__])[0])) from e
    _warn_about(err)
    return out.getvalue()


def alpha(input, output, verify=False):
    """White background -> transparent (exact over white). Returns the output path."""
    _run("alpha", [input, output], {"verify": verify})
    return Path(output)


def split(input, outdir, **options):
    """Cut a picture into wedges (options: centre, angles, tooth, depth, gap, edge, only_inside, simplify, opaque, verify)
    or exactly along grid='4x3' / from_labels='map.png'.
    Returns the content of parts.json as a dict (size, parts, files, hit polygons); the pictures are part_N.png in outdir."""
    _run("split", [input, outdir], options)
    return json.loads((Path(outdir) / "parts.json").read_text(encoding="utf-8"))


def labels(input, output, grid):
    """Write a label map (one picture, every pixel = number of its part) for a grid like '4x3'. Returns the output path.
    Use it with split(..., from_labels=output), or paint/produce your own map (e.g. a segmentation mask)."""
    _run("labels", [input, output], {"grid": grid})
    return Path(output)


def join(parts, output):
    """Put the parts of a split folder back together over white (exact). Returns the output path."""
    _run("join", [parts, output], {})
    return Path(output)


def key(input, outdir, box, **options):
    """Lift an element out of a flat area (box = x0,y0,x1,y1; options: scale, to, threshold, close). Returns outdir."""
    _run("key", [input, outdir], dict(options, box=box))
    return Path(outdir)


def compare(first, second, resize=False):
    """Returns {'ssim': .., 'mean_diff': .., 'max_diff': ..} for two images."""
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            x, y = load_rgb(first), load_rgb(second)
            if x.shape != y.shape:
                if not resize:
                    raise PixelcutError("the images have different sizes (%dx%d vs %dx%d) - use resize=True to scale the second one"
                                        % (x.shape[1], x.shape[0], y.shape[1], y.shape[0]))
                y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
    except SystemExit as e:
        raise PixelcutError(str(e.code).removeprefix("error: ")) from None
    except (KeyboardInterrupt, PixelcutError):
        raise
    except Exception as e:                                    # missing / unreadable / not a picture / decompression bomb
        raise PixelcutError("cannot read the picture: %s" % (str(e).strip().splitlines() or [type(e).__name__])[0]) from e
    ss, mean, mx = measure(x, y)
    return {"ssim": float(ss), "mean_diff": float(mean), "max_diff": float(mx)}


def trace(input, output, **options):
    """Redraw as SVG (approximate; options: render, speckle, colors, layers). Returns the output path."""
    _run("trace", [input, output], options)
    return Path(output)


def upscale(input, output, **options):
    """Upscale (options: scale, model, bin, no_ai). Returns the output path. If the AI engine cannot run, a plain Lanczos resize is
    made instead and a UserWarning says so (the command line prints the same sentence)."""
    import warnings
    said = _run("upscale", [input, output], options)
    fell_back = next((ln for ln in said.splitlines() if "plain Lanczos" in ln), None)
    if fell_back and not options.get("no_ai"):
        warnings.warn(fell_back, UserWarning, stacklevel=2)
    return Path(output)


__all__ = ["__version__", "PixelcutError", "alpha", "split", "labels", "join", "key", "compare", "trace", "upscale"]


if __name__ == "__main__":
    sys.exit(main())
