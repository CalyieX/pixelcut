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
import math
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


__version__ = "1.9.7"
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


def _slot(name):
    """A backup slot is a plain ASCII number ('0', '1', ...): isdigit() alone also accepts things like superscripts."""
    return name.isascii() and name.isdigit()


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _discard(d):
    """Throw a finished backup folder away: ONE rename to .pixelcut_gone_* first (the name alone says 'done'), then delete.
    rmtree removes files in any order, so a kill in the middle of it must never leave a folder that still looks unfinished.
    If the rename is not possible the folder is left alone (a leftover is harmless, a half-deleted note is not)."""
    d = Path(d)
    if not d.name.startswith(".pixelcut_gone_"):
        gone = d.with_name(".pixelcut_gone_" + d.name[len(".pixelcut_"):])
        try:
            os.replace(d, gone)
        except OSError:
            return
        d = gone
    shutil.rmtree(d, ignore_errors=True)


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
    while aside.name.startswith(".pixelcut_gone_"):          # a random name must never look like a finished folder
        aside.rmdir()
        aside = Path(tempfile.mkdtemp(prefix=".pixelcut_", dir=folder))
    installed = []
    try:
        _write_atomic_text(aside / "publish.json", json.dumps({"dests": [str(Path(d).resolve()) for _, d in pairs],
                                                               "hashes": [_sha(s) for s, _ in pairs]}))   # lets a later run undo an abort
        for i, (_, dest) in enumerate(pairs):
            if Path(dest).exists():
                os.replace(dest, aside / str(i))
        (aside / "installing").write_text("1")                # from here on every destination that exists is a new file
        for src, dest in pairs:
            installed.append(dest)                            # before the move: an abort in between must still clean up
            os.replace(src, dest)
        gone = aside.with_name(".pixelcut_gone_" + aside.name[len(".pixelcut_"):])
        for attempt in range(6):                              # an antivirus/indexer may hold a file in there for a moment: try again before giving up
            try:
                os.replace(aside, gone)                       # the commit point: ONE rename. Before it a recovery goes back to the old set, after it the new set stays
                break
            except PermissionError:
                if attempt == 5:
                    raise
                import time
                time.sleep(0.25)
        aside = gone
    except BaseException:
        if not aside.exists():                                # the rename happened: committed, nothing may be undone any more
            _discard(gone)
            raise
        stuck = []
        for dest in installed:
            try:
                Path(dest).unlink(missing_ok=True)
            except OSError as e:
                stuck.append("%s (%s)" % (dest, e))
        try:
            (aside / "restoring").write_text("1")            # new files are gone: a later run must never delete again (it would hit restored originals)
        except OSError as e:                                  # without the marker restoring would be unsafe to repeat: keep the backups where they are
            print("error: could not record rollback progress - your previous files are kept in %s (%s)" % (aside, e), file=sys.stderr)
            raise
        for f in [f for f in aside.iterdir() if _slot(f.name)]:
            try:
                os.replace(f, pairs[int(f.name)][1])
            except OSError as e:
                stuck.append("%s (%s)" % (pairs[int(f.name)][1], e))
        if stuck:                                             # never destroy the only copy: say where it is
            print("error: could not restore everything - your previous files are kept in %s\n  problem with: %s" % (aside, "; ".join(stuck)), file=sys.stderr)
        else:
            _discard(aside)
        raise
    _discard(aside)


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
        if d.name.startswith(".pixelcut_gone_") and locked:   # the name alone says: a finished transaction whose clean-up was cut short
            shutil.rmtree(d, ignore_errors=True)
            continue
        try:
            names = {f.name for f in d.iterdir()}
        except OSError:
            continue
        if not meta.is_file():
            if locked and list(names) == ["publish.json.tmp"]:   # died while writing its note: nothing in here to restore
                _discard(d)
            continue
        if not locked:
            print("note: %s may be the backup of an interrupted run, but this system cannot tell whether it is still running - left untouched" % d, file=sys.stderr)
            continue
        try:
            dests = json.loads(meta.read_text(encoding="utf-8"))["dests"]
            assert isinstance(dests, list) and all(isinstance(x, str) for x in dests)
            assert all(Path(x).resolve().parent == folder for x in dests)      # a renamed folder must not make us write elsewhere
        except (OSError, ValueError, KeyError, AssertionError):
            if names <= {"publish.json", "publish.json.tmp"}:   # only our own half-written note: nothing of anybody's in here
                _discard(d)
            else:
                print("note: %s looks like a backup of an interrupted run but does not belong here - left untouched" % d, file=sys.stderr)
            continue
        try:
            hashes = json.loads(meta.read_text(encoding="utf-8")).get("hashes")
            if not (isinstance(hashes, list) and len(hashes) == len(dests) and all(isinstance(h, str) for h in hashes)):
                hashes = None
        except (OSError, ValueError):
            hashes = None
        left = False
        try:
            committed = (d / "committed").read_text() == "1"                  # an empty file from a failed write does not count
        except OSError:
            committed = False
        if committed:                                                             # written by older versions: the new set is complete
            _discard(d)
            continue
        else:
            if (d / "installing").exists() and not (d / "restoring").exists():
                for i, x in enumerate(dests):                                  # only a file that is exactly the new one we installed (same content) is removed
                    p = Path(x)
                    if hashes is not None and p.is_file() and not link_like(p) and _sha(p) == hashes[i]:
                        p.unlink(missing_ok=True)
                    elif p.exists():
                        print("note: %s was not removed - it is not the file the interrupted run installed" % p, file=sys.stderr)
                try:
                    (d / "restoring").write_text("1")                          # from here a repeated recovery only puts backups back, never deletes
                except OSError:
                    print("note: cannot record recovery progress in %s - left untouched" % d, file=sys.stderr)
                    continue
            restored = 0
            for f in [f for f in d.iterdir() if _slot(f.name)]:
                if int(f.name) < len(dests) and not os.path.lexists(dests[int(f.name)]):   # never over something that is there now
                    os.replace(f, dests[int(f.name)])
                    restored += 1
                else:
                    left = True
                    print("note: %s was not put back over %s - something is there" % (f, dests[int(f.name)] if int(f.name) < len(dests) else "?"), file=sys.stderr)
            if restored:
                print("note: finished an interrupted earlier run in %s - the previous files are back" % folder, file=sys.stderr)
            if left:
                print("note: %s still holds backups that belong to no file in the list - kept" % d, file=sys.stderr)
        if not left:
            _discard(d)


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



# ----------------------------------------------------------------------------- swap
SWAP_SHAPES = ("none", "diamond", "square", "circle", "heart", "star")


def swap_shape_alpha(shape, w, h, size):
    """Anti-aliased alpha (h x w, 0..1) of a shape centred in the box, `size` times the box (supersampled 8x)."""
    s = 8
    mask = np.zeros((h * s, w * s), np.uint8)
    cx, cy = w * s / 2, h * s / 2
    rx, ry = w * s / 2 * size, h * s / 2 * size
    if shape == "circle":
        cv2.ellipse(mask, (int(cx), int(cy)), (max(1, int(rx)), max(1, int(ry))), 0, 0, 360, 255, -1, cv2.LINE_AA)
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if shape == "diamond":
        pts = [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]
    elif shape == "square":
        pts = [(cx - rx, cy - ry), (cx + rx, cy - ry), (cx + rx, cy + ry), (cx - rx, cy + ry)]
    elif shape == "star":
        pts = [(cx + (rx if k % 2 == 0 else rx * .42) * math.sin(k * math.pi / 5), cy - (ry if k % 2 == 0 else ry * .42) * math.cos(k * math.pi / 5)) for k in range(10)]
    else:                                                      # heart
        t = np.linspace(0, 2 * math.pi, 240)
        hx, hy = 16 * np.sin(t) ** 3, 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        pts = list(zip(cx + hx / 17 * rx, cy - hy / 17 * ry))
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255, cv2.LINE_AA)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def swap_heal(rgb, hole, apply=None):
    """Close `hole` (bool mask) from its surroundings. Structure-aware (Telea): a hair line or outline next to the hole continues
    into it instead of being smeared, smooth skin stays smooth. The radius follows the size of the hole."""
    hole_u = hole.astype(np.uint8)
    radius = max(2, int(round(hole.sum() ** .5 * .18)))
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    fill = cv2.cvtColor(cv2.inpaint(bgr, hole_u, radius, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB).astype(np.float32)
    soft = cv2.GaussianBlur(cv2.dilate((hole if apply is None else apply).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 0.9)[:, :, None]   # `apply`: only this part of the hole is written back (the rest was only kept out of the sources)
    return rgb.astype(np.float32) * (1 - soft) + fill * soft


def swap_pick_region(rgb, points, tolerance):
    """Connected colour area(s) at the picked point(s). tolerance None = automatic: grow it step by step and stop before the area
    jumps (a jump means the colour has leaked into the surroundings)."""
    h, w = rgb.shape[:2]

    def flood(t):
        region = np.zeros((h, w), bool)
        for px, py in points:
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(np.ascontiguousarray(rgb).copy(), ff, (px, py), (0, 0, 0), (t,) * 3, (t,) * 3, 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
            region |= ff[1:-1, 1:-1] > 0
        return region
    if tolerance is not None:
        return flood(int(round(tolerance))), int(round(tolerance))
    sizes = []
    for t in (20, 30, 42, 56, 72, 92):
        r = flood(t)
        sizes.append((t, r, int(r.sum())))
    sizes = [x for x in sizes if x[2] >= 4] or [(60,) + (flood(60), int(flood(60).sum()))]
    for (t0, r0, n0), (t1, r1, n1) in zip(sizes, sizes[1:]):
        if n1 <= 1.15 * n0 and n0 <= .08 * h * w:              # the area has stopped growing: this is the element, not its surroundings
            return r0, t0
    ok = [x for x in sizes if x[2] <= .08 * h * w] or sizes[:1]
    return ok[-1][1], ok[-1][0]


def swap_add_tinted(rgb, region):
    """An element that is partly covered or in shade has darker parts of the SAME hue (a red heart under a strand of hair turns dark
    red). Take those connected parts along, so no tinted rest is left behind."""
    h, w = region.shape
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    core = region & (sat > .35) & (val > .35)
    if core.sum() < 4:
        return region
    ang = hue[core] * 2 * np.pi / 180
    mean_hue = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 2
    dh = np.abs((hue - mean_hue + 90) % 180 - 90)
    cand = (dh < 14) & (sat > max(.28, .45 * float(sat[core].mean()))) & (val > .12)     # shade keeps most of the saturation; a pale surface of a nearby hue is no part of the element
    ys0, xs0 = np.nonzero(region)
    kk = max(9, int(max(xs0.max() - xs0.min() + 1, ys0.max() - ys0.min() + 1) * 2.0)) | 1                 # reaches over a strand to the other piece of the element
    near = cv2.dilate(region.astype(np.uint8), np.ones((kk, kk), np.uint8)) > 0
    cand &= near
    both = cand | region
    ys_, xs_ = np.nonzero(region)
    gap = max(3, int(max(xs_.max() - xs_.min() + 1, ys_.max() - ys_.min() + 1) * .3)) | 1            # a strand across the element is a gap of this width: pieces on both sides still belong together
    bridged = cv2.morphologyEx(both.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min(max(gap, 7), 9),) * 2))
    n, lab = cv2.connectedComponents(bridged, connectivity=8)
    keep = np.unique(lab[region])
    grown = np.isin(lab, keep[keep > 0]) & both
    return grown if grown.sum() <= 5 * region.sum() else region


def swap_measure(rgb, healed, core, region):
    """What the element looks like, measured from the picture itself: its shading (brightness variation over its body), its outline
    (rim colour and width) and the shadow it casts on the surface (strength and direction)."""
    h, w = region.shape
    ys, xs = np.nonzero(region)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    lum = lambda im: im[..., 0] * .299 + im[..., 1] * .587 + im[..., 2] * .114
    body = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if body.sum() < 6:
        body = core
    base = rgb[body].astype(np.float32).mean(0)
    # shading: brightness ratio over the body, filled outwards so it covers the whole box, then stretched onto any new shape
    ratio = np.ones((h, w), np.float32)
    lbase = max(float(lum(base[None, None, :])[0, 0]), 1.0)
    ratio[body] = np.clip(lum(rgb.astype(np.float32))[body] / lbase, .55, 1.45)
    wgt = body.astype(np.float32)
    sig = max(1.5, (x1 - x0) / 6)
    num = cv2.GaussianBlur(ratio * wgt, (0, 0), sig)
    den = cv2.GaussianBlur(wgt, (0, 0), sig)
    filled = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 1.0)
    shade = filled[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    shade_abs = filled.astype(np.float32)
    # outline: the 1 px rim of the element against the body
    rim = region & ~(cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    outline = None
    if rim.sum() >= 4:
        rim_c = rgb[rim].astype(np.float32).mean(0)
        if np.abs(rim_c - base).max() > 14 and lum(rim_c[None, None, :])[0, 0] < lbase - 8:
            outline = rim_c                                    # a darker rim: keep it as the new shape's outline
    # cast shadow: the surface beside the element is compared with what the surface further away predicts (a plane fitted to a ring
    # around it, hair and rim left out). Darker than predicted = the element's shadow; where it lies = the light direction.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0) + 1
    dist = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
    far = (dist > size * .9) & (dist < size * 1.8)
    near = (dist > 1.5) & (dist < size * .8)
    L = lum(healed)
    lo = lum(rgb.astype(np.float32))
    skin = lo > 0.55 * float(np.median(L[far])) if far.any() else far
    far, near = far & skin, near & skin
    strength, v = 0.0, np.array([.6, .8], np.float32)             # nothing measured: light from the top left
    if far.sum() > 30 and near.sum() > 8:
        ys_, xs_ = np.nonzero(far)
        A = np.stack([np.ones(len(xs_)), xs_ - cx, ys_ - cy], 1).astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, lo[ys_, xs_].astype(np.float64), rcond=None)
        yn, xn = np.nonzero(near)
        pred = coef[0] + coef[1] * (xn - cx) + coef[2] * (yn - cy)
        drop = np.clip((pred - lo[yn, xn]) / np.maximum(pred, 1.0), 0, 1)
        sel = drop > .03
        if sel.sum() >= 4:
            strength = float(drop[sel].mean())
            wv = drop[sel]
            vv = np.array([np.average(xn[sel] - cx, weights=wv), np.average(yn[sel] - cy, weights=wv)], np.float32)
            if np.linalg.norm(vv) > 1e-3:
                v = vv / np.linalg.norm(vv)
    return {"base": base, "shade": shade, "shade_abs": shade_abs, "outline": outline, "shadow": strength, "dir": v, "box": (x0, y0, x1, y1)}


def swap_surface(healed, region, box):
    """Is the element flat on the picture or does it lie on a curved surface (a cheek, a ball, a fold)? Measured from the healed
    picture around it: a quadratic fit of the surface brightness gives the tilt (direction + strength of the gradient) and the
    bulge (convex or concave). Returns None for a flat element, else a dict used to bend the new shape the same way."""
    h, w = region.shape
    x0, y0, x1, y1 = box
    size = max(x1 - x0 + 1, y1 - y0 + 1)
    r_out = int(size * 1.1) + 4
    near = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    far = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1,) * 2)) > 0
    ring = far & ~near
    if ring.sum() < 60:
        return None
    lum = healed[..., 0] * .299 + healed[..., 1] * .587 + healed[..., 2] * .114
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ys, xs = np.nonzero(ring)
    u, v = (xs - cx) / r_out, (ys - cy) / r_out
    z = lum[ys, xs].astype(np.float64)
    mean = max(float(z.mean()), 8.0)
    z = z / mean - 1
    # robust: ignore texture (hair, lines) by clipping outliers around the median before fitting
    keep = np.abs(z - np.median(z)) < 3 * max(float(np.std(z)), 1e-3)
    u, v, z = u[keep], v[keep], z[keep]
    if len(z) < 60:
        return None
    A = np.stack([np.ones_like(u), u, v, u * u, v * v, u * v], 1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    fit = A @ coef
    var = float(z.var())
    r2 = 1 - float(((z - fit) ** 2).mean()) / var if var > 1e-9 else 0.0
    gx, gy = float(coef[1]), float(coef[2])
    tilt = math.hypot(gx, gy)
    bend = float(coef[3] + coef[4])
    # curved if a smooth gradient/bulge explains the surroundings and is big enough to see across the element
    if r2 < .3 or (tilt < .05 and abs(bend) < .05):
        return None
    return {"tilt": min(tilt, .8), "dir": np.array([gx, gy]) / max(tilt, 1e-6), "bulge": float(np.clip(-bend * .8, -.25, .25)), "r2": r2}


def swap_bend(shape, bw, bh, size, surf):
    """Shape alpha (bh x bw) laid onto a curved surface: foreshortened along the brightness gradient (the surface turns away there)
    and pulled in or out at the edges by the bulge."""
    s = 3
    hi = swap_shape_alpha(shape, bw * s, bh * s, size)
    gx, gy = float(surf["dir"][0]), float(surf["dir"][1])
    c = 1.0 / (1.0 + surf["tilt"] * .55)
    n = 2
    xx, yy = np.meshgrid((np.arange(bw * n) + .5) / n - bw / 2.0, (np.arange(bh * n) + .5) / n - bh / 2.0)
    p, q = xx * gx + yy * gy, -xx * gy + yy * gx
    p = p / c
    rho2 = (p * p + q * q) / max((bw * bw + bh * bh) / 4.0, 1.0)
    k = 1 + surf["bulge"] * rho2
    p, q = p * k, q * k
    ux, uy = p * gx - q * gy, p * gy + q * gx
    mx = ((ux + bw / 2.0) * s).astype(np.float32)
    my = ((uy + bh / 2.0) * s).astype(np.float32)
    warped = cv2.remap(hi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.resize(warped, (bw, bh), interpolation=cv2.INTER_AREA)


def swap_join_split(rgb, picked, tol):
    """A strand across the element cuts it into pieces and a single picked point only reaches one. Take the pieces of the same colour
    that lie right beside it (within a strand's width) - they belong to the same element."""
    ys, xs = np.nonzero(picked)
    size = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    k = max(7, int(size * .5)) | 1
    near = cv2.dilate(picked.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    mean = rgb[picked].astype(np.float32).mean(0)
    same = np.abs(rgb.astype(np.float32) - mean).max(2) <= max(tol * 1.7, 34)
    n, lab = cv2.connectedComponents(same.astype(np.uint8), connectivity=4)
    ids = np.unique(lab[near & same])
    grown = picked | np.isin(lab, ids[ids > 0])
    return grown if grown.sum() <= 3 * picked.sum() else picked


def swap_occluders(rgb, picked, level=95):
    """Things lying OVER the element (a strand of hair across a heart) are not part of it: they show up as a gap inside the element's
    closed outline whose colour is clearly not the surface. Returns that mask (bool); it stays on top, whatever is taken out or put in."""
    ys, xs = np.nonzero(picked)
    size = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    k = max(5, int(size * .45)) | 1
    closed = cv2.morphologyEx(np.pad(picked.astype(np.uint8), k), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))[k:-k, k:-k] > 0
    gap = closed & ~picked
    if gap.sum() < 3:
        return np.zeros_like(picked)
    surface = swap_heal(rgb, closed)
    if level is None:
        return np.zeros_like(picked)
    f3 = rgb.astype(np.float32)
    lum = lambda im: im[..., 0] * .299 + im[..., 1] * .587 + im[..., 2] * .114
    reach = cv2.dilate(closed.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0     # also the part just outside the outline (the heal would eat it)
    cand = reach & ~picked
    surface = swap_heal(rgb, reach)                                                   # what lies below, estimated over the element AND the candidate pixels
    sat = (f3.max(2) - f3.min(2)) / np.maximum(f3.max(2), 1.0)
    s_elem = float(np.median(sat[picked]))
    # Opaque cover or shadow? A shadow is the element itself, only darker: it keeps the element's saturation (a transparency factor on its colour).
    # A cover (hair, line work) is opaque: its colour is its own and does not depend on what lies below, so it is dull next to the element and
    # every pixel of it - the soft ends included - is a mix of ITS colour and the surface.
    shadow_like = sat >= .85 * s_elem
    own = cand & ~shadow_like & (lum(f3) < lum(surface.astype(np.float32)) - 25)
    if own.sum() < 4:
        return np.zeros_like(picked)
    Hc = f3[own & (lum(f3) <= np.percentile(lum(f3)[own], 25))].mean(0)                # the cover's own colour: the darkest dull pixels
    S = surface.astype(np.float32)
    d = S - Hc[None, None, :]
    a = np.clip(((S - f3) * d).sum(2) / np.maximum((d * d).sum(2), 1.0), 0, 1)       # how much of the pixel is cover (opacity)
    res = np.linalg.norm(f3 - (S - a[..., None] * d), axis=2)                           # distance to "cover over surface"
    a_min = .08 + (100 - level) / 250.0
    occ = cand & ~shadow_like & (a > a_min) & (res < 30)
    # Cover lying ON the element (a strand across a heart): its pixels are a mix of the cover's colour and the element's colour, so they sit on
    # the straight line between the two. A shadow does not: it is the element colour made darker AND more saturated, which leaves that line.
    lit = lum(f3) >= np.percentile(lum(f3)[picked], 60)
    B = np.median(f3[picked & lit], axis=0)
    dB = Hc - B
    al = np.clip(((f3 - B) * dB).sum(2) / max(float((dB * dB).sum()), 1.0), 0, 1)
    re_ = np.linalg.norm(f3 - (B[None, None, :] + al[..., None] * dB[None, None, :]), axis=2)
    if os.environ.get("PIXELCUT_DEBUG"): print("occluders: Hc", Hc, "B", B)
    on_element = picked & (al > .3 + (100 - level) / 250.0) & (re_ < SWAP_RE_MAX)
    occ |= on_element
    n, lab, st, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(occ)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 3:
            keep |= lab == i
    return keep


def swap_score(rgb, out, region, occ, base):
    """How good is a removal? Lower is better. Looks for what a viewer would notice: left-over element colour, hard new edges where
    the heal meets the picture, and strand pixels that were kept without being a real line (a lump)."""
    f = rgb.astype(np.float32)
    o = np.clip(out, 0, 255)
    zone = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    residue = float(((np.abs(o - base[None, None, :]).max(2) < 45) & zone).sum()) / max(zone.sum(), 1)
    lum = lambda im: im[..., 0] * .299 + im[..., 1] * .587 + im[..., 2] * .114
    gy, gx = np.gradient(cv2.GaussianBlur(lum(o).astype(np.float32), (0, 0), .8))
    g_out = np.hypot(gx, gy)
    gy, gx = np.gradient(cv2.GaussianBlur(lum(f).astype(np.float32), (0, 0), .8))
    g_in = np.hypot(gx, gy)
    ring = (cv2.dilate(region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~zone
    seam = abs(float(g_out[zone].mean()) - float(g_in[ring].mean())) / (float(g_in[ring].mean()) + 4)   # the healed place should be as busy as its surroundings
    lump = float(occ.sum()) / max(zone.sum(), 1)                                                       # strand pixels beyond a thin line count against it
    # thin dark lines that were in the picture there must still be there: a strand lost in the heal is a visible change
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    line_in = float(cv2.morphologyEx(lum(f).astype(np.float32), cv2.MORPH_BLACKHAT, kern)[zone & (lum(f) < lum(base[None, None, :])[0, 0] * .8)].sum())
    line_out = float(cv2.morphologyEx(lum(o).astype(np.float32), cv2.MORPH_BLACKHAT, kern)[zone & (lum(f) < lum(base[None, None, :])[0, 0] * .8)].sum())
    lost = max(0.0, line_in - line_out) / max(line_in, 1.0)
    reddish = 0.0
    return 12 * residue + seam + 1.5 * max(0.0, lump - .12) + 1.2 * lost + 4 * reddish


def swap_rim(rgb, picked, region0, base):
    """The grow ring around the element should hold only its antialiased rim (a mix of element and surface), never neighbouring
    structure like the lit edge of hair. A ring pixel stays if its colour lies on the line between the element colour and the surface."""
    f = rgb.astype(np.float32)
    surf = swap_heal(rgb, picked)
    ring = region0 & ~picked
    d = base[None, None, :] - surf
    dd = np.maximum((d * d).sum(2), 1.0)
    t = ((f - surf) * d).sum(2) / dd
    resid = np.linalg.norm(f - (surf + t[..., None] * d), axis=2)
    ok = ring & (t > .06) & (resid < 60)
    return picked | ok


def swap_remove(rgb, picked, region0, base):
    """Take the element out. The strand lying over it is found at several strengths; each result is scored by swap_score and the best
    one is used - no per-picture tuning by the caller. Returns (out, region, occ, soft, strand, chosen level, score)."""
    rim = swap_rim(rgb, picked, region0, base)
    keep = region0 & ~rim & (np.abs(rgb.astype(np.float32) - swap_heal(rgb, picked)).max(2) > 30)                                       # ring pixels that are NOT element rim (e.g. lit hair edge): healed around, then put back as they were
    best = None
    for level in (None, 99, 95, 85, 70, 50):
        occ = swap_occluders(rgb, picked, level)
        cov_a = cov_h = None                                   # the dark band across the element is the SHADOW of the strand, not hair: it goes with the shading
        cov = cov_a if cov_a is not None else np.zeros(picked.shape, np.float32)
        region = region0 & ~occ
        if not region.any():
            continue
        lum_ = rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114
        dark = lum_ < 110
        grown = lambda m, k: cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        base_hole = region | grown(occ, 5) | (cov > 0)
        # hair next to the hole is no source for the skin below it: it is kept out of the sources (wide), but only the part right at the hole is rewritten
        out = swap_heal(rgb, base_hole | (dark & grown(region, 15)), apply=region | (grown(occ, 5) & ~dark) | (cov > 0))
        strand, soft = swap_strand_pixels(rgb, out, occ & ~(cov > 0), picked)
        out = out * (1 - soft[:, :, None]) + strand * soft[:, :, None]
        if cov_h is not None:                                  # the cover over the element, translucent as before, now over the surface
            out = out * (1 - cov[:, :, None]) + cov_h[None, None, :] * cov[:, :, None]
        out = np.where(keep[:, :, None], rgb.astype(np.float32), out)
        sc = swap_score(rgb, out, region, occ, base)
        if best is None or sc < best[6] - 1e-4:
            best = (out, region, occ | (cov > 0), soft, strand, level, sc, cov, cov_h)
    return best


def swap_strand_line(occ, on, shape, ss=8):
    """A strand that crosses the element is a smooth thin line. Where it lay over the element the pixel estimate is ragged (stair steps, gaps),
    so the line is REBUILT: its centre path is fitted through the strand pixels (the part outside the element guides the part over it),
    its width follows the measured width, and it is drawn `ss` times finer and averaged down - clean edge, solid core, tapering end.
    Returns coverage (0..1) over the picture or None when the strand is not a thin line."""
    h, w = shape
    n, lab = cv2.connectedComponents(occ.astype(np.uint8), connectivity=8)
    near = cv2.dilate(on.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ids = [i for i in range(1, n) if (near & (lab == i)).any()]
    if not ids:
        return None
    ys, xs = np.nonzero(np.isin(lab, ids))
    yo, xo = np.nonzero(on)
    ext = (ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    swap_axes = ext[1] > ext[0]                                # mostly horizontal: work on the transposed picture
    m = np.isin(lab, ids)
    if swap_axes:
        m = m.T
        on_t = on.T
        hh, ww = w, h
    else:
        on_t = on
        hh, ww = h, w
    yo, xo = np.nonzero(on_t)
    ya, yb = yo.min(), yo.max()
    rows = np.arange(max(ya - 14, 0), min(yb + 4, hh))
    # the strand pixels that lie on the element say where the strand runs; in every row the pixel run nearest to that path is the strand
    # (a hair lock next to it in the same row is not part of it)
    on_rows = np.unique(yo)
    c_on = np.array([np.nonzero(on_t[r])[0].mean() + .5 for r in on_rows])
    p_on = np.polyfit(on_rows + .5, c_on, 1 if on_rows.size < 4 else 2)
    cx, wd, ok = [], [], []
    for r in rows:
        xr = np.nonzero(m[r])[0]
        if xr.size:
            runs = np.split(xr, np.nonzero(np.diff(xr) > 1)[0] + 1)
            pred = np.polyval(p_on, r + .5)
            run = min(runs, key=lambda q: abs(q.mean() + .5 - pred))
            if abs(run.mean() + .5 - pred) <= 3.5:
                cx.append(run.mean() + .5); wd.append(float(run.size)); ok.append(True)
                continue
        cx.append(np.nan); wd.append(np.nan); ok.append(False)
    cx, wd, ok = np.array(cx), np.array(wd), np.array(ok)
    if ok.sum() < 5:
        return None
    med = float(np.median(wd[ok]))
    ok = ok & (wd <= 1.6 * med)                                # rows where the strand runs into a hair lock are wider than the strand: they do not decide its path or width
    if ok.sum() < 5:
        return None
    wd = np.minimum(wd, 1.3 * med)
    r_ok = rows[ok] + .5
    deg = 2 if ok.sum() >= 8 else 1
    pc = np.polyfit(r_ok, cx[ok], deg)
    if wd[ok].max() > 7:                                       # not a thin line (a blob or a thick hair lock): leave it to the pixel estimate
        return None
    wm = np.convolve(np.pad(wd[ok], 1, mode="edge"), np.ones(3) / 3, mode="valid")
    wm = np.clip(wm, 1.0, 6.0)
    r0, r1 = max(ya - 1, rows[0]), min(yb + 1, rows[-1])
    ry = np.arange(r0, r1 + 1) + .5
    xc = np.polyval(pc, ry)
    ww_ = np.interp(ry, r_ok, wm)
    left = np.stack([(xc - ww_ / 2) * ss, ry * ss], 1)
    right = np.stack([(xc + ww_ / 2) * ss, ry * ss], 1)
    poly = np.vstack([left, right[::-1]])
    y0 = max(int(r0) - 3, 0); y1 = min(int(r1) + 4, hh)
    x0 = max(int(min(left[:, 0].min(), right[:, 0].min()) / ss) - 3, 0); x1 = min(int(max(left[:, 0].max(), right[:, 0].max()) / ss) + 4, ww)
    canvas = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    pts = np.round(poly - np.array([x0 * ss, y0 * ss])).astype(np.int32)
    cv2.fillPoly(canvas, [pts], 255)
    cov = cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    full = np.zeros((hh, ww), np.float32)
    full[y0:y1, x0:x1] = cov
    return full.T.copy() if swap_axes else full


def swap_strand_pixels(rgb, healed, occ, picked):
    """The top layer (hair, strand, line work) is never changed. Where it lies outside the element its pixels stay exactly as they were.
    Where it lay ON the element its pixels are a mix of the cover's colour and the element's: that mix is taken apart again into the
    cover's own colour and its opacity (a strand is solid in its core, soft only at its edges), so it can lie on whatever comes below -
    the surface after a removal or a new shape - with clean edges and not with a rim of the old element's colour.
    Returns (pixels, mask)."""
    f = rgb.astype(np.float32)
    pix = f.copy()
    soft = occ.astype(np.float32)
    on = occ & picked
    if on.sum() >= 4 and (occ & ~picked).sum() + on.sum() >= 6:
        lum = f[..., 0] * .299 + f[..., 1] * .587 + f[..., 2] * .114
        lit = lum >= np.percentile(lum[picked], 60)
        B = np.median(f[picked & lit], axis=0)
        dark_px = occ & (lum <= np.percentile(lum[occ], 25))
        Hc = f[dark_px].mean(0)
        dB = Hc - B
        al = np.clip(((f - B) * dB).sum(2) / max(float((dB * dB).sum()), 1.0), 0, 1)
        core = max(float(np.percentile(al[on], 90)), .3)
        a2 = np.clip(al / core, 0, 1)
        pix[on] = Hc[None, :]
        soft[on] = a2[on]
        try:                                                   # optional: a closed-form matting solver (pymatting) refines the opacity; without it the estimate above stands
            from pymatting import estimate_alpha_cf
            ys_, xs_ = np.nonzero(occ)
            y0, y1, x0, x1 = max(ys_.min() - 8, 0), min(ys_.max() + 9, f.shape[0]), max(xs_.min() - 8, 0), min(xs_.max() + 9, f.shape[1])
            oc = occ[y0:y1, x0:x1]
            sure = oc & (lum[y0:y1, x0:x1] <= np.percentile(lum[occ], 40))
            far = ~oc & (cv2.distanceTransform((~oc).astype(np.uint8), cv2.DIST_L2, 3) > 3)
            tri = np.full(oc.shape, .5)
            tri[sure], tri[far] = 1.0, 0.0
            if sure.any() and far.any():
                am = np.clip(estimate_alpha_cf(f[y0:y1, x0:x1].astype(np.float64) / 255, tri), 0, 1)
                full = np.zeros_like(soft)
                full[y0:y1, x0:x1] = am
                soft[on] = np.maximum(full[on], .25 * a2[on])
        except Exception:                                      # not installed or the solver failed: keep the own estimate
            pass
        # Sub-pixel edge: the opacity is a coverage estimate, its .5 level is where the edge runs. Drawn 4x finer as a sharp ramp one fine pixel
        # wide and averaged back down, the strand keeps a solid core and a smooth one-pixel edge - no stair steps, no smear (distance-field /
        # supersampling idea, cf. Green 2007 "Improved alpha-tested magnification", supersampling AA).
        ss = 4
        hh, ww = soft.shape
        ys_, xs_ = np.nonzero(on)
        y0, y1, x0, x1 = max(ys_.min() - 3, 0), min(ys_.max() + 4, hh), max(xs_.min() - 3, 0), min(xs_.max() + 4, ww)
        part = np.maximum(soft[y0:y1, x0:x1], cv2.GaussianBlur(occ[y0:y1, x0:x1].astype(np.float32), (0, 0), .7) * (soft[y0:y1, x0:x1] > .12))   # a strand is as solid along its length as it is at its best spot: dotted opacity is estimation noise
        hi = cv2.resize(part, ((x1 - x0) * ss, (y1 - y0) * ss), interpolation=cv2.INTER_CUBIC)
        hi = np.clip((hi - .45) / (1.0 / ss * .9) + .5, 0, 1)
        fine = cv2.resize(hi, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        soft[y0:y1, x0:x1] = np.where(on[y0:y1, x0:x1], fine, part)
        line = swap_strand_line(occ, on, soft.shape)
        if line is not None:
            zone = picked | on                                 # the rebuilt line also covers pixels of the element the pixel estimate had missed
            soft[zone] = line[zone]
            pix[zone & (line > 0)] = Hc[None, :]
    return pix, soft


def swap_sweep(mask, v, length, ss=4):
    """Pull a mask along a direction: everything it would cover when moved up to `length` px. This is what a cast shadow is - the
    caster slid away from the light. Rendered `ss` times finer than the picture and averaged down, so its edges are smooth, not stair steps."""
    h, w = mask.shape
    hi = cv2.resize(mask.astype(np.float32), (w * ss, h * ss), interpolation=cv2.INTER_LINEAR)
    hi = cv2.GaussianBlur(hi, (0, 0), ss * .35)                    # the cover's own edge is soft already: carry that over
    acc = hi.copy()
    for t in np.arange(.5, length + .01, .5):
        acc = np.maximum(acc, cv2.warpAffine(hi, np.float32([[1, 0, v[0] * t * ss], [0, 1, v[1] * t * ss]]), (w * ss, h * ss), flags=cv2.INTER_LINEAR))
    acc = cv2.GaussianBlur(acc, (0, 0), ss * .45)
    return cv2.resize(acc, (w, h), interpolation=cv2.INTER_AREA)


def swap_cast_fit(rgb, picked, occ):
    """Which way, how far and how strongly does the cover on the element throw its shadow? Measured on the ORIGINAL: slide the cover in
    many directions and lengths and keep the one whose shadow agrees best with the element's darkened part. Returns (direction,
    length, strength) or None when the element shows no shadow of its cover."""
    f = rgb.astype(np.float32)
    lum = f[..., 0] * .299 + f[..., 1] * .587 + f[..., 2] * .114
    cover = occ & (cv2.dilate(picked.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    body = picked & ~occ
    if cover.sum() < 3 or body.sum() < 12:
        return None
    lit = body & (lum >= np.percentile(lum[body], 60))
    ratio = lum / max(float(lum[lit].mean()), 1.0)
    dm = body & (ratio < .8)
    if dm.sum() < 6:
        return None
    ys, xs = np.nonzero(picked)
    size = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    best, best_score = None, 0.0
    for ang in np.arange(0, 360, 15):
        v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))], np.float32)
        for length in range(2, int(size * .8) + 1, 2):
            sw = (swap_sweep(cover, v, length, ss=1) > .5) & body & ~cover
            inter = float((sw & dm).sum())
            score = inter / max(float((sw | dm).sum()), 1.0)
            if score > best_score:
                best, best_score = (v, float(length), sw), score
    if best is None or best_score < .2:
        return None
    v, length, sw = best
    strength = float(np.clip(1.0 - np.median(ratio[sw & dm]), .1, .5))
    return v, length, strength


def swap_style(rgb):
    """Photo or drawing? A drawing uses few distinct colours and has hard-edged shadows; a photo has thousands of tones and soft ones.
    The shadow of a new shape imitates that style - pixelcut is an editor, it must not make a drawing look rendered."""
    q = (rgb // 24).astype(np.int32)
    key = q[..., 0] * 121 + q[..., 1] * 11 + q[..., 2]
    used = np.bincount(key.ravel(), minlength=1331)
    share = used / max(key.size, 1)
    return "drawing" if int((share > .0005).sum()) < 260 else "photo"


def swap_hair_layer(rgb, out, picked, region, base):
    """The top layer (hair) next to and over the element is protected: dark, not element-coloured pixels that are connected to the big dark
    mass around. A strand tip reaching into the element is part of it. Returns (mask, opacity, hair colour): opacity says how much of each
    pixel is hair, so the layer can lie over a NEW shape with the same see-through edges it had over the old one."""
    f = rgb.astype(np.float32)
    lum = lambda im: im[..., 0] * .299 + im[..., 1] * .587 + im[..., 2] * .114
    lf = lum(f)
    cand = (lf < 105) & (np.abs(f - base[None, None, :]).max(2) > 60) & ((f[..., 0] - f[..., 1]) < 45)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 60]
    zone = (cv2.dilate(region.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    hair = np.isin(lab, big) & zone
    if hair.sum() < 3:
        return np.zeros(picked.shape, bool), np.zeros(picked.shape, np.float32), None
    colour = f[hair & (lf <= np.percentile(lf[hair], 40))].mean(0)
    l_h = float(lum(colour[None, None, :])[0, 0])
    opacity = np.clip((lum(out) - lf) / np.maximum(lum(out) - l_h, 20.0), 0, 1) * hair
    return hair, opacity.astype(np.float32), colour


SWAP_RE_MAX = 20                                               # how far from the straight cover-element colour line a pixel may lie and still count as cover (a shadow lies further off)
_SWAP_LAST = {}                                                # what the last swap run worked out (the scaled run needs it to build the full-size result)
SWAP_AUTO_SCALE = False                                        # work-in-progress: analysing big pictures at small scale still leaves a faint rim, so it is off until it is clean
SWAP_WORK_SIZE = 28                                            # px: the element size the analysis is tuned for (kernels, tolerances); bigger elements are analysed at this scale


def swap_scaled(a, rgb, size_el):
    """Analyse small, render fine. The analysis (what is the element, what lies over it, which way does the shadow fall) is tuned for an
    element of about SWAP_WORK_SIZE px, so a bigger picture is first worked at that scale. The result is then carried back to the full
    resolution as a DIFFERENCE to the original - everything that did not change stays bit-exact - and what lies on top of the element
    (strands, hair, outlines) is taken from the full-size original again, so it stays sharp and untouched."""
    import copy, tempfile
    h, w = rgb.shape[:2]
    f = SWAP_WORK_SIZE / float(size_el)
    sw, sh_ = max(int(round(w * f)), 8), max(int(round(h * f)), 8)
    small = cv2.resize(rgb, (sw, sh_), interpolation=cv2.INTER_AREA)
    points = [tuple(int(v) for v in pt.split(",")) for pt in a.at.split(";")]
    with tempfile.TemporaryDirectory() as td:
        tin, tout = str(Path(td) / "in.png"), str(Path(td) / "out.png")
        Image.fromarray(small).save(tin)
        inner = copy.copy(a)
        inner.input, inner.output, inner.cut, inner._inner = tin, tout, None, True
        inner.grow = a.grow or 1                                   # a smoothly enlarged picture has a softer rim: take one px more
        inner.at = ";".join("%d,%d" % (min(int(round(x * f)), sw - 1), min(int(round(y * f)), sh_ - 1)) for x, y in points)
        cmd_swap(inner)
        last = dict(_SWAP_LAST)
    up = lambda m: cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    base = rgb.astype(np.float32)
    delta = last["out"] - small.astype(np.float32)
    changed = cv2.dilate((np.abs(delta).max(2) > .5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # a little more than what changed: the soft rim of the old element reaches out there
    m_up = cv2.GaussianBlur(up(changed), (0, 0), max(1.0, .8 / f))                       # feathered, so the new area blends into the untouched picture
    out_up = cv2.resize(last["out"], (w, h), interpolation=cv2.INTER_CUBIC)
    final = base * (1 - m_up[:, :, None]) + out_up * m_up[:, :, None]
    on = up(last["occ"] & last["picked"])
    col = np.where((on > .5)[:, :, None], cv2.resize(last["strand"], (w, h), interpolation=cv2.INTER_LINEAR), base)
    al = np.clip(up(last["soft"]), 0, 1)[:, :, None]
    final = final * (1 - al) + col * al
    d = last["base"][None, None, :] - out_up                                             # what the old element is at each place, against the new surface
    tt = ((base - out_up) * d).sum(2) / np.maximum((d * d).sum(2), 1.0)
    rim = (tt > .05) & (np.linalg.norm(base - (out_up + tt[..., None] * d), axis=2) < 45)   # a mix of old element and surface = the element's own soft rim, never cover
    ring = cv2.dilate((m_up > .3).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(3 / f) + 1,) * 2)) > 0
    clean = cv2.GaussianBlur((ring & rim & (tt > .12)).astype(np.float32), (0, 0), 1.0)[:, :, None]   # left-over rim of the old element just outside the edited area goes too
    final = final * (1 - clean) + out_up * clean
    hu = np.clip((up(last["hair"]) - .15) / .7, 0, 1) * (~rim)
    hu = hu[:, :, None]
    final = final * (1 - hu) + base * hu
    final = np.clip(final, 0, 255).round().astype(np.uint8)
    save_atomic(Image.fromarray(final), a.output)
    ch = (final != rgb).any(2)
    cy, cx = np.nonzero(ch)
    area = "" if len(cx) == 0 else " in a %dx%d px area" % (cx.max() - cx.min() + 1, cy.max() - cy.min() + 1)
    print("scale: the element is %d px big, analysed at 1/%.1f size and rendered back at full resolution" % (size_el, 1 / f))
    print("check: %d of %d pixels changed%s, the rest (%.2f%%) is identical to the original" % (int(ch.sum()), h * w, area, 100 * (1 - ch.mean())))


def cmd_swap(a):
    """Take an element out of a picture (picked with one point), heal the place and, if asked, set a new shape there that looks like
    it belongs: pixelcut measures the original element's shading, outline and cast shadow and gives them to the new shape."""
    refuse_inputs([a.input], [a.output] + ([a.cut] if a.cut else []))
    rgb = load_rgb(a.input)
    h, w = rgb.shape[:2]
    try:
        points = [tuple(int(v) for v in pt.split(",")) for pt in a.at.split(";")]
        if any(len(pt) != 2 for pt in points):
            raise ValueError
    except ValueError:
        sys.exit("error: --at must look like x,y (whole numbers; several points: x,y;x,y)")
    for px, py in points:
        if not (0 <= px < w and 0 <= py < h):
            sys.exit("error: --at point %d,%d lies outside the picture (%dx%d)" % (px, py, w, h))
    if a.shape not in SWAP_SHAPES:
        sys.exit("error: --shape must be one of %s" % ", ".join(SWAP_SHAPES))
    for name, val, lo, hi in (("--tolerance", a.tolerance, 1, 255), ("--grow", a.grow, 0, 20), ("--size", a.size, .2, 4), ("--shadow", a.shadow, 0, 1)):
        if val is not None and (not np.isfinite(val) or not (lo <= val <= hi)):
            sys.exit("error: %s must be between %g and %g" % (name, lo, hi))
    picked, t = swap_pick_region(rgb, points, a.tolerance)
    picked = swap_join_split(rgb, picked, t)
    core0 = picked.copy()
    if a.tolerance is None:
        picked = swap_add_tinted(rgb, picked)
    if picked.sum() > .25 * h * w:
        sys.exit("error: the picked colour covers more than a quarter of the picture - pick a point on a small element or lower --tolerance")
    if picked.sum() < 4:
        sys.exit("error: almost nothing was picked at --at %s - raise --tolerance or pick the middle of the element" % a.at)
    ys_p, xs_p = np.nonzero(picked)
    size_el = max(xs_p.max() - xs_p.min() + 1, ys_p.max() - ys_p.min() + 1)
    if SWAP_AUTO_SCALE and not getattr(a, "_inner", False) and size_el > 1.6 * SWAP_WORK_SIZE and not a.cut:
        return swap_scaled(a, rgb, size_el)
    region = picked
    if a.grow:
        region = cv2.dilate(picked.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow + 1,) * 2)) > 0
    base_col = rgb[picked].astype(np.float32).mean(0)
    out, region, occ, soft, strand, level, score, cov_a, cov_h = swap_remove(rgb, picked, region, base_col)   # tries several strand strengths and keeps the best
    hair, hair_a, hair_c = swap_hair_layer(rgb, out, picked, region, base_col)
    out = np.where(hair[:, :, None], rgb.astype(np.float32), out)   # the top layer (hair) is never changed
    ys, xs = np.nonzero(region)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    style = swap_style(rgb)
    surf = swap_surface(out, region, (x0, y0, x1, y1))         # analysis first: flat 2D picture or a surface with 3D shading?
    if a.cut:
        layer = np.zeros((h, w, 4), np.uint8)
        layer[:, :, :3] = rgb
        layer[:, :, 3] = region.astype(np.uint8) * 255
        save_atomic(Image.fromarray(layer[y0:y1 + 1, x0:x1 + 1], "RGBA"), a.cut)
    note = ""
    cast_fit = swap_cast_fit(rgb, picked, occ)
    photon = ""
    if cast_fit is not None:                                   # photon check: the shadow falls away from the light, so the light comes from the other side - does the surface shading say the same?
        light = -cast_fit[0]
        where = ("top" if light[1] < -.5 else "bottom" if light[1] > .5 else "") + ("-left" if light[0] < -.5 and abs(light[1]) > .5 else "left" if light[0] < -.5 else "-right" if light[0] > .5 and abs(light[1]) > .5 else "right" if light[0] > .5 else "")
        photon = "light from the %s (shadow falls away from it)" % (where.strip("-") or "front")
        if surf is not None and surf["tilt"] > .08:
            agree = float(np.dot(surf["dir"], light))
            if agree < -.3:                                    # the surface is brighter on the OTHER side: the two measurements disagree - trust the shadow less
                cast_fit = (cast_fit[0], cast_fit[1], cast_fit[2] * .6)
                photon += "; the surface shading points the other way, shadow weakened"
            else:
                photon += "; the surface shading agrees"
    cast = np.zeros((h, w), np.float32)                        # the cover's shadow as a soft mask, drawn the same way onto whatever lies below
    shadow_px = np.zeros((h, w), bool)
    if cast_fit is not None:
        cv_, cl_, cs_ = cast_fit
        cover_in = occ & (cv2.dilate(picked.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        cast = swap_sweep(cover_in, cv_, cl_) * (1 - occ.astype(np.float32))
        shadow_px = picked & ~occ & (cast > .5)
    shade_core = picked & ~occ & ~shadow_px                    # the element's own shading, without the shadow the cover put on it
    if a.shape != "none":
        meas = swap_measure(rgb, out, shade_core, region)
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        shape_a = np.zeros((h, w), np.float32)
        if surf is None or a.flat_surface:
            shape_a[y0:y1 + 1, x0:x1 + 1] = swap_shape_alpha(a.shape, bw, bh, a.size)
        else:                                                  # the element sits on a curved surface: bend the new shape the same way
            shape_a[y0:y1 + 1, x0:x1 + 1] = swap_bend(a.shape, bw, bh, a.size, surf)
        colour = meas["base"]
        if a.colour is not None:
            try:
                colour = np.array([int(a.colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
            except ValueError:
                sys.exit("error: --colour must look like #rrggbb")
        paint = np.zeros((h, w, 3), np.float32)
        paint[:] = colour
        facet = None
        if not a.flat:                                         # a gentle bevel: edges facing the light are a little brighter, edges facing away a little darker
            hgt = cv2.distanceTransform((shape_a > .5).astype(np.uint8), cv2.DIST_L2, 3)
            hgt = np.minimum(hgt / max(.28 * min(bw, bh), 1.0), 1.0)
            hgt = cv2.GaussianBlur(hgt, (0, 0), max(.6, min(bw, bh) / 28))
            gy_, gx_ = np.gradient(hgt)
            lx, ly = -float(meas["dir"][0]), -float(meas["dir"][1])          # light comes from the side the shadow does not fall to
            bevel = np.clip((gx_ * lx + gy_ * ly) * -max(bw, bh) * .9, -1, 1)
            facet = 1.0 + .06 * bevel * (hgt < .6)      # only at the edge, and weak: the shape is red up to the cover, the shadow comes after it
        if not a.flat:                                         # the new shape gets the original's shading
            shade = meas["shade_abs"][y0:y1 + 1, x0:x1 + 1]    # by position, not stretched: a shadow that fell on the old element (the strand's) falls on the new one at the same place
            paint[y0:y1 + 1, x0:x1 + 1] = np.clip(colour[None, None, :] * shade[:, :, None], 0, 255)
        if cast_fit is not None:                               # the cover's shadow on the new shape: starts at the cover's edge, runs the measured way and length
            paint = paint * (1 - cast_fit[2] * cast)[:, :, None]
            paint = np.clip(paint * facet[:, :, None], 0, 255)
        if meas["outline"] is not None and not a.flat:         # ... and its outline: a rim of the same colour and ~1 px width
            inner = cv2.erode(shape_a, np.ones((3, 3), np.uint8))
            rimw = np.clip(shape_a - inner, 0, 1)
            paint = paint * (1 - rimw[:, :, None] * .8) + meas["outline"][None, None, :] * (rimw[:, :, None] * .8)
        vx_s, vy_s = float(meas["dir"][0]), float(meas["dir"][1])
        strength = a.shadow if a.shadow is not None else min(max(meas["shadow"] * 2, .12), .3)
        if strength > 0:                                       # a tight contact shadow plus a wider soft one, in the measured direction
            scale = max(bw, bh) / 16.0
            vx, vy = float(meas["dir"][0]), float(meas["dir"][1])

            def shifted(blur, dist):
                m = np.float32([[1, 0, vx * dist * scale], [0, 1, vy * dist * scale]])
                return cv2.GaussianBlur(cv2.warpAffine(shape_a, m, (w, h)), (0, 0), max(.5, blur))
            sh = 1 - (1 - .55 * shifted(.6 * scale + .4, .8)) * (1 - .45 * shifted(2.0 * scale + .6, 2.6))
            sh = np.clip(sh * strength * 1.6, 0, .85) * (1 - shape_a)
            if style == "drawing":                             # a drawn shadow has a clear edge: step it, keep the measured strength
                sh = np.clip((sh / max(float(sh.max()), 1e-3) - .3) / .12, 0, 1) * float(sh.max()) * (1 - shape_a)
            out = out * (1 - .55 * sh[:, :, None])             # multiply: the surface below gets darker but keeps its own colour and texture
        out = out * (1 - shape_a[:, :, None]) + paint * shape_a[:, :, None]
        top = soft[:, :, None]                                 # ... and lies on top of the new shape again
        out = out * (1 - top) + strand * top
        if cov_h is not None:
            out = out * (1 - cov_a[:, :, None]) + cov_h[None, None, :] * cov_a[:, :, None]   # the translucent cover over the NEW shape too
        if hair_c is not None:                                 # hair over the new shape: see-through at its edges exactly as before, solid where it was solid
            ha = hair_a[:, :, None]
            over = np.where(ha > .9, rgb.astype(np.float32), out * (1 - ha) + hair_c[None, None, :] * ha)
            out = np.where(hair[:, :, None], over, out)
        note = ", new %s (shading%s%s, shadow %.0f%%)" % (a.shape, " kept" if not a.flat else " off", " + outline" if meas["outline"] is not None and not a.flat else "", strength * 100)
        note += "; surface: " + ("flat" if surf is None or a.flat_surface else "curved (%s, tilt %.0f%%) - shape bent to follow it" % ("convex" if surf["bulge"] > .03 else "concave" if surf["bulge"] < -.03 else "tilted", surf["tilt"] * 100))
    if a.shape == "none":                                      # taken out: the shadow the cover threw on the element now falls on the surface below, as a subtle darkening of the surface
        if cast_fit is not None:
            ys_p, xs_p = np.nonzero(picked)
            sig = max(1.2, max(xs_p.max() - xs_p.min() + 1, ys_p.max() - ys_p.min() + 1) / 9.0)
            feather = cv2.GaussianBlur((picked & ~hair & ~occ).astype(np.float32), (0, 0), sig)     # the shadow fades out where the old element ended - no cut-out of its silhouette
            dark = cv2.GaussianBlur(cast, (0, 0), sig * .6) * cast_fit[2] * .6 * feather * (~hair) * (~occ)
            out = out * (1 - dark)[:, :, None]
    _SWAP_LAST.update(out=np.array(out, np.float32), occ=occ, hair=hair, soft=soft, strand=strand, picked=picked, base=base_col)
    save_atomic(Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8)), a.output)
    changed = (np.clip(out, 0, 255).round().astype(np.uint8) != rgb).any(2)
    cy, cx = np.nonzero(changed)
    area = "" if len(cx) == 0 else " in a %dx%d px area" % (cx.max() - cx.min() + 1, cy.max() - cy.min() + 1)
    look = "flat 2D picture, no 3D shading to follow" if surf is None else "3D surface (%s, tilt %.0f%%): shading and curvature are followed" % ("convex" if surf["bulge"] > .03 else "concave" if surf["bulge"] < -.03 else "tilted", surf["tilt"] * 100)
    print("analysis: %s; style: %s (%s shadows); %s" % (look, style, "hard-edged" if style == "drawing" else "soft", ("depth: %d px lie in front of the element and stay in front" % int(occ.sum())) if occ.any() else "depth: nothing lies in front of the element"))
    if photon:
        print("photon check: " + photon)
    print("wrote %s (picked %d px at %s, tolerance %d%s)" % (a.output, int(picked.sum()), a.at, t, note))
    print("check: %d of %d pixels changed%s, the rest (%.2f%%) is identical to the original" % (int(changed.sum()), h * w, area, 100 * (1 - changed.mean())))


# ----------------------------------------------------------------------------- regions (the harness: lets an AI pick elements without pixel guessing)
def cmd_regions(a):
    """Number the distinct elements of a picture. Writes regions.png (the picture with numbered boxes) and regions.json (id, a point
    inside the element for `swap --at`, bounding box, colour, area) - an assistant looks at the numbered picture and names an id."""
    out = Path(a.outdir)
    refuse_inputs([a.input], [out / "regions.png", out / "regions.json"])
    rgb = load_rgb(a.input)
    h, w = rgb.shape[:2]
    if not (a.min_area >= 1 and 1 <= a.max <= 200):
        sys.exit("error: --min-area must be at least 1 and --max between 1 and 200")
    flat = cv2.pyrMeanShiftFiltering(np.ascontiguousarray(rgb), 6, 22) if h * w <= 2_000_000 else rgb   # smooth gradients/antialiasing so one element stays one region
    q = (flat // 48).astype(np.int32)
    key = q[..., 0] * 100 + q[..., 1] * 10 + q[..., 2]
    found = []
    for k in np.unique(key):
        m = cv2.morphologyEx((key == k).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # cut thin bridges: a heart must not stay glued to an eye
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            if area < a.min_area or area > .2 * h * w:
                continue
            comp = lab == i
            ring = (cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~comp
            if not ring.any():
                continue
            col = rgb[comp].astype(np.float32).mean(0)
            contrast = float(np.abs(rgb[ring].astype(np.float32).mean(0) - col).max())
            if contrast < 25:
                continue
            found.append((contrast * min(area, 120) ** .5, i, comp, area, col, st[i]))
    found.sort(key=lambda f: -f[0])
    picked, used = [], np.zeros((h, w), bool)
    for score, i, comp, area, col, st in found:               # best first; skip a candidate that mostly lies inside one already chosen
        if (comp & used).sum() > .5 * area:
            continue
        picked.append((comp, area, col, st))
        used |= comp
        if len(picked) >= a.max:
            break
    if not picked:
        sys.exit("error: no distinct elements found (try a lower --min-area)")
    scale = max(1, int(np.ceil(900 / max(h, w))))
    over = cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST) if scale > 1 else rgb.copy()
    info = []
    for n_, (comp, area, col, st) in enumerate(picked, 1):
        dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        py, px = np.unravel_index(int(dist.argmax()), dist.shape)
        x, y, bw, bh = int(st[cv2.CC_STAT_LEFT]), int(st[cv2.CC_STAT_TOP]), int(st[cv2.CC_STAT_WIDTH]), int(st[cv2.CC_STAT_HEIGHT])
        info.append({"id": n_, "at": "%d,%d" % (px, py), "box": [x, y, x + bw - 1, y + bh - 1], "area": area,
                     "colour": "#%02x%02x%02x" % tuple(int(round(v)) for v in col)})
        cv2.rectangle(over, (x * scale, y * scale), ((x + bw) * scale, (y + bh) * scale), (255, 255, 0), 1)
        cv2.putText(over, str(n_), (x * scale + 1, max(10, y * scale - 2)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(over, str(n_), (x * scale + 1, max(10, y * scale - 2)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1, cv2.LINE_AA)
    out.mkdir(parents=True, exist_ok=True)
    work, _ = scratch_beside(out / "regions.png")
    try:
        Image.fromarray(over).save(work / "regions.png")
        (work / "regions.json").write_text(json.dumps({"size": [w, h], "regions": info}, indent=1), encoding="utf-8")
        publish([(work / n, out / n) for n in ("regions.png", "regions.json")])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("wrote regions.png and regions.json (%d elements; use the id's 'at' with swap --at)" % len(info))


# ----------------------------------------------------------------------------- tools (machine-readable description for AI agents)
def cmd_tools(a):
    """Print every command as a tool definition (name, description, JSON schema) that an AI agent can register as is."""
    import io
    tools = []
    parser = build_parser()
    sub = next(act for act in parser._actions if isinstance(act, argparse._SubParsersAction))
    helps = {c.dest: c.help for c in sub._choices_actions}
    for name, sp in sub.choices.items():
        props, req = {}, []
        for act in sp._actions:
            if act.dest in ("help", "fn"):
                continue
            if act.option_strings:
                key = act.dest
                typ = "boolean" if act.nargs == 0 else ("number" if act.type in (float, int) else "string")
                props[key] = {"type": typ, "description": act.help or ""}
                if act.required:
                    req.append(key)
            else:
                props[act.dest] = {"type": "string", "description": "positional: " + act.dest}
                req.append(act.dest)
        tools.append({"name": "pixelcut_" + name, "description": helps.get(name) or name, "input_schema": {"type": "object", "properties": props, "required": req}})
    print(json.dumps({"tool_prefix": "pixelcut", "usage": "python pixelcut.py <command> <positional...> --option value", "tools": tools}, indent=1))


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

    s = sub.add_parser("swap", help="take an element out (pick it with one point), heal the place, optionally set a new shape there with an automatic shadow")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--at", required=True, help="x,y: a point on the element (its colour is picked, the connected area is taken); several points x,y;x,y add areas, e.g. a darker part over hair")
    s.add_argument("--tolerance", type=float, default=None, help="how far a colour may differ from the picked one and still belong to the element (default: automatic)")
    s.add_argument("--grow", type=int, default=3, help="px added around the element so its soft rim goes too")
    s.add_argument("--shape", default="none", help="none (just remove), diamond, square, circle, heart or star")
    s.add_argument("--size", type=float, default=1.0, help="size of the new shape relative to the element")
    s.add_argument("--colour", help="#rrggbb for the new shape (default: the colour of the removed element)")
    s.add_argument("--shadow", type=float, default=None, help="shadow strength 0..1 (default: measured from the original element)")
    s.add_argument("--flat", action="store_true", help="draw the new shape flat: no shading and no outline taken from the original")
    s.add_argument("--cut", help="also save the removed element (exact pixels) as a transparent png")
    s.add_argument("--flat-surface", action="store_true", help="do not bend the new shape to the surface (it is detected automatically)")
    s.set_defaults(fn=cmd_swap)

    s = sub.add_parser("regions", help="number the distinct elements of a picture (regions.png + regions.json) so an assistant can pick one by id")
    s.add_argument("input"); s.add_argument("outdir")
    s.add_argument("--min-area", type=int, default=12, help="smallest element in px")
    s.add_argument("--max", type=int, default=60, help="most elements to number")
    s.set_defaults(fn=cmd_regions)

    s = sub.add_parser("tools", help="print all commands as tool definitions (JSON) for AI agents")
    s.set_defaults(fn=cmd_tools)

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

    s = sub.add_parser("batch", help="run alpha/labels/trace/upscale/split/key over every picture of a folder",
                       description="pixelcut batch [--recursive] [--overwrite] COMMAND INPUT_DIR OUTPUT_DIR [options of COMMAND]")
    s.add_argument("--recursive", action="store_true", help="also pictures in sub-folders (the folder layout is kept)")
    s.add_argument("--overwrite", action="store_true", help="replace outputs that already exist (default: they are skipped)")
    s.add_argument("command"); s.add_argument("input_dir"); s.add_argument("output_dir")
    s.add_argument("rest", nargs=argparse.REMAINDER, help="options passed to COMMAND, e.g. --scale=2")
    s.set_defaults(fn=cmd_batch)
    return p


# ----------------------------------------------------------------------------- batch
BATCH_PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif")
BATCH_OUTPUT = {"alpha": ".png", "labels": ".png", "trace": ".svg", "upscale": None,          # None: the input's ending, "dir": one folder per picture
                "split": "dir", "key": "dir"}
BATCH_DIR_DONE = {"split": ("parts.json",), "key": ("base.png", "element.png", "preview.png")}


def place_chain(path):
    """Identity of a place and of every folder above it - by (device, file id) of what exists, not by spelling. Long-path, UNC and forward-slash spellings, 8.3 names, junctions and links all name the same place and give the same chain. Not-yet-existing tails are
    keyed by their parts under the deepest existing folder."""
    p = Path(os.path.abspath(str(path)))
    try:
        p = p.resolve()
    except OSError:
        pass
    tail, chain, node = [], [], p
    while True:
        try:
            st = os.stat(str(node))
            base = ("e", st.st_dev, st.st_ino)
            break
        except OSError:
            if node.parent == node:
                return [("p", os.path.normcase(str(p)))]
            tail.append(os.path.normcase(node.name))
            node = node.parent
    chain = [("n", base, tuple(reversed(tail[i:]))) if tail[i:] else base for i in range(len(tail) + 1)]
    while node.parent != node:                                # existing ancestors
        node = node.parent
        try:
            st = os.stat(str(node))
            chain.append(("e", st.st_dev, st.st_ino))
        except OSError:
            break
    return chain                                              # chain[0] = the place itself


def place_inside(path, folder_key):
    """True when `path` lies below the place `folder_key` (a chain[0] of the folder)."""
    return folder_key in place_chain(path)[1:]


def batch_pictures(folder, recursive, not_inside=None):
    """Pictures of a folder, sorted. Hidden files/folders are skipped, and so are folders that are links/junctions (they point somewhere
    else - and may point back into this folder), and the output folder when it lies inside the input folder."""
    found = []
    for root, dirs, files in os.walk(folder, followlinks=False):
        root = Path(root)
        dirs[:] = sorted(d for d in dirs if recursive and not d.startswith(".") and not link_like(root / d)
                         and not (not_inside and place_chain(root / d)[0] == not_inside))
        found += [root / f for f in sorted(files) if not f.startswith(".") and Path(f).suffix.lower() in BATCH_PICTURES]
    return found


def batch_is_done(command, target):
    """A finished output, not just an existing one: an interrupted split leaves a folder behind that must be done again."""
    if command in BATCH_DIR_DONE:
        return (target.is_dir() and all((target / n).is_file() for n in BATCH_DIR_DONE[command])
                and not any((target / w).exists() for w in (".pixelcut_new", ".pixelcut_old", DONE)))
    return target.is_file()


def cmd_batch(a):
    """One command over every picture of a folder. Each picture is its own run: one bad file never stops the others."""
    import warnings
    if a.command not in BATCH_OUTPUT:
        sys.exit("error: batch works with %s (not %s)" % (", ".join(sorted(BATCH_OUTPUT)), a.command))
    src, dst = Path(a.input_dir), Path(a.output_dir)
    if not src.is_dir():
        sys.exit("error: %s is not a folder" % src)
    src_key, dst_key = place_chain(src)[0], place_chain(dst)[0]
    if dst_key == src_key:
        sys.exit("error: the output folder must be a different folder than the input folder")
    if dst.exists() and not dst.is_dir():
        sys.exit("error: %s exists and is not a folder" % dst)
    try:                                                      # a wrong option (syntax) is wrong for every picture: say so once, before anything is written
        parsed = build_parser(_ApiParser).parse_args([a.command, "in.png", "out"] + list(a.rest))
    except PixelcutError as e:
        sys.exit("error: %s" % e)
    if getattr(parsed, "render", None):                       # checked on the parsed value: argparse accepts abbreviations such as --rend
        sys.exit("error: --render names one file - every picture would overwrite it; render the SVGs afterwards")
    inside = dst_key if place_inside(dst, src_key) else None
    pictures = batch_pictures(src, a.recursive, inside)
    if not pictures:
        sys.exit("error: no pictures (%s) in %s%s" % (", ".join(BATCH_PICTURES), src, "" if a.recursive else " (without --recursive only the folder itself is read)"))
    kind = BATCH_OUTPUT[a.command]
    plan = []                                                 # (picture, relative output name)
    for pic in pictures:
        rel = pic.relative_to(src)
        plan.append((pic, rel.with_suffix("") if kind == "dir" else rel.with_suffix(kind or pic.suffix)))
    names = {}
    for pic, name in plan:
        names.setdefault(name.as_posix().lower(), []).append(pic.relative_to(src).as_posix())
    chains = [place_chain(p) for p, _ in plan]
    inputs = {c[0] for c in chains}
    for pic, name in plan:                                    # an output that lands on (or above) an input would destroy it - also when dst is an ancestor of src
        tc = place_chain(dst / name)
        if tc[0] in inputs or (kind == "dir" and any(tc[0] in c for c in chains)) or (inside is None and src_key in tc[1:]):
            sys.exit("error: the output of %s would overwrite one of the input pictures (%s); use an output folder that does not contain the input folder" % (pic.relative_to(src).as_posix(), name.as_posix()))
    dst.mkdir(parents=True, exist_ok=True)
    done, skipped, failed = 0, 0, []
    for n, (pic, name) in enumerate(plan, 1):
        rel = pic.relative_to(src)
        label = "[%d/%d] %s" % (n, len(plan), rel)
        key = name.as_posix().lower()
        clash = names[key][0] if names[key][0] != rel.as_posix() else None
        if clash is None and kind == "dir":                   # a.png (-> folder a) and a/b.png (-> a/b) would share a folder
            clash = next((names[p.as_posix().lower()][0] for p in name.parents if p.as_posix().lower() in names and str(p) != "."), None)
        if clash:
            failed.append((rel.as_posix(), "its output %s is also the output of %s" % (name.as_posix(), clash)))
            print("%s: failed (same output as %s)" % (label, clash))
            continue
        target = dst / name
        if batch_is_done(a.command, target) and not a.overwrite:
            skipped += 1
            print("%s: skipped (%s exists, use --overwrite)" % (label, name.as_posix()))
            continue
        try:
            frames = 1
            try:
                from PIL import Image
                with Image.open(pic) as g:
                    frames = getattr(g, "n_frames", 1)
            except Exception:
                pass                                          # unreadable: the run itself reports it
            target.parent.mkdir(parents=True, exist_ok=True)
            with warnings.catch_warnings(record=True) as notes:
                warnings.simplefilter("always")
                said = _run(a.command, [pic, target, *a.rest], {})
            done += 1
            extra = [ln for ln in said.splitlines() if ln.strip() and not ln.startswith(("wrote", "rendered", "joined"))]
            print("%s: ok%s" % (label, "".join(" (%s)" % w.message for w in notes if issubclass(w.category, UserWarning))))
            for ln in extra:
                print("    " + ln)
            if frames > 1:
                print("    note: %s has %d frames - only the first one is used" % (rel.as_posix(), frames))
        except (PixelcutError, OSError) as e:
            msg = (oserror_message(e).removeprefix("error: ") if isinstance(e, OSError) and oserror_message(e) else str(e)).splitlines()[0]
            failed.append((rel.as_posix(), msg))
            print("%s: failed (%s)" % (label, msg))
    print("batch %s: %d done, %d skipped, %d failed" % (a.command, done, skipped, len(failed)))
    if failed:
        sys.exit("error: %d of %d pictures failed:\n  %s" % (len(failed), len(plan), "\n  ".join("%s: %s" % f for f in failed)))


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


def swap(input, output, at, **options):
    """Take the element at the point `at` (x,y) out and heal the place. Options: shape (none/diamond/square/circle/heart/star), size,
    colour, shadow, tolerance, grow, cut. Returns the output path."""
    if isinstance(at, (list, tuple)) and at and isinstance(at[0], (list, tuple)):
        at = ";".join("%d,%d" % tuple(pt) for pt in at)       # several points
    elif isinstance(at, (list, tuple)):
        at = "%d,%d" % tuple(at)
    _run("swap", [input, output], dict(options, at=at))
    return Path(output)


def regions(input, outdir, **options):
    """Number the distinct elements of a picture; returns the list of {id, at, box, area, colour}."""
    _run("regions", [input, outdir], options)
    return json.loads((Path(outdir) / "regions.json").read_text(encoding="utf-8"))["regions"]


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


def batch(command, input_dir, output_dir, recursive=False, overwrite=False, **options):
    """One command over every picture of a folder (see `pixelcut batch`). Returns the printed progress; raises PixelcutError if a picture
    failed (the message names each one). `options` are the options of the command, e.g. batch("upscale", "in", "out", scale=2)."""
    own = (["--recursive"] if recursive else []) + (["--overwrite"] if overwrite else [])        # batch's own flags must come first
    return _run("batch", own + [command, input_dir, output_dir] + _flags(options), {})


def upscale(input, output, **options):
    """Upscale (options: scale, model, bin, no_ai). Returns the output path. If the AI engine cannot run, a plain Lanczos resize is
    made instead and a UserWarning says so (the command line prints the same sentence)."""
    import warnings
    said = _run("upscale", [input, output], options)
    fell_back = next((ln for ln in said.splitlines() if "plain Lanczos" in ln), None)
    if fell_back and not options.get("no_ai"):
        warnings.warn(fell_back, UserWarning, stacklevel=2)
    return Path(output)


__all__ = ["__version__", "PixelcutError", "alpha", "split", "labels", "join", "key", "swap", "regions", "compare", "trace", "upscale", "batch"]


if __name__ == "__main__":
    sys.exit(main())
