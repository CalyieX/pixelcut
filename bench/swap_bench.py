"""Mess-Pruefstand fuer `pixelcut swap --shape none` mit synthetischer Wahrheit.

Pro Fall wird ein Bild gebaut (Hintergrund + duenne Straehne), dann ein Objekt eingesetzt, ueber dem die Straehne liegt und das einen Schatten
der Straehne traegt. Das Bild OHNE Objekt ist die Wahrheit. pixelcut nimmt das Objekt heraus; gemessen wird gegen die Wahrheit.
Schwellen sind vorab festgelegt (THRESH) und werden nicht pro Lauf angepasst.
Aufruf:  python bench/swap_bench.py [--json out.json] [--quick] [--twice]
"""
import argparse, contextlib, io, json, math, sys, tempfile
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pixelcut

SS = 4
# eingefroren 11.11.2026 (Spock-Review); strand_iou/edge_px: Erststand, wird nach der Baseline verschaerft
THRESH = {"outside_changed": 0, "residual": .005, "dE_hole_mean": 6.0, "strand_iou": .6, "edge_px": 1.8}


def background(kind, h, w, rng):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "skin":
        base = np.array([245, 215, 200], np.float32) - np.stack([(xx / w) * 18, (yy / h) * 26, (xx / w) * 10], -1)
    elif kind == "photo":
        base = np.array([205, 170, 150], np.float32) - np.stack([(xx / w) * 30, (yy / h) * 36, (xx / w) * 24], -1)
        base = base + cv2.GaussianBlur(rng.normal(0, 7, (h, w, 3)).astype(np.float32), (0, 0), 1.1)
    elif kind == "anime":
        base = np.zeros((h, w, 3), np.float32) + np.array([252, 238, 232], np.float32)
        base[:, w // 2:] -= np.array([6, 14, 10], np.float32)
    else:
        base = np.zeros((h, w, 3), np.float32) + np.array([236, 236, 244], np.float32)
    return np.clip(base, 0, 255)


def shape_mask(kind, h, w, cx, cy, r):
    m = np.zeros((h * SS, w * SS), np.uint8)
    c = (int(cx * SS), int(cy * SS)); R = int(r * SS)
    if kind == "circle":
        cv2.circle(m, c, R, 255, -1, cv2.LINE_AA)
    elif kind == "square":
        cv2.rectangle(m, (c[0] - R, c[1] - R), (c[0] + R, c[1] + R), 255, -1)
    else:
        t = np.linspace(0, 2 * math.pi, 240)
        hx, hy = 16 * np.sin(t) ** 3, 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        cv2.fillPoly(m, [np.array(list(zip(c[0] + hx / 17 * R, c[1] - hy / 17 * R)), np.int32)], 255, cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def strand_mask(h, w, cx, cy, r, ang, wid):
    m = np.zeros((h * SS, w * SS), np.uint8)
    d = np.array([math.cos(ang), math.sin(ang)]); L = r * 2.6
    p0, p1 = np.array([cx, cy]) - d * L, np.array([cx, cy]) + d * L
    cv2.line(m, tuple(int(v * SS) for v in p0), tuple(int(v * SS) for v in p1), 255, max(1, int(wid * SS)), cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255, d


def make_case(kind, shp, r, seed):
    rng = np.random.default_rng(seed)
    size = int(r * 5)
    h, w = int(size * .9), size
    cx, cy = w * .5 + rng.uniform(-r * .2, r * .2), h * .5 + rng.uniform(-r * .2, r * .2)
    bg = background(kind, h, w, rng)
    obj_col = [np.array([236, 60, 90]), np.array([60, 110, 220]), np.array([60, 170, 90])][seed % 3].astype(np.float32)
    om = shape_mask(shp, h, w, cx, cy, r)
    ang = math.radians(rng.uniform(55, 125)) if seed % 2 else math.radians(rng.uniform(-125, -55))
    sm, d = strand_mask(h, w, cx + rng.uniform(-r * .15, r * .15), cy, r, ang, max(1.6, r / 10))
    strand_col = np.array([70, 50, 56], np.float32)
    # Schatten der Straehne auf dem Objekt: Straehne nach rechts-unten geschoben, nur auf dem Objekt, dunkler
    M = np.float32([[1, 0, r / 5], [0, 1, r / 8]])
    shifted = cv2.warpAffine(sm, M, (w, h))
    sh = shifted * (1 - sm) * om
    truth = bg * (1 - sm[..., None]) + strand_col * sm[..., None]
    img = bg * (1 - om[..., None]) + obj_col * om[..., None]
    img = img * (1 - .38 * sh[..., None])
    img = img * (1 - sm[..., None]) + strand_col * sm[..., None]
    ys, xs = np.nonzero((om > .99) & (sm < .01) & (shifted < .01))
    j = len(xs) // 2
    return (np.clip(img, 0, 255).round().astype(np.uint8), np.clip(truth, 0, 255).round().astype(np.uint8), om, sm, (int(xs[j]), int(ys[j])))


def run_swap(path_in, path_out, at):
    argv = ["pixelcut", "swap", str(path_in), str(path_out), "--at", "%d,%d" % at, "--shape", "none"]
    old = sys.argv; sys.argv = argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            pixelcut.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            return None, str(e.code)
    finally:
        sys.argv = old
    return np.array(Image.open(path_out).convert("RGB")), buf.getvalue()


def lab(a):
    return cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)


def edge_width(sa):
    """mittlere Kantenbreite in px: Flaeche der Uebergangspixel (0.1..0.9) / Laenge der Kontur"""
    m = (sa > .5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    L = sum(len(c) for c in cnts)
    return float(((sa > .1) & (sa < .9)).sum()) / max(L, 1)


def measure(img, truth, out, om, sm):
    hole = cv2.dilate((om > .02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    far = ~(cv2.dilate(hole.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
    res = {"outside_changed": int((out != img)[far].any(-1).sum())}
    de = np.linalg.norm(lab(out) - lab(truth), axis=2)
    hm = (om > .5) & (sm < .05)
    res["dE_hole_mean"] = float(de[hm].mean()) if hm.any() else 0.0
    res["dE_hole_p95"] = float(np.percentile(de[hm], 95)) if hm.any() else 0.0
    ocol = np.linalg.norm(lab(img) - lab(truth), axis=2)
    res["residual"] = float(((de > 12) & (ocol > 12) & hm).sum() / max(hm.sum(), 1))
    lum = lambda a: a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114
    zone = (cv2.dilate((om > .02).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    got = (lum(out.astype(np.float32)) < 120) & zone
    ref = (sm > .5) & zone
    res["strand_iou"] = float((got & ref).sum() / max((got | ref).sum(), 1))
    # Kantenbreite der Straehne: nur dort messen, wo die Wahrheit die Straehne hat (Textur im Foto zaehlt nicht als Kante); Deckkraft aus dem Dunkelwert
    base = lum(truth.astype(np.float32))
    sa = np.clip((np.median(base[zone & (sm < .02)]) - lum(out.astype(np.float32))) / max(np.median(base[zone & (sm < .02)]) - 60, 1), 0, 1)
    near_strand = cv2.dilate((sm > .02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    res["edge_px"] = edge_width(sa * (zone & near_strand))
    return res


def verdict(res):
    return {"outside_changed": res["outside_changed"] <= THRESH["outside_changed"], "residual": res["residual"] <= THRESH["residual"],
            "dE_hole_mean": res["dE_hole_mean"] <= THRESH["dE_hole_mean"], "strand_iou": res["strand_iou"] >= THRESH["strand_iou"],
            "edge_px": res["edge_px"] <= THRESH["edge_px"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--twice", action="store_true")
    a = ap.parse_args()
    kinds = ("skin", "anime") if a.quick else ("skin", "photo", "anime", "flat")
    shapes = ("heart", "circle") if a.quick else ("heart", "circle", "square")
    radii = (12,) if a.quick else (10, 14, 24)
    rows, tmp = [], Path(tempfile.mkdtemp(prefix="swapbench_"))
    seed = 0
    for kind in kinds:
        for shp in shapes:
            for r in radii:
                seed += 1
                img, truth, om, sm, at = make_case(kind, shp, r, seed)
                pin, pout = tmp / ("c%d.png" % seed), tmp / ("o%d.png" % seed)
                Image.fromarray(img).save(pin)
                out, log = run_swap(pin, pout, at)
                name = "%s/%s/r%d" % (kind, shp, r)
                if out is None:
                    rows.append({"case": name, "error": str(log)[:90]}); print(name, "ERROR", str(log)[:90]); continue
                res = measure(img, truth, out, om, sm)
                if a.twice:
                    out2, _ = run_swap(pin, tmp / ("o%da.png" % seed), at)
                    res["deterministic"] = bool(out2 is not None and (out2 == out).all())
                v = verdict(res)
                res.update(case=name, passed=all(v.values()), failed=[k for k, ok in v.items() if not ok])
                rows.append(res)
                print("%-22s dE %.1f/%.1f resid %.3f iou %.2f edge %.2f out %d %s" % (name, res["dE_hole_mean"], res["dE_hole_p95"], res["residual"], res["strand_iou"], res["edge_px"], res["outside_changed"], "OK" if res["passed"] else "FAIL " + ",".join(res["failed"])))
    ok = [r for r in rows if r.get("passed")]
    print("\n%d von %d Faellen innerhalb der Schwellen" % (len(ok), len(rows)))
    if a.json:
        Path(a.json).write_text(json.dumps({"version": pixelcut.__version__, "thresholds": THRESH, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
