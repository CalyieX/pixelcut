"""Fremdbild-Pruefstand (Spock Schritt 5): echte Fotos (skimage/sklearn, lokal mitgeliefert) als Grund, Form + Straehne daraufgestempelt.
Wahrheit = das Foto mit Straehne. Gezaehlt: Fall bestanden (Schwellen wie swap_bench) UND ehrliche Meldung: ein verschwiegener
Fehler (Fall durchgefallen, Urteil 'ok') zaehlt als Versagen, ein gemeldeter nicht."""
import argparse, contextlib, io, json, sys, tempfile
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
import swap_bench as sb
import skimage.data as sd, sklearn.datasets as skd, os

def photos():
    out = {}
    for n in ("astronaut", "chelsea", "coffee", "rocket", "cat", "lily", "retina", "brick", "grass", "gravel"):
        try: out[n] = getattr(sd, n)()               # only what is bundled offline; the rest is skipped
        except Exception: pass
    d = os.path.join(os.path.dirname(skd.__file__), "images")
    for f in ("china.jpg", "flower.jpg"):
        out[f] = np.array(Image.open(os.path.join(d, f)).convert("RGB"))
    return {k: (v if v.ndim == 3 else np.stack([v] * 3, -1))[..., :3].astype(np.uint8) for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--json")
    a = ap.parse_args(); P = photos(); names = sorted(P); rows = []
    for ni, name in enumerate(names):
        for s in range(a.seeds):
            for shp, r in (("heart", 14), ("circle", 12)):
                seed = 5000 + ni * 100 + s
                rng = np.random.default_rng(seed); src = P[name]
                size = int(r * 5); h, w = int(size * .9), size
                y0 = int(rng.integers(0, max(1, src.shape[0] - h))); x0 = int(rng.integers(0, max(1, src.shape[1] - w)))
                crop = src[y0:y0 + h, x0:x0 + w].astype(np.float32)
                sb.background = lambda kind, hh, ww, rg, c=crop: c.copy()
                img, truth, om, sm, at, bg = sb.make_case("photo", shp, r, seed)
                with tempfile.TemporaryDirectory() as td:
                    pin, pout = Path(td) / "i.png", Path(td) / "o.png"
                    Image.fromarray(img).save(pin)
                    argv = ["pixelcut", "swap", str(pin), str(pout), "--at", "%d,%d" % at, "--shape", "none"]
                    old = sys.argv; sys.argv = argv; buf = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(buf): sb.pixelcut.main()
                    except SystemExit: pass
                    finally: sys.argv = old
                    if not pout.exists(): rows.append({"case": (name, shp, seed), "ok": False, "said_ok": False, "note": "no output"}); continue
                    out = np.array(Image.open(pout).convert("RGB"))
                res = sb.measure(img, truth, out, om, sm, bg); v = sb.verdict(res); good = all(v.values())
                said_ok = "verdict: ok" in buf.getvalue()
                rows.append({"case": (name, shp, seed), "pass": good, "said_ok": said_ok, "silent_fail": (not good) and said_ok, "dE": round(res["dE_hole_mean"], 2),
                             "failed": [k for k, x in v.items() if not x]})
                print("%-12s %-6s %d  %s  dE %.1f  verdict-ok %s  %s" % (name, shp, seed, "PASS" if good else "FAIL", res["dE_hole_mean"], said_ok, "SILENT" if (not good) and said_ok else ""), flush=True)
    n = len(rows); k = sum(r.get("pass", False) for r in rows); silent = sum(r.get("silent_fail", False) for r in rows)
    honest = n - silent
    lo, hi = sb.wilson(honest, n); lo2, hi2 = sb.wilson(k, n)
    print("\n%d Faelle: %d innerhalb der Schwellen (%.2f-%.2f); ehrlich (kein verschwiegener Fehler): %d (%.2f-%.2f); verschwiegen: %d" % (n, k, lo2, hi2, honest, lo, hi, silent))
    if a.json: Path(a.json).write_text(json.dumps(rows, indent=1))
if __name__ == "__main__": main()
