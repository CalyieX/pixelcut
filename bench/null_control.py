"""Nullkontrolle + Vergleichsmarken fuer den Fremdbild-Pruefstand (Spock 17.11.2026).
Eine Messgroesse taugt nur, wenn sie trennt: 'nichts tun' MUSS durchfallen, die Wahrheit MUSS bestehen, cv2.inpaint ist die Untergrenze,
die Wahrheit mit neu gezogener Textur (Rauschen) ist die Obergrenze fuer Textur-Faelle. Gezaehlt nach Fotos, nicht nach Faellen."""
import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
import swap_bench as sb, real_bench as rb

def main():
    P = rb.photos(); rows = []
    for ni, name in enumerate(sorted(P)):
        for shp, r in (("heart", 14), ("circle", 12)):
            seed = 5000 + ni * 100
            rng = np.random.default_rng(seed); src = P[name]; size = int(r * 5); h, w = int(size * .9), size
            y0 = int(rng.integers(0, max(1, src.shape[0] - h))); x0 = int(rng.integers(0, max(1, src.shape[1] - w)))
            crop = src[y0:y0 + h, x0:x0 + w].astype(np.float32); sb.background = lambda k, hh, ww, rg, c=crop: c.copy()
            img, truth, om, sm, at, bg = sb.make_case("photo", shp, r, seed)
            hole = (cv2.dilate((om > .02).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0).astype(np.uint8)
            inp = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), hole, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
            noisy = np.clip(truth.astype(np.float32) + np.random.default_rng(seed + 1).normal(0, 3, truth.shape), 0, 255).astype(np.uint8)
            res = {}
            for k, out in (("nichts_tun", img), ("wahrheit", truth), ("wahrheit_neue_textur", noisy), ("cv2_inpaint", inp)):
                m = sb.measure(img, truth, out, om, sm, bg); v = sb.verdict(m)
                res[k] = {"dE": round(m["dE_hole_mean"], 2), "residual": round(m["residual"], 4), "pass": all(v.values()), "failed": [a for a, b in v.items() if not b]}
            rows.append({"foto": name, "form": shp, **res})
            print("%-11s %-6s nichts_tun:%-5s wahrheit:%-5s neue_textur:%-5s inpaint:%-5s (dE %.1f/%.1f)" % (name, shp, res["nichts_tun"]["pass"], res["wahrheit"]["pass"], res["wahrheit_neue_textur"]["pass"], res["cv2_inpaint"]["pass"], res["nichts_tun"]["dE"], res["cv2_inpaint"]["dE"]), flush=True)
    n = len(rows)
    for k in ("nichts_tun", "wahrheit", "wahrheit_neue_textur", "cv2_inpaint"):
        print("%-22s besteht %d von %d" % (k, sum(r[k]["pass"] for r in rows), n))
    Path(__file__).with_name("null_control_result.json").write_text(json.dumps(rows, indent=1))
if __name__ == "__main__": main()
