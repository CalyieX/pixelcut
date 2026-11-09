"""Browser check of the web version (docs/): split + join inside a real browser engine must give the original back exactly.

    python tests/web_check.py webkit      # the engine of iPhone/Safari
    python tests/web_check.py chromium
    python tests/web_check.py firefox

Needs `pip install playwright` and `playwright install <engine>`. Uses a local web server, loads Pyodide from the CDN.
"""
import functools
import http.server
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent.parent / "docs"

JS = """
async (grid) => {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 240 && document.getElementById("go").disabled; i++) await wait(500);
  if (document.getElementById("go").disabled) return {error: "engine never became ready: " + document.getElementById("status").textContent};
  const c = document.createElement("canvas"); c.width = 220; c.height = 160;
  const ctx = c.getContext("2d"); const id = ctx.createImageData(220, 160);
  let s = 4242; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let i = 0; i < id.data.length; i += 4) { id.data[i] = rnd() * 256; id.data[i+1] = rnd() * 256; id.data[i+2] = rnd() * 256; id.data[i+3] = 255; }
  ctx.putImageData(id, 0, 0);
  const blob = await new Promise((r) => c.toBlob(r, "image/png"));
  const orig = new Uint8Array(await blob.arrayBuffer());
  const f = document.getElementById("files");
  const pick = (cmd) => { document.getElementById("cmd").value = cmd; document.getElementById("cmd").dispatchEvent(new Event("change")); };
  const press = async () => { document.getElementById("go").click(); await wait(300); for (let i = 0; i < 240 && document.getElementById("go").disabled; i++) await wait(250); };
  pick("split");
  let dt = new DataTransfer(); dt.items.add(new File([blob], "o.png")); f.files = dt.files;
  document.getElementById("angles").value = "10,30,200"; document.getElementById("depth").value = "9"; document.getElementById("grid").value = grid;
  await press();
  const splitStatus = document.getElementById("status").textContent;
  const names = py.FS.readdir("/work/out/parts").filter((n) => !n.startsWith("."));
  const parts = names.map((n) => [n, py.FS.readFile("/work/out/parts/" + n)]);
  pick("join");
  dt = new DataTransfer(); for (const [n, b] of parts) dt.items.add(new File([b], n)); f.files = dt.files;
  await press();
  py.FS.writeFile("/work/orig.png", orig);
  const diff = py.runPython(`
from PIL import Image
import numpy as np
a = np.array(Image.open("/work/orig.png").convert("RGB")).astype(int)
b = np.array(Image.open("/work/out/joined.png").convert("RGB")).astype(int)
int(abs(a - b).max()) if a.shape == b.shape else -1
`);
  return {names, splitStatus, joinStatus: document.getElementById("status").textContent, diff};
}
"""


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "chromium"
    grid = sys.argv[2] if len(sys.argv) > 2 else ""            # e.g. 3x2: cut into a grid instead of wedges
    expect = (int(grid.split("x")[0]) * int(grid.split("x")[1]) if grid else 3) + 1
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(Quiet, directory=str(DOCS))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/" % srv.server_address[1]
    with sync_playwright() as p:
        browser = getattr(p, engine).launch()
        page = browser.new_page()
        page.set_default_timeout(300_000)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        res = page.evaluate(JS, grid)
        browser.close()
    srv.shutdown()
    print(engine, res)
    if errors:
        print("page errors:", errors)
    if res.get("error") or res.get("diff") != 0 or len(res.get("names", [])) != expect:
        sys.exit("FAILED: the web version did not give the original back exactly in " + engine)
    print("OK: %s - split and join in the browser are exact (max deviation 0)" % engine)


if __name__ == "__main__":
    main()
