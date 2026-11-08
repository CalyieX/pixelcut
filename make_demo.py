"""Makes the demo picture (own drawing, no third-party art): a yellow comic burst with red lettering on white."""
import math
import sys

from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1] if len(sys.argv) > 1 else "demo.png"
W, H = 600, 480
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
cx, cy = W / 2, H / 2
pts = []
for i in range(28):
    a = math.pi * 2 * i / 28
    rr = 205 if i % 2 == 0 else 120
    pts.append((cx + rr * math.cos(a) * 1.15, cy + rr * math.sin(a) * .95))
d.polygon(pts, fill=(255, 221, 0), outline=(20, 16, 14), width=6)
try:
    f = ImageFont.truetype("arialbd.ttf", 96)
except OSError:
    f = ImageFont.load_default()
d.text((cx, cy), "POW", fill=(226, 36, 28), anchor="mm", font=f, stroke_width=5, stroke_fill=(20, 16, 14))
im.save(out)
