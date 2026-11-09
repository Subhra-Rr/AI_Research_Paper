import urllib.request
import os

template_dir = r"D:\AI_Research_Paper\input\template_jmlr"
os.makedirs(template_dir, exist_ok=True)

files = [
    ("https://raw.githubusercontent.com/JmlrOrg/jmlr-style-file/master/jmlr2e.sty", "jmlr2e.sty"),
    ("https://raw.githubusercontent.com/JmlrOrg/jmlr-style-file/master/sample.tex", "sample.tex")
]

for url, fname in files:
    out_path = os.path.join(template_dir, fname)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        f.write(resp.read())
    print(f"Downloaded {fname}: {os.path.getsize(out_path)} bytes")
