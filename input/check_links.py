import urllib.request
import re

url = "https://www.jmlr.org/format/format.html"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    print("Page fetched. Length:", len(html))
    links = re.findall(r'href=[\'"](.*?)[\'"]', html)
    for l in links:
        if any(w in l.lower() for w in ['sty', 'tex', 'zip', 'tar', 'cls', 'template', 'doc']):
            print("Link:", l)
except Exception as e:
    print("Error:", e)
