import urllib.request, re
html = urllib.request.urlopen("https://ar5iv.labs.arxiv.org/html/2107.05612", timeout=60).read().decode("utf-8", "replace")
srcs = re.findall(r"src=\"([^\"]+)\"", html)
for s in srcs:
    if "png" in s or "jpg" in s or "svg" in s or "fig" in s:
        print(s)
