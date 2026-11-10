import urllib.request, re
html = urllib.request.urlopen("https://ar5iv.labs.arxiv.org/html/2107.05612", timeout=60).read().decode("utf-8", "replace")
paths = sorted(set(re.findall(r"/html/2107\.05612/assets/[^\"'\s>]+", html)))
for p in paths:
    print(p)
print("--- count", len(paths))
