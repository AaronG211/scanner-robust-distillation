#!/bin/zsh
# Build main.pdf with the official MIDL template (midl.cls) and report the page of the last main-text line.
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { tail -30 build.log; exit 1; }
grep -E "Overfull|undefined|Citation .* undefined|Reference .* undefined" main.log | sort | uniq -c | head -20
python3 - <<'PY'
import re, subprocess
txt = subprocess.run(["pdftotext", "-layout", "main.pdf", "-"], capture_output=True, text=True).stdout
pages = txt.split("\f")
def page_of(s):
    for i, p in enumerate(pages, 1):
        if s in p: return i
print("pages total:", len(pages) - (1 if pages[-1].strip() == "" else 0))
print("main text ends on page:", page_of("mostly from a few unlabelled images"), "(limit 10); references start on page:", page_of("References"))
PY
