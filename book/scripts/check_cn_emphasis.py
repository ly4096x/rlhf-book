#!/usr/bin/env python3
"""List lines where CommonMark (markdown-it, as used by colloquium) fails to parse **bold** / *italic*.

CJK text often puts punctuation right inside the marks, e.g. `**2017 年：**Transformer`; CommonMark's
flanking rules then render the asterisks literally. Fix by moving the punctuation outside the marks
(`**2017 年**：Transformer`). Usage: uv run --extra teach python book/scripts/check_cn_emphasis.py teach/cn
"""
import re, sys
from pathlib import Path
from markdown_it import MarkdownIt
md = MarkdownIt('commonmark', {'html': True}).enable('table')
def scan(path, skip_fm=True):
    lines = path.read_text().split('\n')
    i = 0
    if lines and lines[0].strip() == '---':
        i = next((j for j in range(1, len(lines)) if lines[j].strip() == '---'), 0) + 1
    fence = False; box = False; math = False; bad = []
    for n in range(i, len(lines)):
        l = lines[n]
        if l.lstrip().startswith('```'):
            fence = not fence
            box = fence and l.lstrip().startswith('```box')
            continue
        if fence and not box: continue
        if l.strip() == '$$': math = not math; continue
        if math or not l.strip() or l.strip().startswith('<!--'): continue
        txt = re.sub(r'\$[^$\n]+\$', 'M', l)      # ignore math
        txt = re.sub(r'`[^`]*`', 'C', txt)
        if '*' not in txt and '__' not in txt: continue
        html = md.renderInline(re.sub(r'^\s*([-*+]|\d+\.)\s+', '', txt))
        if re.search(r'\*\*|(?<!\w)\*(?!\s)|__', re.sub(r'<[^>]+>', '', html)):
            bad.append((n + 1, l))
    return bad
if __name__ == '__main__':
    tot = 0
    for p in sys.argv[1:]:
        for f in sorted(Path(p).rglob('*.md')) if Path(p).is_dir() else [Path(p)]:
            b = scan(f)
            tot += len(b)
            for n, l in b[:60]: print(f'{f}:{n}: {l[:110]}')
    print('total bad lines:', tot)
