#!/usr/bin/env python3
"""Generate the Chinese (CN) page templates from the upstream English ones.

CN chapters are served from /c/cn/, so upstream's root-absolute links
(/c/01-introduction, /assets/..., /pagefind/...) must become relative, and
UI labels are translated. Deriving the files at build time means upstream
template changes flow into the CN site without hand-merging.

Usage: make_cn_templates.py <output-dir>
Writes: chapter-cn.html, footer.html, nav-cn.js
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "book" / "templates"
UP = "../../"  # /c/cn/ -> site root

NAV_LABELS = [
    # headings / chrome
    ("<strong>Navigation</strong>", "<strong>导航</strong>"),
    ("<h3>Links</h3>", "<h3>链接</h3>"),
    ("<h3>Resources</h3>", "<h3>资源</h3>"),
    ("<h3>Introductions</h3>", "<h3>引言</h3>"),
    ("<h3>Core Training Pipeline</h3>", "<h3>核心训练流程</h3>"),
    ("<h3>Data & Preferences</h3>", "<h3>数据与偏好</h3>"),
    ("<h3>Practical Considerations</h3>", "<h3>实践考量</h3>"),
    ("<h3>Appendices</h3>", "<h3>附录</h3>"),
    (">Home</a>", ">主页</a>"),
    ("<li>Order: ", "<li>购买："),
    (">RL Cheatsheet<", ">RL 速查表<"),
    (">Compare Model Completions<", ">模型补全对比<"),
    (">Accompanying Course<", ">配套课程<"),
    # chapters
    (">Introduction<", ">引言<"),
    (">A Tiny History of RLHF<", ">RLHF 简史<"),
    (">Training Overview<", ">训练概览<"),
    (">Instruction Fine-Tuning<", ">指令微调<"),
    (">Reward Modeling<", ">Reward Modeling<"),
    (">Reinforcement Learning<", ">强化学习<"),
    (">Reasoning and Inference-Time Scaling<", ">推理与推理时扩展<"),
    (">Direct-Alignment Algorithms<", ">直接对齐算法<"),
    (">Rejection Sampling<", ">拒绝采样<"),
    (">The Nature of Preferences<", ">偏好的本质<"),
    (">Preference Data<", ">偏好数据<"),
    (">Synthetic Data &amp; Distillation<", ">合成数据与蒸馏<"),
    (">Tool Use and Function Calling<", ">工具使用与函数调用<"),
    (">Over-Optimization<", ">过度优化<"),
    (">Regularization<", ">正则化<"),
    (">Evaluation<", ">评估<"),
    (">Model Character & Products<", ">模型性格与产品<"),
    (">Definitions<", ">定义<"),
    (">Beyond \"Just Style\"<", ">不止于“风格”<"),
    (">Practical Issues<", ">实践问题<"),
    (">code</a>", ">代码</a>"),
]

CHAPTER_LABELS = [
    ("<h3> Chapter Contents </h3>", "<h3> 章节目录 </h3>"),
    ("← Previous: $prev-chapter$", "← 上一章：$prev-chapter$"),
    ("Next: $next-chapter$ →", "下一章：$next-chapter$ →"),
    (">Citation</h4>", ">引用</h4>"),
    ("<p>If you found this useful for your research, please cite it!</p>",
     "<p>如果这本书对你的研究有帮助，请引用：</p>"),
    ("<p>For the web and arXiv version:</p>", "<p>网页版与 arXiv 版本：</p>"),
    ("<p>For the Manning edition:</p>", "<p>Manning 版本：</p>"),
]


def replace_checked(text, pairs, what):
    """Apply literal replacements; warn (don't fail) when upstream drifted."""
    for old, new in pairs:
        if old not in text:
            print(f"warning: {what}: pattern not found: {old!r}", file=sys.stderr)
            continue
        text = text.replace(old, new)
    return text


def make_nav(src):
    # Chapter links become siblings; other root-absolute links climb to the site root.
    src = re.sub(r'href="/c/([^"]+)"', r'href="\1"', src)
    src = re.sub(r'href="/([^"]*)"', lambda m: f'href="{UP}{m.group(1)}"', src)
    src = replace_checked(src, NAV_LABELS, "nav.js")
    # markCurrentPage only handled root-absolute hrefs; resolve relative ones too.
    src = src.replace("""querySelectorAll('a[href^="/"]')""", "querySelectorAll('a[href]')")
    src = src.replace(
        "new URL(link.getAttribute('href'), window.location.origin).pathname",
        "new URL(link.getAttribute('href'), window.location.href).pathname",
    )
    return src


def make_chapter(src):
    src = src.replace('href="/pagefind/', f'href="{UP}pagefind/')
    src = src.replace('src="/pagefind/', f'src="{UP}pagefind/')
    src = src.replace('<script src="nav.js"', '<script src="nav-cn.js"')
    src = src.replace(
        '<a href="https://rlhfbook.com/" style="color: inherit;',
        f'<a href="{UP}" style="color: inherit;',
    )
    return replace_checked(src, CHAPTER_LABELS, "chapter.html")


def make_footer(src):
    src = src.replace('src="/assets/', f'src="{UP}assets/')
    src = src.replace('data-dark-src="/assets/', f'data-dark-src="{UP}assets/')
    note = (
        '  <p style="color: var(--color-text-faint);">中文翻译 &middot; '
        '<a href="https://github.com/ly4096x/rlhf-book" style="color: inherit;">ly4096x/rlhf-book</a>'
        ' &middot; 原书版权归 Nathan Lambert 所有</p>\n</footer>'
    )
    return src.replace("</footer>", note)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "chapter-cn.html").write_text(make_chapter((TEMPLATES / "chapter.html").read_text()))
    (out / "footer.html").write_text(make_footer((TEMPLATES / "footer.html").read_text()))
    (out / "nav-cn.js").write_text(make_nav((TEMPLATES / "nav.js").read_text()))


if __name__ == "__main__":
    main()
