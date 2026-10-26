"""Post-process the pdoc pages in docs/ (run by docs/build_docs.sh after pdoc).

* Highlights the code blocks pdoc leaves plain (indented examples in docstrings) with Pygments,
  in the markup pdoc uses for fenced blocks; shell commands get a shell lexer, the rest Python.
* Adds the theme in docs/_theme.css: system UI font, a calmer type scale, cards for each class
  and function, styled tables, the accent colour of the README figures, and a dark mode that
  follows the reader's system setting (with its own syntax colours).
* Adds a copy button to every code block.

Idempotent: a page already processed is left as it is.

    uv run --group dev python docs/_postprocess.py docs
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, PythonLexer

MARK = "<!-- mongopredict-theme -->"
SHELL = re.compile(r"^\s*(\$ |pip |uv |bash |export |cd |git |mongod|python -m )")

THEME = Path(__file__).with_name("_theme.css")

COPY_JS = """
document.querySelectorAll('.pdoc .pdoc-code').forEach(function (box) {
  if (box.querySelector('.mp-copy')) return;
  var b = document.createElement('button');
  b.className = 'mp-copy'; b.type = 'button'; b.textContent = 'Copy';
  b.addEventListener('click', function () {
    var text = box.querySelector('pre').innerText;
    navigator.clipboard.writeText(text).then(function () {
      b.textContent = 'Copied'; b.classList.add('done');
      setTimeout(function () { b.textContent = 'Copy'; b.classList.remove('done'); }, 1400);
    });
  });
  box.appendChild(b);
});
"""


def _syntax_css() -> str:
    """Light syntax colours as pdoc ships them, plus dark ones for the dark theme."""
    dark = HtmlFormatter(style="github-dark").get_style_defs(".pdoc .pdoc-code")
    # the backgrounds come from the theme
    dark = "\n".join(ln for ln in dark.splitlines() if not ln.startswith(".pdoc .pdoc-code  {"))
    return "@media (prefers-color-scheme: dark){\n" + dark + "\n}"


def _highlight(match: re.Match) -> str:
    code = html.unescape(match.group(1))
    lexer = BashLexer() if SHELL.match(code) else PythonLexer()
    body = highlight(code, lexer, HtmlFormatter(nowrap=True))
    return f'<div class="pdoc-code codehilite"><pre><span></span>{body}</pre></div>'


def process(page: Path) -> bool:
    s = page.read_text()
    if MARK in s:
        return False
    s = re.sub(r"<pre><code>(.*?)</code></pre>", _highlight, s, flags=re.S)
    head = f"{MARK}\n<style>{THEME.read_text()}\n{_syntax_css()}</style>\n"
    s = s.replace("</head>", head + "</head>", 1) if "</head>" in s else head + s
    script = f"<script>{COPY_JS}</script>\n"
    s = s.replace("</body>", script + "</body>", 1) if "</body>" in s else s + script
    page.write_text(s)
    return True


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    pages = [p for p in root.rglob("*.html") if "_template" not in p.parts]
    done = sum(process(p) for p in pages)
    print(f"styled {done} of {len(pages)} pages")
