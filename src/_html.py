"""Shared HTML bits for the benchmark pages (style, escaping, tables, tiles)."""

from __future__ import annotations

import html
import statistics

CSS = """
:root { --surface-0:#ffffff; --surface-1:#f7f7f5; --surface-2:#ecece7; --border:#d8d8d2;
  --text-1:#14140f; --text-2:#3d3d38; --text-3:#6b6b64; --accent:#1b5fb0; --callout:#eef4fd;
  --good:#1b5fb0; --bad:#c0392b; --code-bg:#f4f4f0; --radius:8px; }
* { box-sizing:border-box; }
body { font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--text-1); background:var(--surface-0); max-width:1100px; margin:0 auto; padding:2rem 1.25rem 4rem; }
h1 { font-size:1.7rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1.2rem; margin:2.2rem 0 .8rem; padding-bottom:.35rem; border-bottom:1px solid var(--border); }
h3 { font-size:1rem; margin:1.4rem 0 .4rem; }
p, li { color:var(--text-2); max-width:80ch; }
a { color:var(--accent); }
code { font-family:ui-monospace,Menlo,monospace; font-size:.85em; background:var(--code-bg);
  border:1px solid var(--border); border-radius:4px; padding:.05em .35em; }
pre { background:var(--code-bg); border:1px solid var(--border); border-radius:var(--radius); padding:1rem 1.1rem;
  white-space:pre-wrap; overflow-wrap:anywhere; font-family:ui-monospace,Menlo,monospace; font-size:.8rem;
  line-height:1.5; color:var(--text-2); }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.82rem; background:var(--surface-1);
  border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; margin:.6rem 0; }
th,td { text-align:right; padding:.4rem .6rem; border-bottom:1px solid var(--border); white-space:nowrap;
  font-variant-numeric:tabular-nums; }
th:first-child,td:first-child { text-align:left; }
th { color:var(--text-2); font-weight:600; background:var(--surface-0); position:sticky; top:0; }
tr:last-child td { border-bottom:none; }
tr.avg td { background:var(--callout); font-weight:600; border-top:2px solid var(--border); }
td.txt { white-space:normal; text-align:left; }
tr.span td { background:var(--surface-0); text-align:left; white-space:nowrap; }
.pos { color:var(--good); } .neg { color:var(--bad); }
.tiles { display:flex; flex-wrap:wrap; gap:.9rem; margin:1rem 0; }
.tile { background:var(--surface-1); border:1px solid var(--border); border-radius:var(--radius);
  padding:.85rem 1rem; min-width:150px; }
.tile .num { font-size:1.5rem; font-weight:650; letter-spacing:-.02em; line-height:1.1; }
.tile .lbl { color:var(--text-2); font-size:.82rem; margin-top:.2rem; max-width:22ch; }
.callout { background:var(--callout); border:1px solid var(--border); border-radius:var(--radius);
  padding:1rem 1.15rem; margin:1rem 0; }
details { margin:.8rem 0; border:1px solid var(--border); border-radius:var(--radius); padding:.4rem .9rem; background:var(--surface-1); }
summary { cursor:pointer; font-weight:600; color:var(--text-2); }
header.top { display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid var(--border);
  padding-bottom:.5rem; margin-bottom:1.2rem; font-size:.9rem; color:var(--text-2); }
.note { color:var(--text-3); font-size:.9rem; }
caption { caption-side:top; text-align:center; color:var(--text-3); font-size:.8rem; padding:0 0 4px; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--border); color:var(--text-2); font-size:.9rem; }
"""


def esc(x) -> str:
    return html.escape(str(x))


def f(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def delta(x, ref, nd=3):
    """value (diff): blue when the diff is positive, red when negative."""
    if x is None or ref is None:
        return ""
    d = x - ref
    cls = "pos" if d > 0 else ("neg" if d < 0 else "")
    sign = "+" if d >= 0 else ""
    return f"{x:.{nd}f} <span class='{cls}'>({sign}{d:.{nd}f})</span>"


def ratio(before, after):
    """speed-up factor, blue when faster, red when slower."""
    r = before / max(after, 1e-9)
    cls = "pos" if r > 1.05 else ("neg" if r < 0.95 else "")
    return f"<span class='{cls}'>{r:.1f}x</span>"


def mean_of(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def average_row(rows, col_vals, label):
    """Average row for a metric table: mean value and, in parentheses, the mean difference to the
    reference column over the tasks where that cell exists."""
    width = max((len(r) for r in rows), default=0)
    cells = [""] * (width - 1)
    for j, pairs in col_vals.items():
        xs, refs = [v for v, _ in pairs], [r for _, r in pairs]
        mean_x = statistics.mean(xs)
        cells[j - 1] = f(mean_x, 3) if all(v == r for v, r in pairs) else delta(mean_x, statistics.mean(refs))
    return avg_row(label, cells)


def avg_row(label, cols):
    """A highlighted average row; ``cols`` are already formatted cells (empty for non-numeric)."""
    return ([label] + list(cols), "avg")


def table(header, rows, text_cols=(), caption=None):
    """HTML table; ``caption`` is a small line centred above it (e.g. the baseline that differences are taken against)."""
    h = "".join(f"<th>{c}</th>" for c in header)
    body = []
    for r in rows:
        cls_row = ""
        if isinstance(r, tuple):
            r, cls_row = r[0], f" class='{r[1]}'"
        cells = []
        if cls_row == " class='span'":  # one cell across every column (a group divider)
            body.append(f"<tr class='span'><td colspan='{len(header)}'>{r[0]}</td></tr>")
            continue
        for j, c in enumerate(r):
            cls = " class='txt'" if j in text_cols else ""
            cells.append(f"<td{cls}>{c}</td>")
        body.append(f"<tr{cls_row}>" + "".join(cells) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f"<div class='wrap'><table>{cap}<thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def tile(num, label):
    return f"<div class='tile'><div class='num'>{num}</div><div class='lbl'>{label}</div></div>"




def heat_span(v, lo, hi, nd=1, sign=True, gamma=1.0):
    """A heatmap cell scaled to the table's own range [lo, hi], so small differences stay visible. All values of one
    sign: a sequential scale (light to dark blue for gains, dark to light red for losses) from the table's minimum to
    its maximum. Mixed signs: a diverging scale, blue above zero and red below, each side scaled to its extreme.
    ``gamma`` > 1 spends more of the colour range on the largest values (position ** gamma), so small differences between
    the best entries stay visible."""
    if v is None:
        return ""
    if lo >= 0:
        t, rgb = ((v - lo) / (hi - lo) if hi > lo else 1.0), "27,95,176"
    elif hi <= 0:
        t, rgb = ((hi - v) / (hi - lo) if hi > lo else 1.0), "192,57,43"
    elif v >= 0:
        t, rgb = v / hi, "27,95,176"
    else:
        t, rgb = v / lo, "192,57,43"
    a = 0.08 + 0.82 * max(0.0, min(t, 1.0)) ** gamma
    fg = "color:#fff;" if a > 0.55 else ""
    txt = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return (f"<span style='display:block;background:rgba({rgb},{a:.2f});{fg}padding:.1rem .35rem;border-radius:3px;"
            f"font-variant-numeric:tabular-nums'>{txt}</span>")


def span_of(values):
    """(lo, hi) of the values that are not None, for heat_span."""
    vals = [v for v in values if v is not None]
    return (min(vals), max(vals)) if vals else (0.0, 0.0)
