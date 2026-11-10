#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared PPT-style slide-deck framework.

Layout, theme and navigation follow the reference deck
``bscat_fiskil_slides.html``. Report generators only supply slide content.
"""

from __future__ import annotations

import html

from deck_assets import CSS as BASE_CSS, NAV_JS


EXTRA_CSS = """
  /* ===== Deck extras: charts / tables / metric rows ===== */
  .status.purple{background:#F4F0FB;color:#6B4FA8;border-color:#E3D9F6}
  .cover-status .status.purple{background:rgba(139,105,212,.2);border-color:rgba(196,176,240,.35);color:#D9CDF8}
  .divider .status.purple{background:rgba(139,105,212,.2);border-color:rgba(196,176,240,.35);color:#D9CDF8}
  .chart-box{
    flex:1;min-height:0;display:flex;flex-direction:column;
    border:1px solid var(--line);border-radius:16px;background:#FBFCFE;
    padding:10px 12px 6px;overflow:hidden;
  }
  .chart-box > div{min-height:0}
  .chart-box .js-plotly-plot{width:100%!important}
  .chart-cap{font-size:10.5px;line-height:1.5;color:var(--faint);padding:6px 6px 0;flex:none}
  .chart-pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;flex:1;min-height:0}
  .chart-stack{display:flex;flex-direction:column;gap:12px;flex:1;min-height:0}
  .chart-stack .chart-box{flex:1 1 0}

  .duo{display:grid;grid-template-columns:1.52fr 1fr;gap:18px;flex:1;min-height:0}
  .duo.wide-right{grid-template-columns:1.25fr 1fr}
  .duo.narrow-right{grid-template-columns:1.75fr 1fr}
  .side-stack{display:flex;flex-direction:column;gap:12px;min-height:0}
  .side-card{
    flex:1;background:#FBFCFE;border:1px solid var(--line);border-radius:14px;
    padding:14px 17px;display:flex;flex-direction:column;justify-content:center;gap:4px;min-height:0;
  }
  .side-card .k{font-size:9.5px;font-weight:700;letter-spacing:.1em;color:var(--faint);text-transform:uppercase}
  .side-card .v{font-size:25px;line-height:1.16;font-weight:700;color:var(--navy);letter-spacing:-.015em;font-variant-numeric:tabular-nums}
  .side-card .v small{font-size:12.5px;color:var(--muted);font-weight:500;margin-left:3px;letter-spacing:0}
  .side-card p{font-size:11.5px;line-height:1.5;color:var(--muted)}
  .side-card.accent{background:linear-gradient(135deg,#0B2B4A 0%,#173F63 100%);border-color:#173F63}
  .side-card.accent .k{color:rgba(255,255,255,.48)}
  .side-card.accent .v{color:#fff}
  .side-card.accent p{color:rgba(255,255,255,.62)}

  .table-box{flex:1;min-height:0;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:#fff;display:flex;flex-direction:column}
  table.deck-table{width:100%;border-collapse:collapse;font-size:12.5px;line-height:1.45}
  .deck-table th{
    background:#F3F7FB;color:var(--navy);font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;
    text-align:left;padding:9px 13px;border-bottom:1px solid var(--line);white-space:nowrap;font-weight:700;
  }
  .deck-table td{padding:8px 13px;border-bottom:1px solid var(--line-soft);color:var(--text);vertical-align:top}
  .deck-table tr:last-child td{border-bottom:none}
  .deck-table tbody tr:nth-child(even) td{background:#FBFCFE}
  .deck-table .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
  .deck-table .strong{color:var(--navy);font-weight:700}
  .table-note{font-size:10.5px;line-height:1.5;color:var(--faint);padding:8px 14px 10px;border-top:1px solid var(--line-soft);flex:none}

  .metric-list{display:flex;flex-direction:column;border:1px solid var(--line);border-radius:14px;background:#fff;overflow:hidden;flex:1;min-height:0}
  .metric-row{display:grid;grid-template-columns:1fr 168px 100px;gap:12px;align-items:center;padding:10px 15px;border-bottom:1px solid var(--line-soft);font-size:12.5px}
  .metric-row:last-child{border-bottom:none}
  .metric-row .name{color:var(--text);font-weight:600;min-width:0}
  .metric-row .val{text-align:right;font-weight:700;color:var(--navy);font-variant-numeric:tabular-nums;white-space:nowrap}
  .metric-row .delta{text-align:right;font-size:10.5px;color:var(--muted);white-space:nowrap}
  .metric-row .val small{font-size:10px;color:var(--faint);font-weight:500;margin-left:3px}

  .cmp-kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;flex:none}
  .cmp-kpi{background:#FBFCFE;border:1px solid var(--line);border-radius:16px;padding:15px 18px 13px;min-width:0}
  .cmp-kpi .label{font-size:10px;font-weight:700;letter-spacing:.08em;color:var(--muted)}
  .cmp-kpi .pair{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-top:7px}
  .cmp-kpi .who{display:inline-flex;align-items:center;gap:5px;font-size:10px;color:var(--faint);min-width:0}
  .cmp-kpi .who i{width:7px;height:7px;border-radius:50%;flex:none}
  .cmp-kpi .val{font-size:21px;font-weight:700;color:var(--navy);font-variant-numeric:tabular-nums;white-space:nowrap}
  .cmp-kpi .delta{margin-top:7px;padding-top:6px;border-top:1px solid var(--line-soft);font-size:10.5px;color:var(--muted)}

  .risk-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;flex:1;min-height:0}
  .risk-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px 22px;display:flex;flex-direction:column;justify-content:center;gap:6px;min-height:0}
  .risk-card .k{font-size:10px;font-weight:700;letter-spacing:.08em;color:var(--faint);text-transform:uppercase}
  .risk-card b{font-size:32px;line-height:1.1;color:var(--navy);font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
  .risk-card b small{font-size:13px;color:var(--muted);font-weight:500;margin-left:3px;letter-spacing:0}
  .risk-card p{font-size:11.5px;line-height:1.55;color:var(--muted)}

  .findings-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;flex:1;min-height:0}
  .finding-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:17px 19px;display:flex;gap:13px;min-height:0;overflow:hidden}
  .finding-card .n{font-size:21px;line-height:1;font-weight:700;color:#D7E2EE;letter-spacing:-.02em;flex:none;padding-top:2px}
  .finding-card p{font-size:12.5px;line-height:1.7;color:var(--text);min-width:0}
  .finding-card b{color:var(--navy)}
  .finding-card mark{background:#FDF4E6;color:#8A5A13;padding:0 3px;border-radius:3px;font-weight:600}

  .bullet-list{display:flex;flex-direction:column;gap:10px;list-style:none;min-height:0}
  .bullet-list li{position:relative;padding-left:17px;font-size:13px;line-height:1.66;color:var(--text)}
  .bullet-list li::before{content:"";position:absolute;left:0;top:8px;width:6px;height:6px;border-radius:50%;background:var(--blue);flex:none}
  .bullet-list b{color:var(--navy)}
  .bullet-list mark{background:#FDF4E6;color:#8A5A13;padding:0 3px;border-radius:3px;font-weight:600}

  .seg-mini{display:flex;flex-direction:column;gap:6px;margin-top:4px}
  .seg-mini > div{display:flex;align-items:baseline;justify-content:space-between;gap:10px;padding-bottom:5px;border-bottom:1px dashed var(--line-soft)}
  .seg-mini > div:last-child{border-bottom:none;padding-bottom:0}
  .seg-mini b{font-size:17px;color:var(--navy);font-weight:700;font-variant-numeric:tabular-nums}
  .seg-mini span{font-size:10.5px;color:var(--muted)}

  .legend-inline{display:flex;align-items:center;gap:14px;font-size:10.5px;color:var(--muted);flex-wrap:wrap;flex:none}
  .legend-inline .key{display:inline-flex;align-items:center;gap:5px}
  .legend-inline i{width:9px;height:9px;border-radius:2px;display:inline-block;flex:none}

  .delta{display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
  .delta.muted{color:var(--faint)}
  .dot{width:7px;height:7px;border-radius:50%;display:inline-block;flex:none}

  .bar-stack{display:flex;flex-direction:column;gap:9px;flex:1;min-height:0;justify-content:center}
  .bar-row{display:grid;grid-template-columns:118px 1fr 128px;gap:12px;align-items:center}
  .bar-row .who{font-size:11px;color:var(--muted);display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
  .bar-row .who i{width:7px;height:7px;border-radius:50%;flex:none;display:inline-block}
  .bar-track{height:10px;background:#EEF2F6;border-radius:99px;overflow:hidden}
  .bar-fill{height:100%;border-radius:99px;width:0}
  .bar-row .vals{text-align:right;font-size:11.5px;font-weight:700;color:var(--navy);font-variant-numeric:tabular-nums;white-space:nowrap}

  .note-strip{
    flex:none;display:flex;align-items:flex-start;gap:12px;padding:13px 18px;
    background:#FFFBF3;border:1px solid #F1E2C9;border-left:4px solid #C98A2E;border-radius:12px;
  }
  .note-strip .mark{width:20px;height:20px;border-radius:50%;background:#F2E0C4;color:#8A5A13;font-size:12px;font-weight:800;display:grid;place-items:center;flex:none;margin-top:1px}
  .note-strip b{display:block;font-size:12px;color:#8A5A13;letter-spacing:.02em}
  .note-strip p{font-size:12px;line-height:1.6;color:#7A5A2A;margin-top:2px}
"""

CSS = BASE_CSS + EXTRA_CSS

FIT_JS = """
(function () {
  function fitPlots() {
    if (!window.Plotly) return;
    document.querySelectorAll('.slide.is-active .chart-box .js-plotly-plot').forEach(function (p) {
      var box = p.closest('.chart-box');
      if (!box) return;
      var cs = getComputedStyle(box);
      var cap = box.querySelector('.chart-cap');
      var w = box.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
      var h = box.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom)
              - (cap ? cap.offsetHeight : 0);
      if (w < 120 || h < 120) return;
      if (Math.abs(w - p.clientWidth) > 4 || Math.abs(h - p.clientHeight) > 4) {
        var holder = p.parentElement;
        if (holder && holder !== box && holder.parentElement === box) {
          holder.style.height = h + 'px';
          holder.style.width = '100%';
        } else {
          p.style.width = w + 'px';
          p.style.height = h + 'px';
        }
        Plotly.Plots.resize(p);
      }
    });
  }
  function schedule() { setTimeout(fitPlots, 60); setTimeout(fitPlots, 420); }
  window.addEventListener('load', function () { setTimeout(fitPlots, 200); });
  window.addEventListener('resize', function () { setTimeout(fitPlots, 120); });
  document.addEventListener('deck:slide', schedule);
})();
"""


def esc(value) -> str:
    return html.escape(str(value))


def status(level: str, label: str) -> str:
    return f'<span class="status {level}"><i></i>{esc(label)}</span>'


def slide_cover(
    *,
    brand: str,
    tag: str,
    eyebrow: str,
    title: str,
    desc: str,
    statuses: list[tuple[str, str]] | None = None,
    metrics: list[tuple[str, str, str]] | None = None,
    hint: str = "按 <b>→</b> 或 <b>←</b> 翻页　·　右下角可点击切换",
) -> str:
    status_html = "".join(
        f'<span class="status {level}"><i></i>{esc(label)}</span>'
        for level, label in (statuses or [])
    )
    metric_html = "".join(
        f'<div class="metric"><div class="k">{esc(k)}</div><b>{v}</b><span>{esc(s)}</span></div>'
        for k, v, s in (metrics or [])
    )
    return f"""
    <section class="slide cover-slide is-active" aria-label="封面">
      <div class="cover-top">
        <span class="cover-brand">{brand}</span>
        <span class="cover-tag">{esc(tag)}</span>
      </div>
      <div class="cover-main">
        <div class="cover-eyebrow reveal">{esc(eyebrow)}</div>
        <h1 class="reveal d1">{title}</h1>
        <p class="reveal d2">{desc}</p>
        <div class="cover-status reveal d3">{status_html}</div>
      </div>
      <div class="cover-bottom reveal d4">
        <div class="cover-metrics">{metric_html}</div>
        <div class="cover-hint">{hint}</div>
      </div>
    </section>"""


def slide_divider(
    *,
    brand: str,
    section: str,
    num: str,
    kicker: str,
    title: str,
    desc: str,
    status_html: str = "",
    meta: list[tuple[str, str]] | None = None,
) -> str:
    meta_html = "".join(
        f"<div><b>{v}</b><span>{esc(s)}</span></div>" for v, s in (meta or [])
    )
    return f"""
    <section class="slide divider" data-num="{esc(num)}" aria-label="{esc(title)}">
      <div>
        <div class="s-top" style="justify-content:space-between">
          <span class="s-brand" style="color:rgba(255,255,255,.7)">{brand}</span>
          <span class="s-section" style="color:rgba(255,255,255,.35)">{esc(section)}</span>
        </div>
        <div class="s-kicker reveal" style="margin-top:40px">{esc(kicker)}</div>
        <h2 class="reveal d1">{esc(title)}</h2>
        <p class="reveal d2">{desc}</p>
        <div class="reveal d3" style="margin-top:28px">{status_html}</div>
      </div>
      <div class="divider-meta reveal d4">{meta_html}</div>
    </section>"""


def slide_content(
    *,
    brand: str,
    section: str,
    kicker: str,
    title: str,
    sub: str = "",
    body: str = "",
    center: bool = False,
) -> str:
    sub_html = f'<p class="s-sub">{sub}</p>' if sub else ""
    body_cls = "s-body center" if center else "s-body"
    return f"""
    <section class="slide" aria-label="{esc(title)}">
      <div class="s-top">
        <span class="s-brand">{brand}</span>
        <span class="s-section">{esc(section)}</span>
      </div>
      <header class="s-head reveal">
        <div class="s-kicker">{esc(kicker)}</div>
        <h2>{esc(title)}</h2>
        {sub_html}
      </header>
      <div class="{body_cls}">
        {body}
      </div>
    </section>"""


def slide_closing(
    *,
    brand: str,
    kicker: str,
    title: str,
    sub: str,
    takeaways: list[tuple[str, str]],
    source: str,
    date: str,
) -> str:
    rows = "".join(
        f'<div class="takeaway"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>'
        for k, v in takeaways
    )
    return f"""
    <section class="slide closing" aria-label="结论">
      <div>
        <div class="s-top" style="justify-content:space-between">
          <span class="s-brand" style="color:rgba(255,255,255,.7)">{brand}</span>
        </div>
        <div class="s-kicker reveal" style="margin-top:26px">{esc(kicker)}</div>
        <h2 class="reveal d1">{esc(title)}</h2>
        <p class="sub reveal d2">{sub}</p>
        <div class="takeaways reveal d3">{rows}</div>
      </div>
      <div class="closing-foot reveal d4">
        <div class="src">{source}</div>
        <div class="date">{esc(date)}</div>
      </div>
    </section>"""


def kpi_cards(cards: list[tuple[str, str, str]]) -> str:
    inner = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'
        for label, value, note in cards
    )
    return f'<div class="kpi-row reveal d1">{inner}</div>'


def cmp_kpi_cards(cards: list[dict]) -> str:
    """cards: label, left_name, left_value, right_name, right_value, delta, colors."""
    inner = ""
    for c in cards:
        inner += (
            '<div class="cmp-kpi">'
            f'<div class="label">{esc(c["label"])}</div>'
            '<div class="pair"><span class="who">'
            f'<i style="background:{c.get("left_color", "#1D4E79")}"></i>{esc(c["left_name"])}</span>'
            f'<span class="val">{c["left_value"]}</span></div>'
            '<div class="pair"><span class="who">'
            f'<i style="background:{c.get("right_color", "#A9680F")}"></i>{esc(c["right_name"])}</span>'
            f'<span class="val">{c["right_value"]}</span></div>'
            f'<div class="delta">{c.get("delta", "")}</div>'
            "</div>"
        )
    return f'<div class="cmp-kpi-row reveal d1">{inner}</div>'


def chart_box(fig_html: str, caption: str = "", extra_cls: str = "") -> str:
    cap = f'<div class="chart-cap">{caption}</div>' if caption else ""
    return f'<div class="chart-box {extra_cls}">{fig_html}{cap}</div>'


def table_box(headers: list[str], rows: list[list[str]], note: str = "", num_cols: tuple[int, ...] = ()) -> str:
    head = "".join(
        f'<th class="{"num" if i in num_cols else ""}">{h}</th>'
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if i in num_cols else ""}">{cell}</td>'
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    note_html = f'<div class="table-note">{note}</div>' if note else ""
    return (
        f'<div class="table-box"><table class="deck-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>{note_html}</div>"
    )


def metric_rows(rows: list[tuple[str, str, str]]) -> str:
    inner = "".join(
        f'<div class="metric-row"><div class="name">{label}</div>'
        f'<div class="val">{value}</div><div class="delta">{delta}</div></div>'
        for label, value, delta in rows
    )
    return f'<div class="metric-list">{inner}</div>'


def side_cards(cards: list[tuple[str, str, str]], accent_first: bool = False) -> str:
    inner = ""
    for i, (label, value, note) in enumerate(cards):
        inner += (
            f'<div class="side-card{" accent" if accent_first and i == 0 else ""}">'
            f'<div class="k">{esc(label)}</div><div class="v">{value}</div>'
            f'<p>{note}</p></div>'
        )
    return f'<div class="side-stack reveal d2">{inner}</div>'


def fact_strip(items: list[tuple[str, str]]) -> str:
    inner = "".join(
        f'<div class="fact"><b>{v}</b><span>{esc(s)}</span></div>' for v, s in items
    )
    return f'<div class="fact-strip reveal d2">{inner}</div>'


def note_strip(title: str, text: str) -> str:
    return (
        '<div class="note-strip reveal d3"><div class="mark">!</div><div>'
        f"<b>{title}</b><p>{text}</p></div></div>"
    )


def bullet_list(items: list[str]) -> str:
    lis = "".join(f"<li>{item}</li>" for item in items)
    return f'<ul class="bullet-list reveal d2">{lis}</ul>'


def finding_cards(items: list[str], start: int = 0) -> str:
    cards = ""
    for i, text in enumerate(items):
        cards += (
            f'<div class="finding-card reveal d{(i % 4) + 1}">'
            f'<div class="n">{start + i + 1:02d}</div><p>{text}</p></div>'
        )
    return f'<div class="findings-grid">{cards}</div>'


def deck_document(*, title: str, slides: list[str], plotly_js: str = "") -> str:
    js_block = f"<script>{plotly_js}</script>" if plotly_js else ""
    body = "".join(slides)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light">
<title>{esc(title)}</title>
{js_block}
<style>{CSS}</style>
</head>
<body>

<div class="stage-wrap">
  <main class="stage" id="stage" aria-label="{esc(title)}">
{body}
  </main>
</div>

<div class="deck-ui" aria-label="幻灯片翻页控制">
  <button class="deck-btn" id="prevBtn" type="button" aria-label="上一页" disabled>←</button>
  <div class="counter" aria-live="polite"><span class="cur" id="curNum">01</span><span class="sep">/</span><span class="total" id="totalNum">00</span></div>
  <button class="deck-btn" id="nextBtn" type="button" aria-label="下一页">→</button>
</div>
<div class="progress" aria-hidden="true"><i id="progressFill"></i></div>

<script>{NAV_JS}</script>
<script>{FIT_JS}</script>

</body>
</html>"""
