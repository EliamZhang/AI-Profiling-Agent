#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用户画像 HTML 报告生成器（统一运行入口）。

运行本脚本会先调用 profile_analysis.py 完成画像分析，再读取其输出的中间结果，
生成自包含的 HTML 报告。报告层不重复任何统计计算，所有数字均来自中间结果。

示例：
    python generate_report.py
    python generate_report.py --input data.csv --output-dir analysis/out
    python generate_report.py --skip-analysis --output-dir analysis/out
"""

from __future__ import annotations

import argparse
import html
import math
import time
import webbrowser
from pathlib import Path

import numpy as np

from profile_analysis import (
    DEFAULT_CONFIG,
    DIMENSION_META,
    DISHONOUR_STATUSES,
    INCOME_ARCHETYPES,
    LENDER_BEHAVIORS,
    OBSERVATION_BANDS,
    PAY_RHYTHMS,
    SCORED_DIMS,
    SEGMENT_ORDER,
    load_config,
    run_analysis,
    load_intermediates,
)

VERSION = "1.1.0"

SEGMENT_COLORS = {
    "优质均衡": "#0f9d76",
    "稳健工薪": "#2f6fed",
    "均衡普通": "#0891b2",
    "月光消费": "#d97706",
    "现金流紧张": "#ea580c",
    "高风险承压": "#dc2626",
    "数据不足": "#94a3b8",
}
DIM_COLORS = {
    "income": "#2f6fed",
    "cashflow": "#0f8b8d",
    "liquidity": "#7c5cff",
    "debt": "#d97706",
    "engagement": "#0891b2",
    "consumption": "#64748b",
    "confidence": "#94a3b8",
}
DIM_LABELS = {k: v["label"] for k, v in DIMENSION_META.items()}
DIM_LABELS["health"] = "综合健康分"

# 报告展示的特征（仅影响展示，不改变任何统计口径）
FEATURE_PANELS = [
    ("inc_wage_share", "工资收入占比"),
    ("inc_salary_cv", "工资金额变异系数"),
    ("inc_primary_concentration", "主收入来源集中度"),
    ("cf_saving_rate", "储蓄率（余额变动口径）"),
    ("cf_net_after_expense", "收支结余率（流水口径）"),
    ("cf_expense_cv", "支出波动率"),
    ("lq_balance_median", "余额中位数"),
    ("lq_reserve_months", "固定支出备用月数"),
    ("db_debt_service_ratio", "负债还款 / 收入"),
    ("db_disburse_count", "历史放款次数"),
    ("en_active_days", "活跃交易天数"),
    ("en_txn_freq", "活跃日交易频次"),
]
SEGMENT_METRICS = [
    ("inc_global_sum", "总收入"),
    ("inc_wage_share", "工资收入占比"),
    ("cf_saving_rate", "储蓄率"),
    ("cf_net_after_expense", "收支结余率"),
    ("lq_balance_median", "余额中位数"),
    ("lq_reserve_months", "备用月数"),
    ("db_debt_share", "债务支出占比"),
    ("db_disburse_count", "历史放款次数"),
    ("db_unresolved_cnt", "未结清拒付次数"),
    ("en_active_days", "活跃交易天数"),
]
THRESHOLD_LABELS = {
    "premium_health": "优质均衡：健康分下限",
    "premium_max_risk_flags": "优质均衡：最多风险标签数",
    "stable_income_score": "稳健工薪：收入得分下限",
    "stable_cashflow_score": "稳健工薪：收支得分下限",
    "stable_debt_score": "稳健工薪：负债得分下限",
    "tight_cashflow_score": "现金流紧张：收支得分下限",
    "tight_liquidity_score": "现金流紧张：流动性得分下限",
    "highrisk_debt_score": "高风险：负债得分上限",
    "highrisk_min_flags": "高风险：最少风险标签数",
    "moon_saving_rate": "月光消费：储蓄率上限",
    "high_debt_score": "负债压力高：得分上限",
    "multi_lender_cnt": "多头借贷：机构数下限",
    "gambling_share": "赌博支出占比下限",
    "low_buffer_negative_share": "低缓冲：负余额天数占比下限",
    "fees_share": "费用/透支占比下限",
    "income_unstable_cv": "收入不稳定：工资 CV 下限",
    "welfare_income_share": "福利依赖：福利收入占比下限",
}


# ---------------------------------------------------------------------------
# 格式化与转义
# ---------------------------------------------------------------------------

def esc(value) -> str:
    return html.escape(str(value), quote=True)


def _float(value, default=math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _num(value, digits=1) -> str:
    v = _float(value)
    if not math.isfinite(v):
        return "—"
    return f"{v:,.{digits}f}"


def _pct(value, digits=1) -> str:
    v = _float(value)
    if not math.isfinite(v):
        return "—"
    return f"{v * 100:.{digits}f}%"


def _pct_points(value, digits=1) -> str:
    v = _float(value)
    if not math.isfinite(v):
        return "—"
    return f"{v:.{digits}f}%"


def _fmt(value, fmt="ratio", digits=None) -> str:
    v = _float(value)
    if not math.isfinite(v):
        return "—"
    if fmt == "pct":
        return _pct(v, 1 if digits is None else digits)
    if fmt == "amount":
        return f"A${v:,.0f}" if digits is None else f"A${v:,.{digits}f}"
    if fmt == "count":
        return f"{v:,.0f}"
    if fmt == "days":
        return f"{v:.1f} 天"
    if fmt == "months":
        return f"{v:.1f} 个月"
    if fmt == "score":
        return f"{v:.1f}"
    return f"{v:,.2f}" if digits is None else f"{v:,.{digits}f}"


def _compact(value, fmt="ratio") -> str:
    v = _float(value)
    if not math.isfinite(v):
        return "—"
    if fmt == "pct":
        a = abs(v)
        return f"{v * 100:.1f}%" if a < 0.1 else f"{v * 100:.0f}%"
    if fmt in ("amount", "count"):
        a = abs(v)
        prefix = "A$" if fmt == "amount" else ""
        if a >= 1e9:
            return f"{prefix}{v / 1e9:.1f}B"
        if a >= 1e6:
            return f"{prefix}{v / 1e6:.1f}M"
        if a >= 1e3:
            return f"{prefix}{v / 1e3:.1f}k"
        return f"{prefix}{v:,.0f}"
    if fmt == "days":
        return f"{v:.0f}天"
    if fmt == "months":
        return f"{v:.1f}月"
    return f"{v:,.2f}"


def _ellipsis(text, limit=18) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _safe(value) -> float:
    return _float(value)


def _signed(value, digits=2) -> str:
    v = _float(value)
    if not math.isfinite(v):
        return "—"
    return f"{v:+.{digits}f}"


# ---------------------------------------------------------------------------
# SVG 基础组件
# ---------------------------------------------------------------------------

def _svg(width, height, body, label="图表") -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}" '
        f'preserveAspectRatio="xMidYMid meet">{body}</svg>'
    )


def _grid_lines(x0, x1, y0, y1, ticks, fmt_fn) -> str:
    out = []
    for t in ticks:
        y = y1 - (y1 - y0) * t
        out.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
            f'stroke="#e6ebf3" stroke-width="1"/>'
            f'<text x="{x0 - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#94a3b8">{esc(fmt_fn(t))}</text>'
        )
    return "".join(out)


def hbar_chart(items, value_fmt="pct", color="#2f6fed", width=720, row_h=34,
               max_value=None, label_width=230) -> str:
    """items: [(label, value, note)]；value 为 0-1 比率或绝对数。"""
    if not items:
        return '<p class="empty">暂无可展示数据</p>'
    values = [max(0.0, _float(v)) for _, v, _ in items]
    vmax = max(values) if values else 0.0
    if max_value is not None:
        vmax = max(vmax, float(max_value))
    vmax = vmax if vmax > 0 else 1.0
    height = len(items) * row_h + 16
    bar_x = label_width + 12
    bar_w = width - bar_x - 78
    parts = []
    for idx, (label, value, note) in enumerate(items):
        v = max(0.0, _float(value))
        y = idx * row_h + 8
        w = bar_w * (v / vmax) if math.isfinite(v) else 0.0
        txt = _pct_points(v * 100) if value_fmt == "pct" else _fmt(v, value_fmt)
        parts.append(
            f'<text x="{label_width}" y="{y + 15:.1f}" text-anchor="end" font-size="12.5" '
            f'fill="#334155">{esc(_ellipsis(label, 16))}</text>'
            f'<rect x="{bar_x}" y="{y + 2:.1f}" width="{bar_w}" height="18" rx="5" fill="#f1f5f9"/>'
            f'<rect x="{bar_x}" y="{y + 2:.1f}" width="{max(0.8, w):.1f}" height="18" rx="5" fill="{color}"/>'
            f'<text x="{bar_x + bar_w + 8}" y="{y + 16:.1f}" font-size="12" font-weight="600" '
            f'fill="#0f172a">{esc(txt)}</text>'
        )
        if note:
            parts.append(
                f'<text x="{bar_x + bar_w + 8}" y="{y + 29:.1f}" font-size="10.5" '
                f'fill="#94a3b8">{esc(note)}</text>'
            )
    return _svg(width, height, "".join(parts), "横向柱状图")


def donut_chart(items, width=300, height=300, center_label="申请记录") -> str:
    """items: [(label, value, color)]"""
    total = sum(max(0.0, _float(v)) for _, v, _ in items)
    cx, cy, r, sw = width / 2, height / 2, 92, 30
    circ = 2 * math.pi * r
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#eef2f7" stroke-width="{sw}"/>']
    offset = 0.0
    for label, value, color in items:
        v = max(0.0, _float(value))
        if total <= 0 or v <= 0:
            continue
        length = circ * v / total
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-dasharray="{length:.3f} {circ - length:.3f}" '
            f'stroke-dashoffset="{-offset:.3f}" transform="rotate(-90 {cx} {cy})">'
            f'<title>{esc(label)}: {int(v):,}（{v / total * 100:.1f}%）</title></circle>'
        )
        offset += length
    parts.append(
        f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="30" font-weight="700" '
        f'fill="#0f172a">{int(round(total)):,}</text>'
        f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" font-size="12" '
        f'fill="#64748b">{esc(center_label)}</text>'
    )
    return _svg(width, height, "".join(parts), "分群占比环形图")


def histogram_chart(hist_rows, stat_row, width=390, height=205) -> str:
    if hist_rows is None or len(hist_rows) == 0:
        return '<p class="empty">样本不足，无法绘制分布</p>'
    hist_rows = hist_rows.sort_values("bin_index")
    counts = [max(0.0, _float(c)) for c in hist_rows["count"]]
    cmax = max(counts) if counts else 0.0
    cmax = cmax if cmax > 0 else 1.0
    left, right, top, bottom = 44, 12, 14, 36
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(counts)
    bw = plot_w / n
    parts = []
    ticks = [0.0, 0.5, 1.0]
    parts.append(_grid_lines(left, left + plot_w, top, top + plot_h, ticks,
                             lambda t: f"{int(round(t * cmax)):,}"))
    color = DIM_COLORS.get(stat_row.get("dimension"), "#2f6fed")
    fmt = stat_row.get("fmt", "ratio")
    for i, c in enumerate(counts):
        h = plot_h * (c / cmax)
        x = left + i * bw
        parts.append(
            f'<rect x="{x + 0.8:.1f}" y="{top + plot_h - h:.1f}" width="{max(0.8, bw - 1.6):.1f}" '
            f'height="{h:.1f}" rx="2" fill="{color}" opacity="0.82">'
            f'<title>{esc(_compact(hist_rows.iloc[i]["bin_left"], fmt))} – '
            f'{esc(_compact(hist_rows.iloc[i]["bin_right"], fmt))}: {int(c):,}</title></rect>'
        )
    median = _safe(stat_row.get("p50"))
    if math.isfinite(median):
        idx = int(np.searchsorted(hist_rows["bin_right"].to_numpy(dtype=float), median))
        idx = min(max(idx, 0), n - 1)
        mx = left + (idx + 0.5) * bw
        parts.append(
            f'<line x1="{mx:.1f}" y1="{top}" x2="{mx:.1f}" y2="{top + plot_h}" '
            f'stroke="#0f172a" stroke-width="1.4" stroke-dasharray="4 3"/>'
            f'<text x="{mx:.1f}" y="{top - 3}" text-anchor="middle" font-size="10" '
            f'fill="#0f172a">中位 {esc(_compact(median, fmt))}</text>'
        )
    lo = _float(hist_rows.iloc[0]["bin_left"])
    hi = _float(hist_rows.iloc[-1]["bin_right"])
    parts.append(
        f'<text x="{left}" y="{height - 14}" font-size="10.5" fill="#94a3b8">'
        f'{esc(_compact(lo, fmt))}</text>'
        f'<text x="{left + plot_w}" y="{height - 14}" text-anchor="end" font-size="10.5" '
        f'fill="#94a3b8">{esc(_compact(hi, fmt))}</text>'
        f'<text x="{left + plot_w / 2}" y="{height - 14}" text-anchor="middle" font-size="10.5" '
        f'fill="#cbd5e1">P1–P99 区间（等宽分箱）</text>'
    )
    return _svg(width, height, "".join(parts), f"{stat_row.get('label', '')}分布")


def rate_ci_chart(rows, width=780, height=300) -> str:
    rows = [r for r in rows if _safe(r.get("matured", r.get("target_matured", 0))) > 0]
    if not rows:
        return '<p class="empty">没有已到期样本，无法计算分组逾期率</p>'
    left, right, top, bottom = 52, 16, 18, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    rates = [_safe(r.get("rate")) * 100 for r in rows]
    highs = [_safe(r.get("ci_high")) * 100 for r in rows]
    ymax = max([h for h in highs if math.isfinite(h)] + rates + [1.0])
    ymax = max(ymax * 1.25, 2.0)
    parts = [_grid_lines(left, left + plot_w, top, top + plot_h, [0, 0.25, 0.5, 0.75, 1.0],
                         lambda t: f"{t * ymax:.1f}%")]
    step = plot_w / len(rows)
    bar_w = min(46.0, step * 0.5)
    for i, r in enumerate(rows):
        rate = max(0.0, _safe(r.get("rate")) * 100)
        low = max(0.0, _safe(r.get("ci_low")) * 100)
        high = max(rate, _safe(r.get("ci_high")) * 100)
        cx = left + step * (i + 0.5)
        h = plot_h * rate / ymax
        y = top + plot_h - h
        label = r.get("segment", r.get("band_label", ""))
        color = SEGMENT_COLORS.get(label, "#dc2626")
        parts.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(0.6, h):.1f}" '
            f'rx="4" fill="{color}" opacity="0.85"><title>{esc(label)}: {rate:.1f}%</title></rect>'
        )
        y_low = top + plot_h - plot_h * low / ymax
        y_high = top + plot_h - plot_h * high / ymax
        parts.append(
            f'<line x1="{cx:.1f}" y1="{y_low:.1f}" x2="{cx:.1f}" y2="{y_high:.1f}" '
            f'stroke="#475569" stroke-width="1.4"/>'
            f'<line x1="{cx - 5:.1f}" y1="{y_high:.1f}" x2="{cx + 5:.1f}" y2="{y_high:.1f}" '
            f'stroke="#475569" stroke-width="1.4"/>'
            f'<line x1="{cx - 5:.1f}" y1="{y_low:.1f}" x2="{cx + 5:.1f}" y2="{y_low:.1f}" '
            f'stroke="#475569" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{min(top + plot_h + 18, height - 30)}" text-anchor="middle" '
            f'font-size="11.5" fill="#334155">{esc(_ellipsis(label, 8))}</text>'
            f'<text x="{cx:.1f}" y="{height - 12}" text-anchor="middle" font-size="10" '
            f'fill="#94a3b8">n={int(_safe(r.get("matured", r.get("target_matured", 0))))}</text>'
        )
        if rate > 0:
            parts.append(
                f'<text x="{cx:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="10.5" '
                f'font-weight="600" fill="#0f172a">{rate:.1f}%</text>'
            )
    return _svg(width, height, "".join(parts), "分组逾期率及置信区间")


def diverging_chart(items, width=820, row_h=30, label_width=250) -> str:
    items = [it for it in items if math.isfinite(_safe(it[1]))]
    if not items:
        return '<p class="empty">没有可用的组间差异指标</p>'
    vmax = max(abs(_safe(v)) for _, v, _ in items)
    vmax = vmax if vmax > 0 else 1.0
    height = len(items) * row_h + 34
    center = label_width + (width - label_width - 70) / 2
    half = (width - label_width - 90) / 2
    parts = [
        f'<line x1="{center:.1f}" y1="18" x2="{center:.1f}" y2="{height - 14}" '
        f'stroke="#cbd5e1" stroke-width="1"/>',
        f'<text x="{center:.1f}" y="13" text-anchor="middle" font-size="10.5" fill="#94a3b8">0</text>',
        f'<text x="{center - half:.1f}" y="13" text-anchor="middle" font-size="10.5" fill="#64748b">负样本更高</text>',
        f'<text x="{center + half:.1f}" y="13" text-anchor="middle" font-size="10.5" fill="#64748b">正样本更高</text>',
    ]
    for i, (label, value, note) in enumerate(items):
        v = _safe(value)
        y = 24 + i * row_h
        w = half * min(abs(v), vmax) / vmax
        if v >= 0:
            x = center
            color = "#dc2626"
        else:
            x = center - w
            color = "#2f6fed"
        parts.append(
            f'<text x="{label_width}" y="{y + 14:.1f}" text-anchor="end" font-size="12" '
            f'fill="#334155">{esc(_ellipsis(label, 18))}</text>'
            f'<rect x="{x:.1f}" y="{y + 2:.1f}" width="{max(0.8, w):.1f}" height="17" rx="4" '
            f'fill="{color}" opacity="0.82"><title>{esc(label)}: SMD {v:.2f}（{esc(note)}）</title></rect>'
            f'<text x="{center + (w + 7 if v >= 0 else -w - 7):.1f}" y="{y + 15:.1f}" '
            f'text-anchor="{"start" if v >= 0 else "end"}" font-size="10.5" fill="#475569">'
            f'{_signed(v)}</text>'
        )
    return _svg(width, height, "".join(parts), "标准化均值差")


def stacked_score_chart(rows, width=760, row_h=42, label_width=140, score_low=40, score_high=70) -> str:
    if not rows:
        return '<p class="empty">暂无维度评分数据</p>'
    height = len(rows) * row_h + 44
    bar_x = label_width + 14
    bar_w = width - bar_x - 120
    parts = []
    legend = [(f"低分 (<{score_low:g})", "#f87171"), (f"中分 ({score_low:g}–{score_high:g})", "#fbbf24"), (f"高分 (≥{score_high:g})", "#34d399")]
    lx = bar_x
    for text, color in legend:
        parts.append(
            f'<rect x="{lx}" y="8" width="11" height="11" rx="2" fill="{color}"/>'
            f'<text x="{lx + 16}" y="18" font-size="11" fill="#475569">{esc(text)}</text>'
        )
        lx += 120
    for i, r in enumerate(rows):
        y = 34 + i * row_h
        low = max(0.0, _safe(r.get("low_pct")))
        mid = max(0.0, _safe(r.get("mid_pct")))
        high = max(0.0, _safe(r.get("high_pct")))
        total = low + mid + high
        if total <= 0:
            total = 1.0
        parts.append(
            f'<text x="{label_width}" y="{y + 17:.1f}" text-anchor="end" font-size="12.5" '
            f'fill="#334155">{esc(r.get("label", ""))}</text>'
        )
        x = bar_x
        for share, color in ((low, "#f87171"), (mid, "#fbbf24"), (high, "#34d399")):
            w = bar_w * share / total
            if w > 0:
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" fill="{color}"/>')
                if w > 34:
                    parts.append(
                        f'<text x="{x + w / 2:.1f}" y="{y + 15:.1f}" text-anchor="middle" '
                        f'font-size="10.5" fill="#7c2d12">{share:.0f}%</text>'
                    )
            x += w
        parts.append(
            f'<text x="{bar_x + bar_w + 10}" y="{y + 16:.1f}" font-size="11.5" fill="#0f172a" '
            f'font-weight="600">均值 {_num(r.get("mean"), 1)}</text>'
            f'<text x="{bar_x + bar_w + 10}" y="{y + 30:.1f}" font-size="10" fill="#94a3b8">'
            f'n={int(_safe(r.get("scored_users"))):,}</text>'
        )
    return _svg(width, height, "".join(parts), "维度得分结构")


# ---------------------------------------------------------------------------
# HTML 组件
# ---------------------------------------------------------------------------

def table(headers, rows, cls="", aligns=None) -> str:
    if not rows:
        return '<p class="empty">暂无数据</p>'
    aligns = aligns or ["left"] * len(headers)
    head = "".join(
        f'<th style="text-align:{a}">{esc(h)}</th>' for h, a in zip(headers, aligns)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td style="text-align:{a}">{c}</td>' for c, a in zip(row, aligns)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def status_pill(status: str) -> str:
    mapping = {"ok": ("good", "通过"), "warn": ("warn", "关注"), "info": ("info", "说明")}
    cls, text = mapping.get(str(status).lower(), ("info", str(status)))
    return f'<span class="pill {cls}">{esc(text)}</span>'


def kpi_card(label, value, note="", tone="") -> str:
    return (
        f'<div class="kpi {tone}"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{esc(note)}</div></div>'
    )


def section(title, body, sec_id, subtitle="") -> str:
    sub = f'<p class="section-sub">{esc(subtitle)}</p>' if subtitle else ""
    return (
        f'<section id="{sec_id}"><div class="section-head"><h2>{esc(title)}</h2>{sub}</div>'
        f'{body}</section>'
    )


# ---------------------------------------------------------------------------
# 报告各章节
# ---------------------------------------------------------------------------

def render_header(ov, title: str) -> str:
    chips = [
        f'分析期间 {ov.get("period_text", "—")}',
        f'输入 {ov.get("input_file", "—")}',
        f'{ov.get("input_size_mb", 0):,.1f} MB · {ov.get("columns_in_file", 0):,} 列',
        f'生成于 {ov.get("generated_at", "—")}',
    ]
    chip_html = "".join(f'<span class="chip">{esc(c)}</span>' for c in chips)
    return (
        '<header class="hero"><div class="wrap">'
        '<div class="eyebrow">PROFILING · 交易行为画像</div>'
        f'<h1>{esc(title)}</h1>'
        '<p class="hero-sub">基于银行流水特征的用户分群、健康评分与风险识别；'
        '报告仅消费中间统计结果，不重复计算。</p>'
        f'<div class="chips">{chip_html}</div>'
        '</div></header>'
        + render_nav()
    )


def render_nav() -> str:
    links = [
        ("summary", "总览"), ("findings", "核心结论"), ("data", "数据质量"),
        ("segments", "客群结构"), ("income", "收入形态"), ("lender", "负债结构"),
        ("dishonour", "拒付恢复"), ("dimensions", "维度评分"), ("features", "特征分布"),
        ("observations", "近期变化与缓冲"),
        ("risk", "风险标签"), ("consumption", "消费结构"), ("target", "目标分析"),
        ("correlation", "维度相关"), ("appendix", "附录"),
    ]
    html_links = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in links)
    return f'<nav class="toc"><div class="wrap toc-inner">{html_links}</div></nav>'


def render_summary(data) -> str:
    ov = data["overview"]
    seg = data["segment_summary"]
    flags = data["risk_flags"]
    target = data["target"]

    def flag_pct(name):
        row = flags[flags["flag"] == name]
        return _safe(row["prevalence_pct"].iloc[0]) if len(row) else math.nan

    def seg_row(name):
        row = seg[seg["segment"] == name]
        return row.iloc[0] if len(row) else None

    high = seg_row("高风险承压")
    insufficient = seg_row("数据不足")
    health = ov.get("health", {})
    health_med = health.get("p50", math.nan)
    cards = [
        kpi_card("分析申请数", f'{int(ov.get("rows", 0)):,}', "申请记录粒度"),
        kpi_card("可用交易流水", f'{int(ov.get("txn_rows", 0)):,}',
                 f'覆盖率 {ov.get("txn_share_pct", 0):.1f}%', "accent"),
        kpi_card("健康分中位数", _num(health_med, 1), "0–100，越高越稳健"),
        kpi_card("高风险客群", f'{high["share_pct"]:.1f}%' if high is not None else "—",
                 f'{int(high["users"]):,} 条' if high is not None else "", "danger"),
        kpi_card("未结清拒付", f'{flag_pct("flag_unresolved_dishonour"):.1f}%', "占可用流水用户"),
        kpi_card("多头借贷", f'{flag_pct("flag_multi_lender"):.1f}%', "≥3 家放款机构"),
        kpi_card("目标逾期率", _pct(target.get("rate")),
                 f'正样本 {int(target.get("positives", 0)):,} / 已到期 {int(target.get("valid_n", 0)):,}',
                 "warn" if _safe(target.get("positives")) < 30 else ""),
        kpi_card("数据不足", f'{insufficient["share_pct"]:.1f}%' if insufficient is not None else "—",
                 "无流水或健康分覆盖不足"),
    ]
    intro = (
        '<div class="callout"><strong>阅读提示：</strong>'
        '健康分与客群划分仅覆盖具备交易流水的申请；'
        '“数据不足”表示无流水或健康分覆盖率不足；个别维度仍可能有分数，不应将该客群解读为低风险或高风险。'
        '目标标签正样本很少时，分组逾期率仅作探索性参考。</div>'
    )
    return section("执行总览", f'<div class="kpi-grid">{"".join(cards)}</div>{intro}',
                   "summary", "关键规模、覆盖率与风险指标")


def render_findings(data) -> str:
    ins = data["insights"]
    findings = ins.get("key_findings", [])
    cards = "".join(
        f'<div class="finding"><div class="finding-title">{esc(f.get("title", ""))}</div>'
        f'<p>{esc(f.get("text", ""))}</p></div>'
        for f in findings
    )
    recs = "".join(f"<li>{esc(r)}</li>" for r in ins.get("recommendations", []))
    caveats = "".join(f"<li>{esc(c)}</li>" for c in ins.get("caveats", []))
    body = f'''
    <div class="finding-grid">{cards}</div>
    <div class="two-col">
      <div class="card"><h3>策略建议</h3><ol class="recs">{recs or "<li>暂无</li>"}</ol></div>
      <div class="card warn-card"><h3>使用边界</h3><ul class="caveats">{caveats or "<li>暂无</li>"}</ul></div>
    </div>'''
    return section("核心结论", body, "findings", "由中间统计结果自动生成，均可回溯到指标表")


def render_data(data) -> str:
    ov = data["overview"]
    q = data["quality_checks"]
    rows = []
    for _, r in q.iterrows():
        rows.append([
            esc(r["check"]),
            f'<strong>{esc(r["value"])}</strong>',
            status_pill(r["status"]),
            esc(r["detail"]),
        ])
    qtable = table(["检查项", "结果", "状态", "口径说明"], rows, cls="data-table",
                   aligns=["left", "left", "center", "left"])
    meta_cards = [
        kpi_card("数据字典匹配", f'{int(ov.get("dictionary", {}).get("matched_features", 0)):,}',
                 f'共 {int(ov.get("dictionary", {}).get("dictionary_features", 0)):,} 个特征'),
        kpi_card("哨兵冲突", f'{int(ov.get("dictionary", {}).get("sentinel_conflicts", 0)):,}',
                 "缺失值默认值核对"),
        kpi_card("注册特征", f'{int(ov.get("features_registered", 0)):,}',
                 f'参与评分 {int(ov.get("features_scored", 0)):,} 个'),
        kpi_card("本次读取列", f'{int(ov.get("columns_read", 0)):,}',
                 f'源表共 {int(ov.get("columns_in_file", 0)):,} 列'),
    ]
    note = (
        '<div class="callout"><strong>口径提醒：</strong>'
        '储蓄率（<code>balance</code> 模块）以余额变动为分子分母；'
        '收支结余率、支出收入比（<code>category</code> 模块）以交易分类汇总为口径。'
        '两者分母不同，不能直接互相验证或混用。'
        '活跃交易天数采用应用级去重天数，而非多账户累加天数。</div>'
    )
    return section("数据质量与口径", f'<div class="kpi-grid small">{"".join(meta_cards)}</div>'
                   f'{qtable}{note}', "data", "门槛、覆盖与恒等式校验")


def render_segments(data) -> str:
    ov = data["overview"]
    seg = data["segment_summary"]
    profiles = data["segment_profiles"]
    ins = data["insights"]
    order = [s for s in ov.get("segment_order", SEGMENT_ORDER) if s in set(seg["segment"])]
    seg = seg.set_index("segment").loc[order].reset_index()
    items = [(r["segment"], r["users"], SEGMENT_COLORS.get(r["segment"], "#94a3b8"))
             for _, r in seg.iterrows()]
    legend = "".join(
        f'<div class="legend-row"><span class="dot" style="background:{SEGMENT_COLORS.get(r["segment"], "#94a3b8")}"></span>'
        f'<span class="legend-label">{esc(r["segment"])}</span>'
        f'<span class="legend-value">{int(r["users"]):,} · {r["share_pct"]:.1f}%</span></div>'
        for _, r in seg.iterrows()
    )
    rows = []
    for _, r in seg.iterrows():
        rate = r.get("target_rate")
        ci = "—" if not math.isfinite(_safe(rate)) else f'{_pct(rate)} [{_pct(r.get("target_ci_low"))}, {_pct(r.get("target_ci_high"))}]'
        rows.append([
            f'<span class="dot" style="background:{SEGMENT_COLORS.get(r["segment"], "#94a3b8")}"></span> {esc(r["segment"])}',
            f'{int(r["users"]):,}', f'{r["share_pct"]:.1f}%',
            _num(r.get("health_median"), 1), _num(r.get("risk_flags_mean"), 2),
            f'{int(r.get("target_matured", 0)):,}', f'{int(r.get("target_pos", 0)):,}', ci,
        ])
    summary_tbl = table(
        ["客群", "记录数", "占比", "健康分中位数", "风险标签均值", "已到期", "正样本", "逾期率 [95%CI]"],
        rows, cls="data-table",
        aligns=["left", "right", "right", "right", "right", "right", "right", "right"],
    )
    narr = ins.get("segment_narratives", {})
    narr_cards = "".join(
        f'<div class="seg-note" style="border-left-color:{SEGMENT_COLORS.get(name, "#94a3b8")}">'
        f'<div class="seg-note-title">{esc(name)}</div><p>{esc(text)}</p></div>'
        for name, text in narr.items()
    )
    # 分群指标矩阵
    prof = profiles.copy()
    prof["metric_label"] = prof["metric"].map(dict(SEGMENT_METRICS))
    prof = prof[prof["metric"].isin([m for m, _ in SEGMENT_METRICS])]
    fmap = {r["key"]: r for _, r in data["feature_stats"].iterrows()}
    matrix_rows = []
    for metric, label in SEGMENT_METRICS:
        row = [esc(label)]
        for name in order:
            sub = prof[(prof["segment"] == name) & (prof["metric"] == metric)]
            if len(sub):
                fmt = fmap.get(metric, {}).get("fmt", "ratio") if metric in fmap else "ratio"
                row.append(_fmt(sub.iloc[0]["p50"], fmt))
            else:
                row.append("—")
        matrix_rows.append(row)
    matrix = table(["指标（中位数）"] + order, matrix_rows, cls="matrix",
                   aligns=["left"] + ["right"] * len(order))
    body = f'''
    <div class="segment-top">
      <div class="card donut-card">{donut_chart(items)}<div class="legend">{legend}</div></div>
      <div class="card">{summary_tbl}</div>
    </div>
    <div class="card"><h3>分群画像摘要</h3><div class="seg-notes">{narr_cards}</div></div>
    <div class="card"><h3>分群指标对比（中位数）</h3>
      <p class="muted">用于比较客群结构差异；缺失值以 “—” 表示，不代表 0。</p>{matrix}</div>'''
    return section("客群结构", body, "segments", "规则分群：健康分、现金流与风险标签联合判定")


def render_income(data) -> str:
    summary = data.get("income_summary")
    if summary is None or len(summary) == 0:
        return section("收入形态与发薪节律",
                       '<p class="empty">本次运行未生成收入形态中间结果。</p>',
                       "income", "工资、福利与多源收入的结构分型")
    smap = {str(r["metric"]): r for _, r in summary.iterrows()}

    def sv(key, default=math.nan):
        row = smap.get(key)
        return _safe(row["value"]) if row is not None else default

    gate = sv("gate_users", 0.0)

    def share(n):
        return f"占可用流水 {n / gate * 100:.1f}%" if gate else ""

    kpis = [
        kpi_card("稳定工资型", f"{int(sv('stable_wage', 0)):,}", share(sv("stable_wage", 0)), "accent"),
        kpi_card("不稳定工资型", f"{int(sv('unstable_wage', 0)):,}", share(sv("unstable_wage", 0)), "warn"),
        kpi_card("多源/非工资型", f"{int(sv('multi_source', 0)):,}", share(sv("multi_source", 0))),
        kpi_card("福利依赖型", f"{int(sv('welfare', 0)):,}", share(sv("welfare", 0))),
        kpi_card("无收入型", f"{int(sv('no_income', 0)):,}", share(sv("no_income", 0)), "danger"),
        kpi_card("连续有工资月数中位数", _num(sv("median_consecutive_months"), 1),
                 "截至窗口末连续发薪月数"),
        kpi_card("主雇主连续月数中位数", _num(sv("median_employer_streak"), 1),
                 "识别到主雇主的用户"),
        kpi_card("发薪间隔中位数", _fmt(sv("median_pay_interval"), "days"),
                 "识别到发薪节律的用户"),
    ]

    arch = data.get("income_archetypes")
    arch_colors = {
        "稳定工资": "#0f9d76", "不稳定工资": "#f59e0b", "多源/非工资": "#2f6fed",
        "福利依赖": "#7c5cff", "无收入": "#dc2626",
    }
    arch_items = []
    if arch is not None and len(arch):
        amap = arch.set_index("archetype")
        arch_items = [
            (name, _safe(amap.loc[name, "users"]) if name in amap.index else 0,
             arch_colors.get(name, "#94a3b8"))
            for name in INCOME_ARCHETYPES
        ]
    arch_donut = donut_chart(arch_items, center_label="可用流水用户")
    arch_legend = "".join(
        f'<div class="legend-row"><span class="dot" style="background:{color}"></span>'
        f'<span class="legend-label">{esc(label)}</span>'
        f'<span class="legend-value">{int(value):,} · {value / gate * 100:.1f}%</span></div>'
        for label, value, color in arch_items if gate
    )

    rhythm = data.get("income_pay_rhythm")
    rhythm_rows = []
    rhythm_items = []
    if rhythm is not None and len(rhythm):
        rmap = rhythm.set_index("rhythm")
        for name in PAY_RHYTHMS:
            if name not in rmap.index:
                continue
            r = rmap.loc[name]
            rhythm_items.append((name, _safe(r["users"]), f'{r["share_pct"]:.1f}%'))
            rhythm_rows.append([
                esc(name), f'{int(r["users"]):,}', f'{r["share_pct"]:.1f}%',
                _fmt(r["wage_share_median"], "pct"), _fmt(r["salary_cv_median"], "ratio"),
                _fmt(r["income_median"], "amount"),
                *[f'{r.get(f"{a}_pct", math.nan):.1f}%' for a in INCOME_ARCHETYPES],
            ])
    rhythm_chart = hbar_chart(rhythm_items, value_fmt="count", color="#0f9d76",
                              width=620, row_h=32, label_width=120)
    rhythm_tbl = table(
        ["发薪节律", "用户数", "占比", "工资占比中位数", "工资CV中位数", "收入中位数"]
        + [f"{a}占比" for a in INCOME_ARCHETYPES],
        rhythm_rows, cls="data-table",
        aligns=["left"] + ["right"] * (5 + len(INCOME_ARCHETYPES)),
    )

    arch_rows = []
    if arch is not None and len(arch):
        amap = arch.set_index("archetype")
        for name in INCOME_ARCHETYPES:
            if name not in amap.index:
                continue
            r = amap.loc[name]
            arch_rows.append([
                f'<span class="dot" style="background:{arch_colors.get(name, "#94a3b8")}"></span> {esc(name)}',
                f'{int(r["users"]):,}', f'{r["share_pct"]:.1f}%',
                _fmt(r["wage_share_median"], "pct"), _fmt(r["salary_cv_median"], "ratio"),
                _num(r["salary_active_months_median"], 1),
                _num(r["consecutive_months_median"], 1),
                _num(r["employer_streak_median"], 1),
                _fmt(r["pay_interval_median"], "days"),
                _fmt(r["income_median"], "amount"),
                _fmt(r["centrelink_share_median"], "pct"),
                _num(r["source_count_median"], 1),
            ])
    arch_tbl = table(
        ["收入形态", "用户数", "占比", "工资占比中位数", "工资CV中位数", "有工资月份中位数",
         "连续发薪月数中位数", "主雇主连续月数中位数", "发薪间隔中位数", "收入中位数",
         "福利收入占比中位数", "收入来源数中位数"],
        arch_rows, cls="data-table",
        aligns=["left"] + ["right"] * 11,
    )

    seg = data.get("income_segments")
    seg_rows = []
    if seg is not None and len(seg):
        ov = data["overview"]
        order = [s for s in ov.get("segment_order", SEGMENT_ORDER) if s in set(seg["segment"])]
        seg = seg.set_index("segment").loc[order].reset_index()
        for _, r in seg.iterrows():
            seg_rows.append([
                f'<span class="dot" style="background:{SEGMENT_COLORS.get(r["segment"], "#94a3b8")}"></span> {esc(r["segment"])}',
                f'{int(r["users"]):,}',
                *[f'{r.get(f"{a}_pct", math.nan):.1f}%' for a in INCOME_ARCHETYPES],
                f'{r.get("irregular_pay_pct", math.nan):.1f}%',
                f'{r.get("unrecognized_pay_pct", math.nan):.1f}%',
                _num(r.get("consecutive_months_median"), 1),
                _num(r.get("employer_streak_median"), 1),
            ])
    seg_tbl = table(
        ["客群", "用户数"] + [f"{a}占比" for a in INCOME_ARCHETYPES]
        + ["不规律发薪占比", "无工资/未识别占比", "连续发薪月数中位数", "主雇主连续月数中位数"],
        seg_rows, cls="data-table",
        aligns=["left"] + ["right"] * (2 + len(INCOME_ARCHETYPES) + 2),
    )

    tgt = data.get("income_archetype_target")
    tgt_rows = []
    matured_total = 0
    if tgt is not None and len(tgt):
        tmap = tgt.set_index("archetype")
        matured_total = int(tgt["matured"].sum())
        for name in INCOME_ARCHETYPES:
            if name not in tmap.index:
                continue
            r = tmap.loc[name]
            ci = ("—" if not math.isfinite(_safe(r["rate"]))
                  else f'{_pct(r["ci_low"])} – {_pct(r["ci_high"])}')
            tgt_rows.append([
                esc(name), f'{int(r["users"]):,}', f'{int(r["matured"]):,}',
                f'{int(r["positives"]):,}', _pct(r["rate"]), ci,
            ])
    tgt_tbl = table(["收入形态", "用户数", "已到期", "正样本", "逾期率", "95%CI"], tgt_rows,
                    cls="data-table", aligns=["left", "right", "right", "right", "right", "right"])
    target_note = (
        f'<div class="callout warn"><strong>小样本警示：</strong>收入形态分组后已到期样本合计仅 '
        f'{matured_total:,} 条，单组逾期率极易被个别样本驱动，仅用于提出假设。</div>'
        if matured_total else
        '<div class="callout">当前没有已到期目标样本，收入形态与逾期的关联暂不展示。</div>'
    )

    cfg = data.get("run_meta", {}).get("config", {})
    stable_share = _float(cfg.get("income_stable_wage_share"), 0.5)
    stable_months = _float(cfg.get("income_stable_min_months"), 5)
    stable_cv = _float(cfg.get("income_stable_cv"), 0.5)
    unstable_share = _float(cfg.get("income_unstable_wage_share"), 0.3)
    welfare_share = _float(cfg.get("flag_thresholds", {}).get("welfare_income_share"), 0.5)
    note = (
        '<div class="callout"><strong>口径说明：</strong>'
        f'稳定工资型要求工资收入占比 ≥{stable_share * 100:.0f}%、有工资月份 ≥{stable_months:.0f}、'
        f'工资金额 CV ≤{stable_cv:.2f}；福利收入占比 ≥{welfare_share * 100:.0f}% 优先判为福利依赖型；'
        f'其余有工资但工资占比 ≥{unstable_share * 100:.0f}% 判为不稳定工资型，'
        '工资占比更低的归入多源/非工资型。'
        '发薪节律来自行业模块的四类互斥标记，无工资或流水不足以识别时归入“无工资/未识别”。'
        '本节全部为观察项，不进入健康分，也不改变客群划分；现金收入与家庭内部转账可能造成误分类。</div>'
    )
    body = f'''
    <div class="kpi-grid">{''.join(kpis)}</div>
    <div class="two-col wide">
      <div class="card">{arch_donut}<div class="legend">{arch_legend}</div></div>
      <div class="card"><h3>发薪节律分布</h3>{rhythm_chart}</div>
    </div>
    <div class="card"><h3>收入形态画像（近182天）</h3>
      <p class="muted">中位数按形态内用户计算；工资CV仅在识别到工资金额时有值。</p>
      {arch_tbl}</div>
    <div class="card"><h3>发薪节律 × 收入形态</h3>{rhythm_tbl}</div>
    <div class="card"><h3>各客群收入形态分布</h3>
      <p class="muted">占比以客群内可用流水用户为分母；形态划分为规则判定，不含模型预测。</p>
      {seg_tbl}</div>
    <div class="card"><h3>收入形态 × 逾期率（探索性）</h3>{target_note}{tgt_tbl}</div>
    {note}'''
    return section("收入形态与发薪节律", body, "income",
                   "基于现有工资、福利与收入来源字段的结构分型，观察项")


def render_lender(data) -> str:
    summary = data.get("lender_summary")
    if summary is None or len(summary) == 0:
        return section("负债结构深挖", '<p class="empty">本次运行未生成借贷结构中间结果。</p>',
                       "lender", "放款、还款与机构结构")
    smap = {str(r["metric"]): r for _, r in summary.iterrows()}

    def sv(key, default=math.nan):
        row = smap.get(key)
        return _safe(row["value"]) if row is not None else default

    gate = sv("gate_users", 0.0)
    has_d = sv("has_disburse", 0.0)
    repay_only = sv("repay_only", 0.0)
    no_lender = sv("no_lender", 0.0)
    recent_high = sv("recent_share_high", 0.0)
    share_high = _float(data.get("run_meta", {}).get("config", {}).get("lender_recent_share_high"), 0.5)

    def share(n):
        return f"占可用流水 {n / gate * 100:.1f}%" if gate else ""

    kpis = [
        kpi_card("有放款记录", f"{int(has_d):,}", share(has_d), "accent"),
        kpi_card("仅还款用户", f"{int(repay_only):,}", share(repay_only)),
        kpi_card("无借贷记录", f"{int(no_lender):,}", share(no_lender) + "；不等于无负债", "warn"),
        kpi_card("放款机构数中位数", _num(sv("disburse_jgs_median"), 1),
                 "有放款记录用户"),
        kpi_card("近28天净融资扩张", f"{int(sv('recent_net_expand', 0)):,}",
                 share(sv("recent_net_expand", 0)), "danger"),
        kpi_card(f"近28天放款占比≥{share_high * 100:.0f}%", f"{int(recent_high):,}",
                 f"占有放款记录 {recent_high / has_d * 100:.1f}%" if has_d else "有放款记录用户",
                 "warn"),
        kpi_card("单机构放款集中度中位数", _pct(sv("lender_concentration_median")),
                 "最大机构放款 / 放款总额"),
    ]

    coverage = [
        ("有放款记录", has_d, "#2f6fed"),
        ("仅有还款记录", repay_only, "#f59e0b"),
        ("无借贷记录", no_lender, "#cbd5e1"),
        ("数据不足", sv("data_unknown", 0.0), "#94a3b8"),
    ]
    cov_donut = donut_chart(coverage, center_label="可用流水用户")
    cov_legend = "".join(
        f'<div class="legend-row"><span class="dot" style="background:{color}"></span>'
        f'<span class="legend-label">{esc(label)}</span>'
        f'<span class="legend-value">{int(value):,} · {value / gate * 100:.1f}%</span></div>'
        for label, value, color in coverage if gate
    )

    inst = data.get("lender_institutions")
    inst_items = []
    if inst is not None and len(inst):
        inst_items = [(str(r["bucket"]), _safe(r["users"]), f'{r["share_pct"]:.1f}%')
                      for _, r in inst.iterrows()]
    inst_chart = hbar_chart(inst_items, value_fmt="count", color="#7c5cff",
                            width=600, row_h=32, label_width=100)

    behavior = data.get("lender_behavior")
    beh_order = {name: idx for idx, name in enumerate(LENDER_BEHAVIORS)}
    all_beh = None
    if behavior is not None and len(behavior):
        all_beh = behavior[behavior["segment"] == "全部"].copy()
        all_beh["_order"] = all_beh["behavior"].map(beh_order).fillna(len(beh_order))
        all_beh = all_beh.sort_values("_order")
    beh_items = ([(r["behavior"], _safe(r["users"]), f'{r["share_pct"]:.1f}%')
                  for _, r in all_beh.iterrows()] if all_beh is not None else [])
    beh_chart = hbar_chart(beh_items, value_fmt="count", color="#0f9d76",
                           width=600, row_h=32, label_width=110)

    ov = data["overview"]
    seg_lender = data.get("lender_segments")
    seg_order = []
    if seg_lender is not None and len(seg_lender):
        present = set(seg_lender["segment"])
        seg_order = [s for s in ov.get("segment_order", SEGMENT_ORDER) if s in present]
        seg_lender = seg_lender.set_index("segment").loc[seg_order].reset_index()
    seg_rows = []
    if seg_lender is not None and len(seg_lender):
        for _, r in seg_lender.iterrows():
            seg_rows.append([
                f'<span class="dot" style="background:{SEGMENT_COLORS.get(r["segment"], "#94a3b8")}"></span> {esc(r["segment"])}',
                f'{int(r["users"]):,}',
                f'{r["has_disburse_pct"]:.1f}%', f'{r["repay_only_pct"]:.1f}%',
                f'{r["no_lender_pct"]:.1f}%',
                _num(r["disburse_jgs_median"], 1), f'{r["multi_lender_pct"]:.1f}%',
                f'{r["net_expand_pct"]:.1f}%', f'{r["net_shrink_pct"]:.1f}%',
                _pct(r["recent_disburse_share_median"]),
                _pct(r["lender_concentration_median"]),
            ])
    seg_tbl = table(
        ["客群", "用户数", "有放款", "仅还款", "无记录", "机构数中位数",
         "机构≥3（占有放款）", "近28天净扩张", "近28天净偿还",
         "近28天放款占比中位数", "单机构集中度中位数"],
        seg_rows, cls="data-table",
        aligns=["left"] + ["right"] * 10,
    )

    beh_matrix_rows = []
    seg_cols = ["全部"] + seg_order
    if behavior is not None and len(behavior):
        bmap = {(r["segment"], r["behavior"]): _safe(r["share_pct"])
                for _, r in behavior.iterrows()}
        for bname in LENDER_BEHAVIORS:
            row = [esc(bname)]
            for seg_name in seg_cols:
                v = bmap.get((seg_name, bname))
                row.append(f"{v:.1f}%" if math.isfinite(_safe(v)) else "—")
            beh_matrix_rows.append(row)
    beh_matrix = table(["借贷行为"] + seg_cols, beh_matrix_rows, cls="matrix",
                       aligns=["left"] + ["right"] * len(seg_cols))

    products = data.get("lender_products")
    prod_rows = []
    if products is not None and len(products):
        for _, r in products.iterrows():
            prod_rows.append([
                esc(r["label"]),
                "放款" if r["direction"] == "disburse" else "还款",
                f'{int(r["users"]):,}', f'{r["user_share_pct"]:.1f}%',
                _fmt(r["amount"], "amount"), f'{r["amount_share_pct"]:.1f}%',
                f'{int(r["count"]):,}', _fmt(r["avg_ticket"], "amount"),
            ])
    prod_tbl = table(["产品", "方向", "用户数", "用户占比", "金额", "金额占比", "笔数", "笔均"],
                     prod_rows, cls="data-table",
                     aligns=["left", "center", "right", "right", "right", "right", "right", "right"])

    top = data.get("lender_top_competitors")
    comp_rows = []
    if top is not None and len(top):
        for _, r in top.iterrows():
            comp_rows.append([
                esc(r["competitor"]), f'{int(r["users"]):,}', f'{r["user_share_pct"]:.1f}%',
                _fmt(r["amount"], "amount"), f'{r["amount_share_pct"]:.1f}%',
            ])
    comp_tbl = table(["放款机构", "用户数", "占可识别借款用户", "放款金额", "占竞品放款金额"],
                     comp_rows, cls="data-table",
                     aligns=["left", "right", "right", "right", "right"])

    tgt = data.get("lender_behavior_target")
    tgt_rows = []
    matured_total = 0
    if tgt is not None and len(tgt):
        tgt = tgt.copy()
        tgt["_order"] = tgt["behavior"].map(beh_order).fillna(len(beh_order))
        tgt = tgt.sort_values("_order")
        matured_total = int(tgt["matured"].sum())
        for _, r in tgt.iterrows():
            ci = ("—" if not math.isfinite(_safe(r["rate"]))
                  else f'{_pct(r["ci_low"])} – {_pct(r["ci_high"])}')
            tgt_rows.append([
                esc(r["behavior"]), f'{int(r["users"]):,}', f'{int(r["matured"]):,}',
                f'{int(r["positives"]):,}', _pct(r["rate"]), ci,
            ])
    tgt_tbl = table(["借贷行为", "用户数", "已到期", "正样本", "逾期率", "95%CI"], tgt_rows,
                    cls="data-table",
                    aligns=["left", "right", "right", "right", "right", "right"])

    target_note = (
        f'<div class="callout warn"><strong>小样本警示：</strong>借贷行为分组后已到期样本合计仅 '
        f'{matured_total:,} 条，单组逾期率极易被个别样本驱动，仅用于提出假设。</div>'
        if matured_total else
        '<div class="callout">当前没有已到期目标样本，借贷行为与逾期的关联暂不展示。</div>'
    )
    tol = _float(data.get("run_meta", {}).get("config", {}).get("lender_net_tolerance"), 0.05)
    tol_txt = f"{tol * 100:.0f}%" if math.isfinite(tol) else "5%"
    note = (
        '<div class="callout"><strong>口径说明：</strong>'
        '本节指标均为观察项，不进入健康分，也不改变客群划分。'
        '“无借贷记录”只代表流水与机构映射未识别到借贷交易，不等于实际无负债。'
        '产品金额占比在已分类交易内计算，未映射交易不计入产品分项；'
        '“占可识别借款用户”的分母为能识别出放款机构的用户。'
        '风险标签“多头借贷”仅统计机构名单内竞品，本节机构数采用全量去重口径，两者不可直接比较。</div>'
    )
    body = f'''
    <div class="kpi-grid">{''.join(kpis)}</div>
    <div class="two-col wide">
      <div class="card">{cov_donut}<div class="legend">{cov_legend}</div></div>
      <div class="card"><h3>放款机构数分布（占有放款记录用户）</h3>{inst_chart}</div>
    </div>
    <div class="two-col wide">
      <div class="card"><h3>近28天借贷行为分型</h3>{beh_chart}</div>
      <div class="card"><h3>行为分型说明</h3>
        <p class="muted">按近28天放款与还款金额判定：仅偿还、仅新增借款、净偿还、
        借还平衡、净融资扩张。借还金额差异在 ±{esc(tol_txt)} 以内视为平衡。
        行为分型只描述近期资金流向，不判断负债是否健康。</p>
        <p class="muted">“近28天无借贷”表示存在历史借贷而近期无借还记录；“无借贷记录”表示182天窗口内未识别到借还记录。
        两者均不能直接解释为实际零负债；字段缺失单列为数据不足。</p>
      </div>
    </div>
    <div class="card"><h3>各客群负债参与与融资动态</h3>
      <p class="muted">占比均以客群内可用流水用户为分母；“机构≥3”以该客群有放款记录用户为分母。</p>
      {seg_tbl}</div>
    <div class="card"><h3>各客群借贷行为分布</h3>{beh_matrix}</div>
    <div class="card"><h3>放款与还款产品结构（近182天）</h3>
      <p class="muted">用户占比分别以有放款、有还款记录用户为分母；金额占比在已分类产品内计算。</p>
      {prod_tbl}</div>
    <div class="two-col wide">
      <div class="card"><h3>头部放款机构（Top {len(comp_rows)}）</h3>{comp_tbl}</div>
      <div class="card"><h3>借贷行为 × 逾期率（探索性）</h3>{target_note}{tgt_tbl}</div>
    </div>
    {note}'''
    return section("负债结构深挖", body, "lender",
                   "基于现有 lender 字段：参与结构、机构分布、行为分型与产品构成")


def render_dishonour(data) -> str:
    summary = data.get("dishonour_summary")
    if summary is None or len(summary) == 0:
        return section("拒付恢复", '<p class="empty">本次运行未生成拒付恢复中间结果。</p>',
                       "dishonour", "拒付结清、未结清与恢复结构")
    smap = {str(r["metric"]): r for _, r in summary.iterrows()}

    def sv(key, default=math.nan):
        row = smap.get(key)
        return _safe(row["value"]) if row is not None else default

    gate = sv("gate_users", 0.0)
    any_d = sv("any_dishonour", 0.0)

    def sgate(n):
        return f"占可用流水 {n / gate * 100:.1f}%" if gate else ""

    def sdish(n):
        return f"占有拒付记录 {n / any_d * 100:.1f}%" if any_d else ""

    kpis = [
        kpi_card("有拒付记录", f"{int(any_d):,}", sgate(any_d), "accent"),
        kpi_card("存在未结清", f"{int(sv('unresolved_users', 0)):,}",
                 sgate(sv("unresolved_users", 0)), "danger"),
        kpi_card("拒付已全部解决", f"{int(sv('resolved_only', 0)):,}",
                 sdish(sv("resolved_only", 0))),
        kpi_card("部分未解决", f"{int(sv('partial', 0)):,}", sdish(sv("partial", 0)), "warn"),
        kpi_card("全部未解决", f"{int(sv('unresolved_only', 0)):,}",
                 sdish(sv("unresolved_only", 0)), "danger"),
        kpi_card("拒付结清率", _pct(sv("resolved_ratio")), "按拒付次数口径"),
        kpi_card("未结清金额", _fmt(sv("unresolved_amt"), "amount"), sgate(sv("unresolved_users", 0))),
        kpi_card("未结清金额中位数", _fmt(sv("unresolved_amt_median"), "amount"),
                 "存在未结清拒付的用户"),
        kpi_card("单类未结清机构数最大值", _num(sv("unresolved_inst_max"), 0),
                 "单一拒付类型、单用户"),
    ]

    recovery = data.get("dishonour_recovery")
    status_colors = {
        "无拒付": "#cbd5e1", "已全部解决": "#0f9d76",
        "部分解决": "#f59e0b", "未解决": "#dc2626",
    }
    status_items = []
    status_legend = ""
    if recovery is not None and len(recovery):
        all_rows = recovery[recovery["segment"] == "全部"].set_index("status")
        status_items = [
            (s, _safe(all_rows.loc[s, "users"]) if s in all_rows.index else 0,
             status_colors.get(s, "#94a3b8"))
            for s in DISHONOUR_STATUSES
        ]
        status_legend = "".join(
            f'<div class="legend-row"><span class="dot" style="background:{color}"></span>'
            f'<span class="legend-label">{esc(label)}</span>'
            f'<span class="legend-value">{int(value):,} · {value / gate * 100:.1f}%</span></div>'
            for label, value, color in status_items if gate
        )
    status_donut = donut_chart(status_items, center_label="可用流水用户")

    types = data.get("dishonour_types")
    type_items = []
    type_rows = []
    if types is not None and len(types):
        ranked = types.sort_values("unresolved_amt", ascending=False)
        type_items = [(r["label"], _safe(r["unresolved_amt"]),
                       f'结清率 {_pct(r["recovery_rate"])}')
                      for _, r in ranked.iterrows() if _safe(r["unresolved_amt"]) > 0]
        for _, r in ranked.iterrows():
            type_rows.append([
                esc(r["label"]), f'{int(r["users"]):,}',
                f'{int(r["unresolved_cnt"]):,}', _fmt(r["unresolved_amt"], "amount"),
                _num(r["unresolved_inst_max"], 0),
                f'{int(r["resolved_cnt"]):,}', _fmt(r["resolved_amt"], "amount"),
                f'{int(r["catchup_cnt"]):,}', _pct(r["recovery_rate"]),
            ])
    type_chart = hbar_chart(type_items, value_fmt="amount", color="#dc2626",
                            width=620, row_h=34, label_width=110)
    type_tbl = table(
        ["拒付类型", "涉及用户数", "未结清次数", "未结清金额", "未结清机构数最大值",
         "已结清次数", "已结清金额", "补扣结清次数", "结清率"],
        type_rows, cls="data-table", aligns=["left"] + ["right"] * 8,
    )

    seg_status_rows = []
    seg_perf_rows = []
    if recovery is not None and len(recovery):
        seg_names = [s for s in SEGMENT_ORDER if s in set(recovery["segment"]) and s != "数据不足"]
        for name in seg_names:
            row = [f'<span class="dot" style="background:{SEGMENT_COLORS.get(name, "#94a3b8")}"></span> {esc(name)}']
            for s in DISHONOUR_STATUSES:
                sub = recovery[(recovery["segment"] == name) & (recovery["status"] == s)]
                row.append(f'{int(sub["users"].iloc[0]):,}' if len(sub) else "0")
            seg_status_rows.append(row)
    seg_perf = data.get("dishonour_segments")
    if seg_perf is not None and len(seg_perf):
        ov = data["overview"]
        order = [s for s in ov.get("segment_order", SEGMENT_ORDER) if s in set(seg_perf["segment"])]
        seg_perf = seg_perf.set_index("segment").loc[order].reset_index()
        for _, r in seg_perf.iterrows():
            seg_perf_rows.append([
                f'<span class="dot" style="background:{SEGMENT_COLORS.get(r["segment"], "#94a3b8")}"></span> {esc(r["segment"])}',
                f'{int(r["users"]):,}', f'{r["any_dishonour_pct"]:.1f}%',
                f'{r["resolved_only_pct"]:.1f}%', f'{r["partial_pct"]:.1f}%',
                f'{r["unresolved_only_pct"]:.1f}%', f'{r["unresolved_user_pct"]:.1f}%',
                _pct(r["resolved_ratio"]), _fmt(r["unresolved_amt_median"], "amount"),
            ])
    seg_status_tbl = table(["客群"] + DISHONOUR_STATUSES, seg_status_rows, cls="matrix",
                           aligns=["left"] + ["right"] * len(DISHONOUR_STATUSES))
    seg_perf_tbl = table(
        ["客群", "用户数", "有拒付占比", "已全部解决", "部分未解决", "全部未解决",
         "存在未结清占比", "拒付结清率", "未结清金额中位数"],
        seg_perf_rows, cls="data-table", aligns=["left"] + ["right"] * 8,
    )

    tgt = data.get("dishonour_recovery_target")
    tgt_rows = []
    matured_total = 0
    if tgt is not None and len(tgt):
        tmap = tgt.set_index("status")
        matured_total = int(tgt["matured"].sum())
        for s in DISHONOUR_STATUSES:
            if s not in tmap.index:
                continue
            r = tmap.loc[s]
            ci = ("—" if not math.isfinite(_safe(r["rate"]))
                  else f'{_pct(r["ci_low"])} – {_pct(r["ci_high"])}')
            tgt_rows.append([
                esc(s), f'{int(r["users"]):,}', f'{int(r["matured"]):,}',
                f'{int(r["positives"]):,}', _pct(r["rate"]), ci,
            ])
    tgt_tbl = table(["拒付状态", "用户数", "已到期", "正样本", "逾期率", "95%CI"], tgt_rows,
                    cls="data-table", aligns=["left", "right", "right", "right", "right", "right"])
    target_note = (
        f'<div class="callout warn"><strong>小样本警示：</strong>拒付状态分组后已到期样本合计仅 '
        f'{matured_total:,} 条，单组逾期率极易被个别样本驱动，仅用于提出假设。</div>'
        if matured_total else
        '<div class="callout">当前没有已到期目标样本，拒付状态与逾期的关联暂不展示。</div>'
    )

    note = (
        '<div class="callout"><strong>口径说明：</strong>'
        '拒付状态来自 `3gp` 拒付识别口径：在观察窗口内识别到同机构补扣的记为已解决，'
        '否则记为未解决；“未解决”不等同于当前仍逾期，也不等同于客户主观违约。'
        '结清率按拒付次数计算，未结清金额占比按金额计算，两者分母不同。'
        '本节全部为观察项，不进入健康分，也不改变客群划分；'
        '拒付识别依赖银行流水可见性与机构映射，未映射交易可能被归入其他类型。</div>'
    )
    body = f'''
    <div class="kpi-grid">{''.join(kpis)}</div>
    <div class="two-col wide">
      <div class="card">{status_donut}<div class="legend">{status_legend}</div></div>
      <div class="card"><h3>未结清金额 × 拒付类型</h3>{type_chart}</div>
    </div>
    <div class="card"><h3>拒付类型结构（近182天）</h3>
      <p class="muted">“涉及用户数”为该类型下存在已结清或未结清拒付的用户；机构数最大值按单用户、单类型统计。</p>
      {type_tbl}</div>
    <div class="card"><h3>各客群拒付状态人数</h3>{seg_status_tbl}</div>
    <div class="card"><h3>各客群拒付恢复指标</h3>
      <p class="muted">未结清金额中位数仅在存在未结清拒付的用户内计算。</p>
      {seg_perf_tbl}</div>
    <div class="card"><h3>拒付状态 × 逾期率（探索性）</h3>{target_note}{tgt_tbl}</div>
    {note}'''
    return section("拒付恢复", body, "dishonour",
                   "拒付结清、未结清金额与机构结构，观察项")


def render_observations(data) -> str:
    bands = data.get("observation_bands")
    if bands is None or bands.empty:
        return section("近期变化与支出缓冲", '<p class="empty">请重新运行分析以生成新增维度。</p>', "observations")
    stats = data["feature_stats"].set_index("key")
    cards = []
    details = []
    for key in OBSERVATION_BANDS:
        if key not in stats.index:
            continue
        stat = stats.loc[key]
        group = bands[(bands["key"] == key) & (bands["segment"] == "全部")]
        items = [(r["band"], _float(r["share_pct"], 0) / 100, f'{int(r["users"]):,}条') for _, r in group.iterrows()]
        chart = hbar_chart(items, value_fmt="pct", width=600, label_width=175, row_h=32)
        cards.append(
            f'<div class="card"><h3>{esc(stat["label"])}</h3>'
            f'<p class="muted">中位数 {_fmt(stat["p50"], stat["fmt"])} · 有效 {int(stat["valid_n"]):,} 条'
            f' · 流水覆盖率 {_pct_points(stat["coverage_txn_pct"])}</p>{chart}'
            f'<p class="muted">{esc(stat["definition"])}</p></div>'
        )
        target_rows = [[esc(r["band"]), f'{int(r["users"]):,}', _pct_points(r["share_pct"]),
                        f'{int(r["matured"]):,}', f'{int(r["positives"]):,}', _pct(r["rate"]),
                        f'{_pct(r["ci_low"])} – {_pct(r["ci_high"])}'] for _, r in group.iterrows()]
        profiles = data["segment_profiles"]
        profiles = profiles[profiles["metric"] == key]
        segment_rows = [[esc(r["segment"]), f'{int(r["n"]):,}', _fmt(r["p50"], stat["fmt"]),
                         _fmt(r["p25"], stat["fmt"]), _fmt(r["p75"], stat["fmt"])] for _, r in profiles.iterrows()]
        details.append(
            f'<details class="card"><summary>{esc(stat["label"])}：分群与目标交叉</summary>'
            + table(["客群", "有效条数", "中位数", "P25", "P75"], segment_rows, cls="data-table")
            + table(["观察区间", "申请数", "占流水比例", "有效标签数", "正样本", "目标率", "95%CI"], target_rows, cls="data-table")
            + '</details>'
        )
    note = ('<div class="callout">新增六个指标仅作观察，不计入健康分。趋势比较近28天与此前154天的日均金额，'
            '两个区间互不重叠；窗口完整性与近期新入账账户会影响结果，不能直接解释为收入增长率。'
            '历史基线为0、源字段缺失或分母为0时归为数据不足。支出覆盖天数按历史支出速度估算，'
            '并非未来可用资金预测。刚性支出与固定账单可能重叠，不可相加。'
            '分箱为观察阈值，目标关联需结合样本数与置信区间解读。</div>')
    return section("近期变化与支出缓冲", note + '<div class="two-col">' + ''.join(cards) + '</div>' + ''.join(details),
                   "observations", "近期收支变化、刚性支出负担与余额覆盖能力；分析单位为申请记录")


def render_dimensions(data) -> str:
    dims = data["dimension_summary"]
    dims = dims[dims["dimension"].isin(SCORED_DIMS)].copy()
    order = [d for d in SCORED_DIMS if d in set(dims["dimension"])]
    dims = dims.set_index("dimension").loc[order].reset_index()
    thresholds = data["overview"].get("thresholds", {})
    chart = stacked_score_chart(dims.to_dict("records"), score_low=float(thresholds.get("score_low", 40)),
                                score_high=float(thresholds.get("score_high", 70)))
    rows = []
    for _, r in dims.iterrows():
        rows.append([
            f'<span class="dot" style="background:{DIM_COLORS.get(r["dimension"], "#94a3b8")}"></span> {esc(r["label"])}',
            f'{r["weight"] * 100:.0f}%' if _safe(r["weight"]) > 0 else "—",
            _num(r["mean"], 1), _num(r["p50"], 1),
            f'{_num(r["p10"], 1)} – {_num(r["p90"], 1)}',
            f'{int(r["scored_users"]):,}', f'{r["scored_pct"]:.1f}%',
            f'{int(r["features_scored"])} / {int(r["features_total"])}',
        ])
    tbl = table(["维度", "健康分权重", "均值", "中位数", "P10–P90", "评分人数", "覆盖率", "评分特征"],
                rows, cls="data-table",
                aligns=["left", "right", "right", "right", "right", "right", "right", "right"])
    health = data["overview"].get("health", {})
    min_cov = _safe(data["run_meta"].get("config", {}).get("min_health_coverage"))
    note = (
        f'<div class="callout">综合健康分 = 各维度得分按权重加权，要求最低覆盖率 '
        f'{_pct(min_cov) if math.isfinite(min_cov) else "—"}。'
        f'样本健康分中位数 {_num(health.get("p50"), 1)}，P10–P90 '
        f'{_num(health.get("p10"), 1)}–{_num(health.get("p90"), 1)}。'
        '负债维度得分越高表示压力越低。</div>'
    )
    body = f'<div class="card">{chart}</div><div class="card">{tbl}</div>{note}'
    return section("维度评分", body, "dimensions", "百分位评分：得分反映用户在样本中的相对位置")


def render_features(data) -> str:
    stats = data["feature_stats"]
    hist = data["feature_histograms"]
    fmap = {r["key"]: r for _, r in stats.iterrows()}
    cards = []
    for key, fallback in FEATURE_PANELS:
        row = fmap.get(key)
        if row is None:
            continue
        h = hist[hist["key"] == key]
        subtitle = (
            f'n={int(row["valid_n"]):,} · 覆盖率 {row["coverage_txn_pct"]:.1f}% · '
            f'中位 {_fmt(row["p50"], row["fmt"])} · P90 {_fmt(row["p90"], row["fmt"])}'
        )
        cards.append(
            f'<div class="chart-card"><div class="chart-title">{esc(row["label"])}'
            f'<span>{esc(fallback if fallback != row["label"] else "")}</span></div>'
            f'{histogram_chart(h, row)}<div class="chart-meta">{esc(subtitle)}</div></div>'
        )
    body = f'<div class="chart-grid">{"".join(cards)}</div>'
    note = (
        '<div class="callout">直方图截取 P1–P99 区间，区间外样本不进入柱形但计入统计表；'
        '虚线为样本中位数。金额单位沿用源数据（推断为澳元）。</div>'
    )
    return section("特征分布", body + note, "features", "关键特征的分布形态、集中趋势与长尾")


def render_risk(data) -> str:
    flags = data["risk_flags"].copy()
    flags = flags.sort_values("prevalence_pct", ascending=False)
    items = [(r["label"], _safe(r["prevalence_pct"]) / 100.0, "")
             for _, r in flags.iterrows()]
    chart = hbar_chart(items, value_fmt="pct", color="#dc2626", label_width=190)
    rows = []
    for _, r in flags.iterrows():
        kind = '<span class="tag risk">风险</span>' if bool(r["is_risk"]) else '<span class="tag obs">观察</span>'
        rows.append([
            esc(r["label"]), kind, esc(r["description"]),
            f'{int(r["users"]):,}', f'{int(r["denominator"]):,}',
            f'{r["prevalence_pct"]:.1f}%',
        ])
    tbl = table(["标签", "类型", "定义", "触发数", "分母", "触发率"], rows, cls="data-table",
                aligns=["left", "center", "left", "right", "right", "right"])
    body = f'<div class="two-col wide"><div class="card">{chart}</div><div class="card">{tbl}</div></div>'
    return section("风险标签", body, "risk", "风险标签用于分群与人工复核，不单独作为授信拒绝依据")


def render_consumption(data) -> str:
    mix = data["category_mix"].copy()
    mix = mix.sort_values("mean_share", ascending=False).head(16)
    items = [(r["label"], _safe(r["mean_share"]), f'非零 {r["nonzero_pct"]:.1f}%')
             for _, r in mix.iterrows()]
    chart = hbar_chart(items, value_fmt="pct", color="#0f8b8d", label_width=200, row_h=32)
    rows = []
    for _, r in mix.iterrows():
        rows.append([
            esc(r["label"]), _pct(r["mean_share"]), _pct(r["median_share"]),
            _pct(r["p75_share"]), f'{r["nonzero_pct"]:.1f}%',
        ])
    tbl = table(["类别", "均值占比", "中位占比", "P75", "非零用户占比"], rows, cls="data-table",
                aligns=["left", "right", "right", "right", "right"])
    body = f'<div class="two-col wide"><div class="card">{chart}</div><div class="card">{tbl}</div></div>'
    note = ('<div class="callout">支出占比以交易分类汇总为分母，转账类占比通常较高，'
            '解读消费结构时应剔除转账后再比较可选消费与必需支出。</div>')
    return section("消费结构", body + note, "consumption", "交易分类口径的支出构成")


def render_target(data) -> str:
    target = data["target"]
    if not target.get("available") or int(target.get("valid_n", 0)) == 0:
        return section("目标分析", '<div class="callout">目标列不存在或没有已到期样本，本次未做目标关联分析。</div>',
                       "target", "逾期标签关联")
    by_seg = data["target_by_segment"]
    by_band = data["target_by_score_band"]
    smd = data["target_feature_smd"]
    kpis = [
        kpi_card("已到期样本", f'{int(target.get("valid_n", 0)):,}', f'目标：{target.get("label", "")}'),
        kpi_card("正样本", f'{int(target.get("positives", 0)):,}', target.get("label", ""), "danger"),
        kpi_card("整体逾期率", _pct(target.get("rate")), "样本内比例", "warn"),
        kpi_card("95% 置信区间", f'{_pct(target.get("ci_low"))} – {_pct(target.get("ci_high"))}',
                 "Wilson 区间"),
    ]
    seg_chart = rate_ci_chart([
        {"segment": r["segment"], "matured": r["target_matured"], "rate": r["target_rate"],
         "ci_low": r["target_ci_low"], "ci_high": r["target_ci_high"]}
        for _, r in by_seg.iterrows()
    ])
    seg_rows = []
    for _, r in by_seg.iterrows():
        ci = ("—" if not math.isfinite(_safe(r["target_rate"]))
              else f'{_pct(r["target_ci_low"])} – {_pct(r["target_ci_high"])}')
        seg_rows.append([
            esc(r["segment"]), f'{int(r["users"]):,}', f'{int(r["target_matured"]):,}',
            f'{int(r["target_pos"]):,}', _pct(r["target_rate"]), ci,
        ])
    seg_tbl = table(["客群", "用户数", "已到期", "正样本", "逾期率", "95%CI"], seg_rows,
                    cls="data-table", aligns=["left", "right", "right", "right", "right", "right"])

    health_band = by_band[by_band["score_key"] == "health"].copy()
    band_chart = rate_ci_chart([
        {"segment": r["band_label"], "matured": r["matured"], "rate": r["rate"],
         "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
        for _, r in health_band.iterrows()
    ])
    band_rows = []
    for _, r in health_band.iterrows():
        ci = ("—" if not math.isfinite(_safe(r["rate"]))
              else f'{_pct(r["ci_low"])} – {_pct(r["ci_high"])}')
        band_rows.append([
            esc(r["band_label"]), f'{int(r["users"]):,}', f'{int(r["matured"]):,}',
            f'{int(r["positives"]):,}', _pct(r["rate"]), ci,
        ])
    band_tbl = table(["健康分区间", "用户数", "已到期", "正样本", "逾期率", "95%CI"], band_rows,
                     cls="data-table", aligns=["left", "right", "right", "right", "right", "right"])

    top_smd = smd.head(12)
    smd_items = [(r["label"], _safe(r["smd"]), f'n+={int(r["n_pos"])}, n-={int(r["n_neg"])}')
                 for _, r in top_smd.iterrows()]
    smd_chart = diverging_chart(smd_items)
    smd_rows = []
    for _, r in top_smd.iterrows():
        dim = str(r.get("dimension", ""))
        dim_label = "评分" if dim == "score" else DIM_LABELS.get(dim, dim)
        smd_rows.append([
            esc(r["label"]), esc(dim_label), f'{int(r["n_pos"]):,}',
            f'{int(r["n_neg"]):,}', _num(r["mean_pos"], 2), _num(r["mean_neg"], 2),
            f'{_signed(r["smd"])}',
        ])
    smd_tbl = table(["指标", "维度", "正样本 n", "负样本 n", "正样本均值", "负样本均值", "SMD"],
                    smd_rows, cls="data-table",
                    aligns=["left", "left", "right", "right", "right", "right", "right"])
    body = f'''
    <div class="kpi-grid">{''.join(kpis)}</div>
    <div class="callout warn"><strong>小样本警示：</strong>正样本仅 {int(target.get("positives", 0)):,} 条，
    分组逾期率与 SMD 可能由个别样本驱动。所有目标关联仅用于提出假设，不能作为上线或拒绝依据。</div>
    <div class="card"><h3>分群逾期率</h3>{seg_chart}{seg_tbl}</div>
    <div class="card"><h3>健康分分层逾期率</h3>{band_chart}{band_tbl}</div>
    <div class="card"><h3>正负样本差异（SMD，Top 12）</h3>
      <p class="muted">SMD 为正表示正样本（逾期）该指标更高；|SMD| &lt; 0.2 通常视为差异很小。</p>
      {smd_chart}{smd_tbl}</div>'''
    return section("目标分析（探索性）", body, "target", "逾期标签与客群、健康分、特征的关联")


def _corr_color(v) -> tuple[str, str]:
    v = _safe(v)
    if not math.isfinite(v):
        return "#f8fafc", "#cbd5e1"
    a = min(abs(v), 1.0)
    if v >= 0:
        bg = f"rgba(220,38,38,{0.06 + 0.60 * a:.3f})"
    else:
        bg = f"rgba(47,111,237,{0.06 + 0.60 * a:.3f})"
    fg = "#ffffff" if a > 0.55 else "#0f172a"
    return bg, fg


def render_correlation(data) -> str:
    corr = data["correlations"]
    keys = [d for d in SCORED_DIMS] + ["health"]
    cmap = {(r["row"], r["col"]): _safe(r["corr"]) for _, r in corr.iterrows()}
    head = "".join(f"<th>{esc(DIM_LABELS.get(k, k))}</th>" for k in keys)
    rows = []
    for a in keys:
        cells = [f'<th class="row-head">{esc(DIM_LABELS.get(a, a))}</th>']
        for b in keys:
            v = cmap.get((a, b), cmap.get((b, a), math.nan))
            bg, fg = _corr_color(v)
            txt = "—" if not math.isfinite(v) else f"{v:.2f}"
            cells.append(f'<td style="background:{bg};color:{fg}">{txt}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    heat = (f'<div class="table-wrap"><table class="heatmap"><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')
    body = f'<div class="card">{heat}</div>'
    note = ('<div class="callout">红色为正相关、蓝色为负相关，颜色深浅表示相关强度。'
            '维度之间相关性过高意味着信息重叠，健康分权重与分群阈值应结合业务复核。</div>')
    return section("维度相关性", body + note, "correlation", "Pearson 相关系数（评分用户）")


def render_appendix(data) -> str:
    ov = data["overview"]
    stats = data["feature_stats"]
    rows = []
    for _, r in stats.iterrows():
        rows.append([
            esc(r["key"]), esc(r["label"]), esc(r.get("dimension_label", "")),
            esc(r.get("module_label", "")), esc(r.get("definition", "")),
            esc(r.get("direction_label", "")),
            "是" if bool(r.get("in_score")) else "否",
            f'{r["coverage_txn_pct"]:.1f}%',
            _fmt(r["p50"], r["fmt"]),
        ])
    feat_tbl = table(["键", "指标", "维度", "模块", "定义", "方向", "评分", "流水覆盖率", "中位数"],
                     rows, cls="data-table appendix-table",
                     aligns=["left", "left", "left", "left", "left", "center", "center", "right", "right"])
    th = ov.get("thresholds", {})
    seg_th = th.get("segment", {})
    flag_th = th.get("flag", {})
    seg_rows = [[esc(THRESHOLD_LABELS.get(k, k)), esc(v)] for k, v in seg_th.items()]
    flag_rows = [[esc(THRESHOLD_LABELS.get(k, k)), esc(v)] for k, v in flag_th.items()]
    th_html = (
        '<div class="two-col">'
        f'<div class="card"><h3>分群阈值</h3>{table(["参数", "取值"], seg_rows, cls="data-table")}</div>'
        f'<div class="card"><h3>风险标签阈值</h3>{table(["参数", "取值"], flag_rows, cls="data-table")}</div>'
        '</div>'
    )
    meta = data["run_meta"]
    meta_rows = [
        ["输入文件", esc(meta.get("input_path", ""))],
        ["输出目录", esc(meta.get("output_dir", ""))],
        ["分块数", f'{int(meta.get("chunks", 0)):,}'],
        ["第一遍耗时", f'{meta.get("timing_sec", {}).get("pass1", 0):.2f}s'],
        ["第二遍耗时", f'{meta.get("timing_sec", {}).get("pass2", 0):.2f}s'],
        ["分析总耗时", f'{meta.get("timing_sec", {}).get("total_before_report", 0):.2f}s'],
        ["分析版本", esc(meta.get("version", ""))],
    ]
    meta_tbl = table(["项", "值"], meta_rows, cls="data-table")
    outputs = ov.get("outputs", [])
    out_html = "".join(f"<code>{esc(o)}</code>" for o in outputs) or "—"
    body = f'''
    <div class="card"><h3>分析配置与耗时</h3>{meta_tbl}</div>
    {th_html}
    <div class="card"><h3>中间结果文件</h3><div class="file-list">{out_html}</div></div>
    <details class="card"><summary><h3 style="display:inline">特征字典（{len(stats)} 项）</h3></summary>
      {feat_tbl}</details>'''
    return section("附录", body, "appendix", "配置、阈值与完整指标口径")


# ---------------------------------------------------------------------------
# 样式与页面组装
# ---------------------------------------------------------------------------

CSS = """
:root{
  --bg:#f5f7fb; --card:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --brand:#2f6fed; --good:#0f9d76; --warn:#d97706; --bad:#dc2626;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,-apple-system,sans-serif;
  line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1200px;margin:0 auto;padding:0 26px}
.hero{background:linear-gradient(135deg,#08172d 0%,#12335f 55%,#1b4c8a 100%);color:#fff;
  padding:46px 0 34px}
.eyebrow{font-size:12px;letter-spacing:.18em;color:#8fb3e0;font-weight:600}
.hero h1{margin:8px 0 10px;font-size:34px;line-height:1.25;font-weight:750}
.hero-sub{margin:0;color:#c7d8ee;max-width:820px;font-size:15px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.chip{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.16);
  color:#dce9f8;border-radius:999px;padding:5px 12px;font-size:12px}
.toc{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.94);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.toc-inner{display:flex;gap:4px;overflow-x:auto;padding:9px 26px}
.toc a{color:#475569;text-decoration:none;font-size:13px;white-space:nowrap;
  padding:5px 10px;border-radius:7px}
.toc a:hover{background:#eef3fb;color:var(--brand)}
section{padding:34px 0 6px}
.section-head{margin-bottom:16px}
h2{font-size:23px;margin:0;letter-spacing:-.01em}
.section-sub{margin:4px 0 0;color:var(--muted);font-size:13.5px}
h3{font-size:16px;margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(15,23,42,.04);min-width:0}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:16px}
.kpi-grid.small{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:15px 17px;
  position:relative;overflow:hidden}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:#cbd5e1}
.kpi.accent:before{background:var(--brand)} .kpi.danger:before{background:var(--bad)}
.kpi.warn:before{background:var(--warn)} .kpi.good:before{background:var(--good)}
.kpi-label{font-size:12px;color:var(--muted);font-weight:600}
.kpi-value{font-size:27px;font-weight:750;letter-spacing:-.02em;margin:3px 0 1px}
.kpi-note{font-size:11.5px;color:#94a3b8;min-height:17px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.two-col.wide{grid-template-columns:1fr 1.25fr}
@media(max-width:900px){.two-col,.two-col.wide{grid-template-columns:1fr}}
.finding-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-bottom:16px}
.finding{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:15px 17px}
.finding-title{font-size:12px;color:var(--brand);font-weight:700;letter-spacing:.04em;margin-bottom:4px}
.finding p{margin:0;font-size:13.5px;color:#334155}
.recs,.caveats{margin:0;padding-left:20px;font-size:13.5px;color:#334155}
.recs li,.caveats li{margin-bottom:7px}
.warn-card{border-color:#fde3b8;background:#fffcf5}
.callout{background:#f0f5ff;border:1px solid #d8e4fb;border-radius:12px;
  padding:13px 16px;font-size:13.5px;color:#274264;margin:0 0 16px}
.callout.warn{background:#fff8ec;border-color:#fde3b8;color:#7a4b0d}
.callout code{background:#e2e8f0;padding:1px 5px;border-radius:4px;font-size:12px}
.table-wrap{overflow-x:auto;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 11px;border-bottom:1px solid #edf1f7;vertical-align:top}
thead th{background:#f8fafc;color:#475569;font-size:12px;font-weight:650;white-space:nowrap}
tbody tr:hover{background:#fafcff}
td{color:#334155}
.data-table td:first-child{font-weight:600;color:#1e293b}
.matrix{font-size:12.5px}
.matrix td,.matrix th{padding:8px 9px}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:650;white-space:nowrap}
.pill.good{background:#e5f7ef;color:#0b7a59} .pill.warn{background:#fdf0d9;color:#96570a}
.pill.info{background:#eaf1fd;color:#2454a6}
.tag{display:inline-block;padding:1px 8px;border-radius:6px;font-size:11px;font-weight:650}
.tag.risk{background:#fee9e9;color:#a51c1c} .tag.obs{background:#eef2f7;color:#546274}
.empty{color:#94a3b8;font-size:13px;margin:8px 0}
.muted{color:var(--muted);font-size:12.5px;margin:0 0 10px}
svg{max-width:100%;height:auto;display:block}
.segment-top{display:grid;grid-template-columns:340px 1fr;gap:16px;align-items:start}
@media(max-width:900px){.segment-top{grid-template-columns:1fr}}
.donut-card{display:flex;flex-direction:column;align-items:center}
.legend{width:100%;margin-top:4px}
.legend-row{display:flex;align-items:center;gap:8px;font-size:12.5px;padding:3px 0}
.dot{display:inline-block;width:9px;height:9px;border-radius:3px;flex:none}
.legend-label{flex:1;color:#334155}
.legend-value{color:#64748b;font-variant-numeric:tabular-nums}
.seg-notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}
.seg-note{border-left:4px solid #94a3b8;background:#fafbfe;border-radius:0 8px 8px 0;padding:10px 13px}
.seg-note-title{font-weight:700;font-size:13px;margin-bottom:2px}
.seg-note p{margin:0;font-size:12.5px;color:#475569}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;margin-bottom:16px}
.chart-card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:12px 14px 9px}
.chart-title{font-size:13.5px;font-weight:700;margin-bottom:2px}
.chart-title span{font-weight:400;color:#94a3b8;font-size:11px;margin-left:6px}
.chart-meta{font-size:11px;color:#94a3b8;padding-top:4px;border-top:1px dashed #eef2f7}
.heatmap{font-size:12.5px;text-align:center}
.heatmap td,.heatmap th{padding:8px 10px;border:1px solid #fff}
.heatmap .row-head{text-align:left;background:#f8fafc;font-weight:650}
.file-list{display:flex;flex-wrap:wrap;gap:7px}
.file-list code{background:#f1f5f9;border:1px solid #e2e8f0;padding:3px 8px;border-radius:6px;font-size:11.5px}
details.card summary{cursor:pointer;list-style:none;margin:-2px 0 10px}
details.card summary::-webkit-details-marker{display:none}
.appendix-table{font-size:11.8px}
footer{color:#94a3b8;font-size:12px;padding:30px 0 40px;text-align:center}
@media print{
  .toc{position:static} body{background:#fff}
  .card,.kpi,.finding{box-shadow:none;break-inside:avoid}
}
"""


def render_footer(ov) -> str:
    return (
        '<footer><div class="wrap">'
        f'交易行为用户画像分析报告 · v{esc(VERSION)} · 数据截至 {esc(ov.get("period_text", "—"))} · '        f'生成于 {esc(ov.get("generated_at", "—"))}<br>'
        '本报告由 profile_analysis.py 的中间结果自动生成，未重新计算任何统计量。'
        '</div></footer>'
    )


def build_html(data, title: str | None = None) -> str:
    ov = data["overview"]
    page_title = title or data.get("run_meta", {}).get("config", {}).get("report_title") or DEFAULT_CONFIG["report_title"]
    body = "".join([
        render_header(ov, page_title),
        '<main class="wrap">',
        render_summary(data),
        render_findings(data),
        render_data(data),
        render_segments(data),
        render_income(data),
        render_lender(data),
        render_dishonour(data),
        render_observations(data),
        render_dimensions(data),
        render_features(data),
        render_risk(data),
        render_consumption(data),
        render_target(data),
        render_correlation(data),
        render_appendix(data),
        '</main>',
        render_footer(ov),
    ])
    return (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{esc(page_title)}</title>'
        f'<style>{CSS}</style></head><body>{body}</body></html>'
    )


# ---------------------------------------------------------------------------
# 命令行与主流程
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="用户画像 HTML 报告生成（统一入口，自动调用分析脚本）")
    p.add_argument("--config", help="JSON 配置文件路径")
    p.add_argument("--input", help="输入特征 CSV")
    p.add_argument("--dict", dest="dict_path", help="dict.csv 数据字典路径")
    p.add_argument("--output-dir", help="中间结果与报告输出目录")
    p.add_argument("--cache-dir", help="子集缓存目录")
    p.add_argument("--chunk-size", type=int, help="分块行数")
    p.add_argument("--reservoir-size", type=int, help="每特征分位抽样上限")
    p.add_argument("--target", dest="target_column", help="目标标签列")
    p.add_argument("--seed", type=int, help="随机种子")
    p.add_argument("--subset-cache", choices=["auto", "on", "off"], help="窄表缓存策略")
    p.add_argument("--no-user-export", action="store_true", help="不导出用户级评分明细")
    p.add_argument("--allow-missing-features", action="store_true", help="缺失特征时跳过而不是报错")
    p.add_argument("--skip-analysis", action="store_true", help="复用已有中间结果，不重新分析")
    p.add_argument("--report-name", help="报告文件名（默认取配置）")
    p.add_argument("--open", action="store_true", help="生成后用默认浏览器打开")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {
        "input_path": args.input,
        "dict_path": args.dict_path,
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
        "chunk_size": args.chunk_size,
        "reservoir_size": args.reservoir_size,
        "target_column": args.target_column,
        "seed": args.seed,
        "subset_cache": args.subset_cache,
    }
    if args.no_user_export:
        overrides["export_user_scores"] = False
    if args.allow_missing_features:
        overrides["allow_missing_features"] = True

    cfg = load_config(args.config, overrides)
    t0 = time.time()
    if not args.skip_analysis:
        run_analysis(cfg)
    else:
        print("跳过画像分析，复用已有中间结果。", flush=True)

    data = load_intermediates(cfg["output_dir"])
    report_name = args.report_name or cfg.get("report_name") or "profiling_report.html"
    report_path = Path(cfg["output_dir"]) / report_name
    html_text = build_html(data, cfg.get("report_title") if args.config or not args.skip_analysis else None)
    report_path.write_text(html_text, encoding="utf-8")
    elapsed = time.time() - t0
    print(f"报告已生成: {report_path}（{report_path.stat().st_size / 1024:,.0f} KB，用时 {elapsed:.1f}s）", flush=True)
    if args.open:
        webbrowser.open(report_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
