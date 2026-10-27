#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Swap In vs Swap Out 置换画像（临时脚本）。

独立放在 analysis/swap_compare/，不修改现有产品配置与分析引擎：
1. 合并两份 v1_2 特征文件，加 cohort 列；
2. 用 products/fundo.json 的同一套指标/阈值对合并样本统一评分；
3. 按 cohort 聚合 user_scores 与原始金额特征，围绕“置入了谁、为什么可能置入、
   置出了谁、为什么可能置出”生成自包含 HTML。

运行：python analysis/swap_compare/run_swap_comparison.py
"""
from __future__ import annotations

import html
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
MERGED = BASE / "merged_input.csv"
OUT_DIR = BASE / "out"
CACHE_DIR = BASE / "cache"
REPORT = BASE / "swap_in_vs_swap_out_comparison.html"
CSS_SOURCE = ROOT / "fundo_vs_wagego_comparison.html"
IN_PATH = ROOT / "data" / "existing_swap_in_v1_2_with_feature_all.csv"
OUT_PATH = ROOT / "data" / "existing_swap_out_v1_2_with_feature_all.csv"

sys.path.insert(0, str(ROOT))
from profile_analysis import (  # noqa: E402  (只读复用现有清洗口径)
    UNRESOLVED_AMT_COLUMNS,
    RESOLVED_AMT_COLUMNS,
    clean_source_numeric,
)

KEYS = ("swap_in", "swap_out")
SEGMENTS = ["优质均衡", "稳健工薪", "均衡普通", "月光消费", "现金流紧张", "高风险承压"]
ARCHETYPES = ["稳定工资", "不稳定工资", "多源/非工资", "福利依赖"]
CROSS_GROUPS = ["稳定工资", "不稳定工资", "多源/非工资"]
LENDER_BEHAVIORS = ["净偿还", "借还平衡", "净融资扩张", "仅新增借款", "仅偿还", "近28天无借贷", "无借贷记录"]
DISHONOUR_STATUS = ["无拒付", "已全部解决", "部分解决", "未解决"]
BURN_BANDS = ["支出低于收入80%", "支出接近收入", "支出达到或超过收入", "数据不足"]
BUFFER_BANDS = ["不足7天", "7至30天", "30至90天", "至少90天"]
DIM_LABELS = {
    "income": "收入稳定性",
    "cashflow": "收支储蓄",
    "liquidity": "流动性缓冲",
    "debt": "负债压力",
    "engagement": "消费活跃度",
}
FLAG_LABELS = {
    "flag_unresolved_dishonour": "存在未结清拒付",
    "flag_high_debt": "负债压力高",
    "flag_multi_lender": "多头借贷（名单内）",
    "flag_payday": "现金贷/工资贷使用",
    "flag_negative_cashflow": "现金净流出",
    "flag_low_buffer": "流动性缓冲不足",
    "flag_gambling": "赌博类支出",
    "flag_income_unstable": "收入不稳定",
    "flag_fees_heavy": "费用/透支负担",
    "flag_welfare_dependent": "福利收入依赖",
    "flag_no_wage": "无工资入账",
}
RAW_COLS = {
    "inc_global_sum": "bank_txn_income_global_sum_182d",
    "inc_wage_sum": "bank_txn_income_wages_sum_182d",
    "cf_expense_sum": "bank_txn_expense_global_sum_182d",
    "cf_surplus_sum": "bank_txn_surplus_surplus_sum_182d",
    "db_disburse_sum": "bank_txn_lender_disburse_amount_sum",
    "db_disburse_count": "bank_txn_lender_disburse_amount_count",
    "db_cash_advance_sum": "bank_txn_lender_disburse_advance_amount_sum",
    "db_repay_sum": "bank_txn_lender_repay_amount_sum",
    "lq_balance_median": "bank_txn_worthiness_balance_median_avg_acct_182d",
    "lq_negative_share": "bank_txn_worthiness_balance_negative_share_avg_acct_182d",
    "lq_reserve_months": "bank_txn_worthiness_reserve_months_182d",
    "cf_saving_rate": "bank_txn_balance_global_savings_rate_182d",
    "en_active_days": "bank_txn_category_active_days_182d",
    "inc_wage_share": "bank_txn_worthiness_wage_income_amt_share_in_income_tag_182d",
    "db_dishonour_ratio": "bank_txn_dishonour_lender_ratio_182d",
}


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def ensure_merged() -> None:
    if MERGED.exists() and MERGED.stat().st_mtime >= max(IN_PATH.stat().st_mtime, OUT_PATH.stat().st_mtime):
        return
    print("merging cohorts ...", flush=True)
    frames = {}
    for key, path in zip(KEYS, (IN_PATH, OUT_PATH)):
        df = pd.read_csv(path, dtype={"user_id": "string", "application_id": "string"})
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
        frames[key] = df
    if list(frames[KEYS[0]].columns) != list(frames[KEYS[1]].columns):
        raise RuntimeError("两份样本的列顺序不一致，无法直接合并")
    parts = []
    for key in KEYS:
        df = frames[key]
        df.insert(0, "cohort", key)
        parts.append(df)
    pd.concat(parts, ignore_index=True).to_csv(MERGED, index=False)


def ensure_analysis() -> None:
    user_scores = OUT_DIR / "user_scores.csv"
    if user_scores.exists() and user_scores.stat().st_mtime >= MERGED.stat().st_mtime:
        return
    print("running profiling engine on merged sample ...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-I", str(ROOT / "profile_analysis.py"),
            "--config", str(ROOT / "products" / "fundo.json"),
            "--input", str(MERGED),
            "--output-dir", str(OUT_DIR),
            "--cache-dir", str(CACHE_DIR),
        ],
        check=True,
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(
        MERGED,
        usecols=["cohort", "user_id", "application_id", "sample_datetime", "duedate_1m_5"],
        dtype={"user_id": "string", "application_id": "string"},
    )
    scores = pd.read_csv(OUT_DIR / "user_scores.csv", dtype={"user_id": "string", "application_id": "string"})
    cohort_map = dict(zip(base.application_id, base.cohort))
    scores["cohort"] = scores.application_id.map(cohort_map)
    if scores.cohort.isna().any():
        raise RuntimeError("user_scores 中存在无法匹配 cohort 的 application_id")

    raw_cols = sorted(set(RAW_COLS.values()) | set(UNRESOLVED_AMT_COLUMNS) | set(RESOLVED_AMT_COLUMNS))
    raw = pd.read_csv(MERGED, usecols=["cohort", "application_id"] + raw_cols,
                      dtype={"application_id": "string"})
    unresolved = np.vstack([clean_source_numeric(raw[c], c) for c in UNRESOLVED_AMT_COLUMNS])
    resolved = np.vstack([clean_source_numeric(raw[c], c) for c in RESOLVED_AMT_COLUMNS])
    raw["db_unresolved_amt"] = np.sum(unresolved, axis=0)
    raw["db_resolved_amt"] = np.sum(resolved, axis=0)
    return base, scores, raw


# ---------------------------------------------------------------------------
# 统计与格式
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def rate(series: pd.Series) -> tuple[float, tuple[float, float]]:
    s = series.dropna()
    k, n = int(s.sum()), len(s)
    return (k / n * 100 if n else float("nan")), wilson(k, n)


def median(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.median()) if len(s) else float("nan")


def share(series: pd.Series, value) -> float:
    s = series.dropna()
    return float(s.eq(value).mean() * 100) if len(s) else float("nan")


def raw_median(raw: pd.DataFrame, key: str, cohort: str) -> float:
    col = RAW_COLS.get(key, key)
    arr = clean_source_numeric(raw.loc[raw.cohort.eq(cohort), col], col)
    s = pd.Series(arr).dropna()
    return float(s.median()) if len(s) else float("nan")


def fmt(v: float, kind: str) -> str:
    if not math.isfinite(v):
        return "—"
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "pct2":
        return f"{v:.2f}%"
    if kind == "money":
        return f"A${v:,.0f}"
    if kind == "k":
        return f"A${v / 1000:,.1f}k"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "two":
        return f"{v:.2f}"
    if kind == "days":
        return f"{v:.1f} 天"
    if kind == "months":
        return f"{v:.1f} 个月"
    return f"{v:.1f}"


def esc(text) -> str:
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# 图表组件（复用 Fundo × WageGo 对比页的视觉语言）
# ---------------------------------------------------------------------------

def section(key: str, number: str, title: str, subtitle: str, content: str) -> str:
    return f'''<section id="{key}" class="section"><div class="wrap">
    <div class="section-head"><div><div class="eyebrow">{number}</div><h2>{title}</h2></div>
    <p class="section-sub">{subtitle}</p></div>{content}</div></section>'''


def chart_value(v: float, kind: str) -> str:
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "score":
        return f"{v:.1f}"
    return fmt(v, kind)


def chart_axis(v: float, kind: str) -> str:
    if kind == "pct":
        return f"{v:g}%"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "score":
        return f"{v:g}"
    return fmt(v, kind)


def chart_delta(d: float, kind: str, diff_unit: str = "") -> str:
    if kind == "pct":
        return f"{d:+.1f}pp"
    if kind == "int":
        return f"{d:+,.0f}{diff_unit}"
    if kind == "score":
        return f"{d:+.1f}{diff_unit or ' 分'}"
    return f"{d:+.1f}{diff_unit}"


def paired(title: str, rows, caption: str = "统一 0–100% 刻度", maximum: float = 100,
           kind: str = "pct", note: str = "", diff_unit: str = "",
           axis_suffix: str | None = None) -> str:
    rows = list(rows)
    if axis_suffix is None:
        axis_suffix = "%" if kind == "pct" else ""
    out = (f'<div class="compare pd-chart"><div class="compare-head"><h3>{title}</h3>'
           f'<span>{caption}</span></div>'
           '<div class="pd-legend"><span><i class="pd-key in"></i>Swap In</span>'
           '<span><i class="pd-key out"></i>Swap Out</span></div>')
    out += (f'<div class="pd-axis-row"><span></span><div class="pd-axis">'
            f'<span>0{axis_suffix}</span><span>{chart_axis(maximum / 2, kind)}</span>'
            f'<span>{chart_axis(maximum, kind)}</span></div><span></span></div>')
    for label, values, detail in rows:
        a, b = float(values[0]), float(values[1])
        pa = max(0.0, min(a / maximum * 100, 100.0))
        pb = max(0.0, min(b / maximum * 100, 100.0))
        left, width = min(pa, pb), abs(pb - pa)
        va, vb = chart_value(a, kind), chart_value(b, kind)
        d = b - a
        detail_html = f'<small>{esc(detail)}</small>' if detail else ""
        out += f'''<div class="pd-row"><div class="compare-label">{esc(label)}{detail_html}</div>
        <div class="pd-track" role="img" aria-label="{esc(label)}：Swap In {va}，Swap Out {vb}，差值 {chart_delta(d, kind, diff_unit)}">
        <span class="pd-mid"></span><span class="pd-link" style="left:{left:.4f}%;width:{width:.4f}%"></span>
        <span class="pd-point in" style="left:{pa:.4f}%" title="Swap In：{va}"></span>
        <span class="pd-point out" style="left:{pb:.4f}%" title="Swap Out：{vb}"></span></div>
        <div class="pd-values"><span class="in">IN {va}</span><span class="out">OUT {vb}</span><b>{chart_delta(d, kind, diff_unit)}</b></div></div>'''
    if not note:
        note = "IN = Swap In（圆点）；OUT = Swap Out（方点）。差值 = Swap Out − Swap In。"
    out += f'<p class="table-note">{note}</p>'
    body = "".join(
        f'<tr><td>{esc(label)}</td><td>{chart_value(float(values[0]), kind)}</td>'
        f'<td>{chart_value(float(values[1]), kind)}</td>'
        f'<td>{chart_delta(float(values[1]) - float(values[0]), kind, diff_unit)}</td></tr>'
        for label, values, _detail in rows
    )
    out += ('<details class="pd-data"><summary>查看数据表</summary>'
            '<table><thead><tr><th>指标</th><th>Swap In</th><th>Swap Out</th><th>差值</th>'
            f'</tr></thead><tbody>{body}</tbody></table></details>')
    return out + "</div>"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ensure_merged()
    ensure_analysis()
    base, scores, raw = load_data()
    obs = scores[scores.flag_no_txn.eq(0)].copy()
    scored = obs[obs.health_score.notna()].copy()

    S = {}
    for key in KEYS:
        b = base[base.cohort.eq(key)]
        g = obs[obs.cohort.eq(key)]
        gs = scored[scored.cohort.eq(key)]
        S[key] = {
            "apps": len(b),
            "users": int(b.user_id.nunique()),
            "obs": len(g),
            "scored": len(gs),
            "period": (pd.to_datetime(b.sample_datetime).min(), pd.to_datetime(b.sample_datetime).max()),
            "health": (float(gs.health_score.quantile(.25)), float(gs.health_score.median()),
                       float(gs.health_score.quantile(.75))),
            "health_median": median(gs.health_score),
            "segments": {s: share(gs.segment, s) for s in SEGMENTS},
            "dims": {d: median(gs[f"score_{d}"]) for d in DIM_LABELS},
            "archetypes": {s: share(g.income_archetype, s) for s in ARCHETYPES},
            "lender_behaviors": {s: share(g.lender_behavior, s) for s in LENDER_BEHAVIORS},
            "dishonour_status": {s: share(g.dishonour_status, s) for s in DISHONOUR_STATUS},
            "burn_bands": {s: share(g.tr_recent_burn_band, s) for s in BURN_BANDS},
            "buffer_bands": {s: share(g.rs_expense_buffer_days_band, s) for s in BUFFER_BANDS},
            "flags": {f: rate(g[f]) for f in FLAG_LABELS},
            "flag_count_mean": float(g.risk_flag_count.mean()),
            "flag_count_ge3": float(g.risk_flag_count.ge(3).mean() * 100),
            "joint": None,
            "joint_n": 0,
            "income_decline": share(g.tr_income_daily_ratio_band, "下降超过20%"),
            "expense_rise": share(g.tr_expense_daily_ratio_band, "上升至少20%"),
            "commit_ge30": share(g.rs_commitment_income_band, "30%至60%") + share(g.rs_commitment_income_band, "至少60%"),
            "fixed_ge20": share(g.rs_fixed_income_band, "20%至40%") + share(g.rs_fixed_income_band, "至少40%"),
            "disburse_rate": float(g.db_disburse_count.gt(0).mean() * 100),
            "active_lender_ge1": float(g.db_active_lender_cnt.ge(1).mean() * 100),
            "median": {
                "balance": median(g.lq_balance_median),
                "negative_share": raw_median(raw, "lq_negative_share", key) * 100,
                "saving_rate": median(g.cf_saving_rate) * 100,
                "unresolved_cnt": median(g.db_unresolved_cnt),
                "disburse_cnt": median(g.db_disburse_count),
                "active_lender": median(g.db_active_lender_cnt),
                "buffer_days": median(g.rs_expense_buffer_days),
                "tr_burn": median(g.tr_recent_burn) * 100,
                "active_days": median(g.en_active_days),
                "wage_share": median(g.inc_wage_share) * 100,
                "discretionary": median(g.cs_discretionary_share) * 100,
            },
            "raw": {
                "inc_global_sum": raw_median(raw, "inc_global_sum", key),
                "inc_wage_sum": raw_median(raw, "inc_wage_sum", key),
                "cf_expense_sum": raw_median(raw, "cf_expense_sum", key),
                "cf_surplus_sum": raw_median(raw, "cf_surplus_sum", key),
                "db_disburse_sum": raw_median(raw, "db_disburse_sum", key),
                "db_cash_advance_sum": raw_median(raw, "db_cash_advance_sum", key),
                "db_repay_sum": raw_median(raw, "db_repay_sum", key),
                "lq_reserve_months": raw_median(raw, "lq_reserve_months", key),
                "db_dishonour_ratio": raw_median(raw, "db_dishonour_ratio", key),
            },
        }
        r = raw[raw.cohort.eq(key)]
        S[key]["raw"]["unresolved_amt_p50"] = float(r.db_unresolved_amt.median())
        S[key]["raw"]["unresolved_amt_mean"] = float(r.db_unresolved_amt.mean())
        affected = r.loc[r.db_unresolved_amt.gt(0), "db_unresolved_amt"]
        S[key]["raw"]["unresolved_amt_affected_mean"] = float(affected.mean()) if len(affected) else float("nan")
        S[key]["raw"]["unresolved_amt_affected_n"] = int(len(affected))
        denom = r.db_unresolved_amt.sum() + r.db_resolved_amt.sum()
        S[key]["raw"]["resolved_amt_ratio"] = (
            float(r.db_resolved_amt.sum() / denom * 100) if denom else float("nan"))
        joint = g[["flag_payday", "flag_unresolved_dishonour", "flag_low_buffer"]].eq(1).all(axis=1)
        S[key]["joint"] = float(joint.mean() * 100)
        S[key]["joint_n"] = int(joint.sum())
        S[key]["joint_rate"] = rate(joint.astype(int))

    # ---- 文本中反复使用的量 ----
    def v(key, path):
        cur = S[key]
        for part in path.split("."):
            cur = cur[int(part)] if part.isdigit() else cur[part]
        return cur

    def pv(path):
        return [v(k, path) for k in KEYS]

    def pct_pair(path, kind="pct"):
        return " / ".join(fmt(x, kind) for x in pv(path))

    def inout(path, kind="pct"):
        a, b = pv(path)
        return f"置入 {fmt(a, kind)} / 置出 {fmt(b, kind)}"

    def pp(path):
        a, b = pv(path)
        return f"{b - a:+.1f}pp"

    # 收入形态内分层与标准化（福利依赖组 Swap Out 为 0 条，不纳入）
    cross = {}
    for flag in ("flag_payday", "flag_unresolved_dishonour"):
        cross[flag] = {}
        for grp in CROSS_GROUPS:
            vals = []
            for key in KEYS:
                g = obs[obs.cohort.eq(key) & obs.income_archetype.eq(grp)]
                vals.append((rate(g[flag])[0], len(g)))
            cross[flag][grp] = vals
    total_cross = sum(len(obs[obs.income_archetype.eq(g)]) for g in CROSS_GROUPS)
    cross_w = {g: len(obs[obs.income_archetype.eq(g)]) / total_cross for g in CROSS_GROUPS}
    cross_std = {
        flag: [sum(cross_w[g] * cross[flag][g][i][0] for g in CROSS_GROUPS) for i in range(2)]
        for flag in cross
    }

    # ---- 报告 ----
    css = CSS_SOURCE.read_text(encoding="utf-8").split("<style>", 1)[1].split("</style>", 1)[0]
    css += """
/* Swap In / Swap Out overrides */
:root{--fundo:#0f766e;--fundo-dark:#115e59;--fundo-soft:#ccfbf1;--series-in:#0d9488;--wagego:#b45309;--wagego-dark:#92400e;--wagego-soft:#fef3c7}
.flag-table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}
.flag-table th,.flag-table td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
.flag-table th:first-child,.flag-table td:first-child{text-align:left}
.flag-table th{font-size:11.5px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.03em}
.flag-table td.ci{color:var(--muted);font-size:11.5px}
.flag-table td.d{font-weight:700}
.flag-table tr:hover td{background:#f8fafc}
.logic-table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}
.logic-table th,.logic-table td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.logic-table th{font-size:11.5px;color:var(--muted);font-weight:700;letter-spacing:.03em}
.logic-table td:first-child{font-weight:700;white-space:nowrap}
.logic-table td:nth-child(2),.logic-table td:nth-child(3){font-variant-numeric:tabular-nums}
@media(max-width:640px){.logic-table{font-size:12px}.logic-table th,.logic-table td{padding:7px 6px}.logic-table td:first-child{white-space:normal}}
.cohort-pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700;margin-right:6px}
.cohort-pill.in{background:var(--fundo-soft);color:var(--fundo-dark)}
.cohort-pill.out{background:var(--wagego-soft);color:var(--wagego-dark)}
@media(max-width:640px){.flag-table{font-size:12px}.flag-table th,.flag-table td{padding:7px 5px}}
/* Same-row paired comparison */
.pd-chart .compare-head{margin-bottom:10px}
.pd-legend{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;font-size:11.5px;color:var(--muted);margin:2px 0 8px}
.pd-legend span{display:inline-flex;align-items:center;gap:6px}
.pd-key{display:inline-block;width:10px;height:10px;border:2px solid #fff;box-shadow:0 0 0 1px currentColor}
.pd-key.in{background:var(--series-in);color:var(--series-in);border-radius:50%}
.pd-key.out{background:var(--wagego);color:var(--wagego);border-radius:2px}
.pd-axis-row{display:grid;grid-template-columns:190px 1fr 184px;gap:18px;align-items:center;margin:8px 0 2px}
.pd-axis{display:flex;justify-content:space-between;padding:0 7px;font-size:11px;color:var(--muted)}
.pd-row{display:grid;grid-template-columns:190px 1fr 184px;gap:18px;align-items:center;padding:13px 0;border-bottom:1px dashed var(--line)}
.pd-row:last-of-type{border-bottom:0}
.pd-track{position:relative;height:34px;margin:0 7px}
.pd-track:before{content:"";position:absolute;left:0;right:0;top:16px;height:2px;background:#e2e8f0}
.pd-mid{position:absolute;left:50%;top:7px;bottom:7px;width:1px;background:#dbe3ec}
.pd-link{position:absolute;top:15px;height:4px;border-radius:999px;background:#94a3b8}
.pd-point{position:absolute;z-index:1;width:11px;height:11px;transform:translateX(-50%);border:2px solid #fff;box-shadow:0 0 0 1px currentColor}
.pd-point.in{top:5px;background:var(--series-in);color:var(--series-in);border-radius:50%}
.pd-point.out{top:18px;background:var(--wagego);color:var(--wagego);border-radius:2px}
.pd-values{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:2px 10px;font-size:11.5px;line-height:1.5;font-variant-numeric:tabular-nums;color:#334155}
.pd-values .in,.pd-values .out{color:#334155}
.pd-values b{min-width:58px;text-align:right;color:#475569}
.pd-range{position:relative;height:36px;margin:0 7px}
.pd-range-lane{position:absolute;left:0;right:0;height:14px;border-radius:999px;background:#f1f5f9}
.pd-range-lane .range-iqr{top:3px;height:8px;border-radius:999px}
.pd-range-lane .range-median{top:0;height:14px;width:3px}
.pd-data{margin-top:12px;padding-top:10px;border-top:1px dashed var(--line)}
.pd-data summary{cursor:pointer;font-size:11.5px;font-weight:700;color:var(--muted)}
.pd-data table{width:100%;margin-top:8px;border-collapse:collapse;font-size:11.5px;font-variant-numeric:tabular-nums}
.pd-data th,.pd-data td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
.pd-data th:first-child,.pd-data td:first-child{text-align:left}
.pd-data th{font-size:10.5px;color:var(--muted);font-weight:700}
@media(max-width:960px){.pd-axis-row,.pd-row{grid-template-columns:155px 1fr 160px;gap:12px}}
@media(max-width:640px){
.pd-axis-row{grid-template-columns:1fr;gap:0}
.pd-axis-row>span{display:none}
.pd-axis{margin-top:8px}
.pd-row{grid-template-columns:1fr;gap:6px}
.pd-row .compare-label{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
.pd-row .compare-label small{margin:0}
.pd-values{justify-content:flex-start}
.pd-values b{margin-left:auto}
.pd-data{font-size:11px}
}
/* Editorial summary */
:root{--shadow:0 1px 2px rgba(15,23,42,.03)}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1b2430 100%);padding:46px 0 30px}
.hero::after{display:none}
.hero h1{font-size:36px;margin:10px 0 14px}
.hero-lede{font-size:15px;max-width:920px}
.hero-grid{margin-top:24px}
.hero-stat{background:rgba(255,255,255,.055);border-color:rgba(255,255,255,.12);backdrop-filter:none}
.hero-stat .value .f{color:#5eead4}
.hero-stat .value .w{color:#fcd34d}
.hero-meta .dot.f{background:#5eead4}
.hero-meta .dot.w{background:#fcd34d}
.hero-note{max-width:920px}
.section{padding:44px 0 6px}
.section-head{padding-top:24px;margin-bottom:20px}
.section-head h2{font-size:26px}
.verdict{background:#fff;border:1px solid var(--line);border-left:4px solid var(--fundo);border-radius:12px;padding:22px 24px;margin:0 0 16px}
.verdict-label{font-size:10.5px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--fundo-dark);margin-bottom:8px}
.verdict p{font-size:20px;line-height:1.6;letter-spacing:-.01em;margin:0;color:var(--ink)}
.verdict strong{font-weight:800}
.summary-cards .card{padding:20px 22px}
.summary-cards .card h3{display:flex;align-items:center;gap:8px}
.summary-table{margin:0 0 16px;border-top:1px solid var(--line);font-size:13px}
.summary-table>div{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:12px;padding:8px 0;border-bottom:1px solid var(--line)}
.summary-table>div>span:nth-child(2),.summary-table>div>span:nth-child(3){text-align:right;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--ink)}
.summary-table .summary-head{font-size:10.5px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em;padding:6px 0}
.summary-table .summary-head span{font-weight:600;color:var(--faint)}
.summary-cards .card p:last-child{font-size:13px;color:var(--body);line-height:1.7}
.takeaway-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:16px}
.takeaway{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;min-width:0}
.takeaway .tag{font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin-bottom:6px}
.takeaway h3{font-size:14.5px;line-height:1.4;margin:0 0 6px}
.takeaway p{font-size:13px;color:var(--body);line-height:1.7;margin:0}
.reasons{display:grid;gap:0;margin:18px 0 12px;border-top:1px solid var(--line)}
.reason{display:grid;grid-template-columns:28px 1fr;gap:10px;padding:14px 0;border-bottom:1px solid var(--line)}
.reason .num{font-size:11px;font-weight:800;color:var(--faint);padding-top:2px}
.reason h3{font-size:14.5px;line-height:1.4;margin:0 0 4px}
.reason p{font-size:13.3px;line-height:1.7;color:var(--body);margin:0}
.reason p b{color:var(--ink);font-variant-numeric:tabular-nums}
.notes{border-left:2px solid var(--line-strong);padding:2px 0 2px 14px;margin:16px 0 0;font-size:12.5px;line-height:1.75;color:var(--muted)}
.notes p{margin:0 0 7px}
.notes p:last-child{margin-bottom:0}
.notes strong{color:#475569}
.boundary{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-top:16px}
.boundary h3{font-size:14px;margin:0 0 8px}
.boundary ul{margin:0;padding-left:18px;font-size:13px;line-height:1.8;color:var(--body)}
.boundary li{margin-bottom:5px}
.boundary li:last-child{margin-bottom:0}
.boundary b{color:var(--ink)}
@media(max-width:960px){
.hero h1{font-size:32px}
.takeaway-grid{grid-template-columns:1fr}
}
@media(max-width:640px){
.hero{padding:38px 0 26px}
.hero h1{font-size:28px}
.hero-lede{font-size:14px}
.hero-grid{grid-template-columns:repeat(2,1fr);gap:10px}
.hero-stat{padding:12px 14px}
.hero-stat .value{font-size:21px}
.hero-stat .sub{font-size:10px}
.hero-meta{font-size:11px;gap:6px 14px}
.verdict{padding:18px}
.verdict p{font-size:17.5px}
.summary-cards .card{padding:18px}
.reason{grid-template-columns:22px 1fr;gap:8px}
.boundary{padding:16px}
.pd-row{padding:9px 0}
.pd-track{height:30px}
.pd-track:before{top:14px}
.pd-link{top:13px}
.pd-mid{top:5px;bottom:5px}
.pd-point.in{top:3px}
.pd-point.out{top:16px}
}
"""

    period_start = min(S[k]["period"][0] for k in KEYS)
    period_end = max(S[k]["period"][1] for k in KEYS)
    total_apps = sum(S[k]["apps"] for k in KEYS)
    hero_stats = [
        ("申请数", [S[k]["apps"] for k in KEYS], "int", f"置入 / 置出 · 合并 {total_apps:,} 条"),
        ("工资收入占比", pv("median.wage_share"), "pct", "中位数 · 每条申请先算占比"),
        ("现金贷/工资贷使用", pv("flags.flag_payday.0"), "pct", "有流水申请"),
        ("存在未结清拒付", pv("flags.flag_unresolved_dishonour.0"), "pct", "182 天窗口"),
    ]
    hero = "".join(
        f'<div class="hero-stat"><div class="label">{label}</div>'
        f'<div class="value"><span class="f">{fmt(vals[0], kind)}</span>'
        f'<span class="slash">/</span><span class="w">{fmt(vals[1], kind)}</span></div>'
        f'<div class="sub">{sub}</div></div>'
        for label, vals, kind, sub in hero_stats)

    header = f'''<header class="hero"><div class="wrap">
<div class="eyebrow light">EXISTING CUSTOMER SWAP · COHORT PROFILE</div>
<h1><span style="color:#5eead4">Swap In</span> <span class="x">×</span> <span style="color:#fcd34d">Swap Out</span><br>新模型置入了谁，又置出了谁</h1>
<p class="hero-lede">置换不是“好客户 vs 坏客户”，而是一次<strong>风险类型交换</strong>。置入组工资可见性更弱、现金流更薄，但负债与拒付更轻；置出组工资更稳定，但借贷与拒付更重。两组同产品、同版特征，合并 {total_apps:,} 条统一评分。</p>
<div class="hero-meta">
<span><i class="dot f"></i>Swap In · 置入 · {S["swap_in"]["apps"]:,} 条申请 · {S["swap_in"]["users"]:,} 人</span>
<span><i class="dot w"></i>Swap Out · 置出 · {S["swap_out"]["apps"]:,} 条申请 · {S["swap_out"]["users"]:,} 人</span>
<span>样本期 {period_start:%Y-%m-%d} 至 {period_end:%Y-%m-%d} · 均为 Old 用户</span>
</div>
<div class="hero-grid">{hero}</div>
<p class="hero-note">所有成对数值按 <strong>Swap In / Swap Out</strong> 排列。健康分与维度分来自合并样本的统一断点，是相对分数，不是违约概率；两组按置换结果划分，不是随机实验，差异只作描述。</p>
</div></header>'''

    nav_items = [("summary", "结论"), ("swap_in", "置入侧"),
                 ("swap_out", "置出侧"), ("logic", "交换逻辑与边界")]
    nav = ('<nav class="nav" aria-label="报告章节"><div class="nav-inner">'
           + "".join(f'<a href="#{k}">{label}</a>' for k, label in nav_items) + "</div></nav>")

    parts = []

    # 01 结论
    summary_rows = [
        ("工资收入占比", pv("median.wage_share")),
        ("多源/非工资收入", pv("archetypes.多源/非工资")),
        ("有放款记录", pv("disburse_rate")),
        ("存在未结清拒付", pv("flags.flag_unresolved_dishonour.0")),
    ]
    summary_table = ('<div class="summary-table"><div class="summary-head"><span>指标</span>'
                     '<span>置入</span><span>置出</span></div>'
                     + "".join(
                         f'<div><span>{label}</span><span>{fmt(a, "pct")}</span>'
                         f'<span>{fmt(b, "pct")}</span></div>'
                         for label, (a, b) in summary_rows)
                     + "</div>")
    content = ('<div class="verdict"><div class="verdict-label">核心结论</div>'
               '<p>置入的是<strong>“工资更不可见、负债更轻、现金流更薄”</strong>的人；'
               '置出的是<strong>“工资更稳定、但借贷与拒付更重”</strong>的人。</p></div>')
    content += summary_table
    content += f'''<div class="two-col summary-cards">
    <div class="card"><h3><span class="cohort-pill in">Swap In</span>置入侧 · 为什么可能进</h3>
    <p>工资占比只有 {fmt(S["swap_in"]["median"]["wage_share"], "pct")}，无工资入账 {fmt(S["swap_in"]["flags"]["flag_no_wage"][0], "pct")}；历史放款笔数中位 {fmt(S["swap_in"]["median"]["disburse_cnt"], "int")}、金额 {fmt(S["swap_in"]["raw"]["db_disburse_sum"], "money")}。与置出组相比，负债和拒付更轻；代价是支出≥收入 {fmt(S["swap_in"]["burn_bands"]["支出达到或超过收入"], "pct")}、低缓冲 {fmt(S["swap_in"]["flags"]["flag_low_buffer"][0], "pct")}。</p></div>
    <div class="card"><h3><span class="cohort-pill out">Swap Out</span>置出侧 · 为什么可能出</h3>
    <p>工资更稳定，但放款笔数中位 {fmt(S["swap_out"]["median"]["disburse_cnt"], "int")}、金额 {fmt(S["swap_out"]["raw"]["db_disburse_sum"], "money")}；未结清拒付 {fmt(S["swap_out"]["median"]["unresolved_cnt"], "int")} 笔、{fmt(S["swap_out"]["raw"]["unresolved_amt_p50"], "money")}，部分解决 {fmt(S["swap_out"]["dishonour_status"]["部分解决"], "pct")}。现金流压力反而低于置入组。</p></div>
    </div>'''
    content += f'''<div class="takeaway-grid">
    <div class="takeaway"><div class="tag">判断 01</div><h3>不是好坏客户，是风险类型交换</h3>
    <p>置入组用工资可见性和现金流缓冲，换更低的负债/拒付；置出组相反。两组的差别在风险类型，不在综合优劣。</p></div>
    <div class="takeaway"><div class="tag">判断 02</div><h3>现有健康分不是置换边界</h3>
    <p>现有健康分中位 {fmt(S["swap_in"]["health_median"], "one")} / {fmt(S["swap_out"]["health_median"], "one")}（置入 / 置出），置出组反而更高；高风险承压 {fmt(S["swap_in"]["segments"]["高风险承压"], "pct")} / {fmt(S["swap_out"]["segments"]["高风险承压"], "pct")} 也更高。分数与分群信号不一致，说明置换不是按综合健康分切，模型可能用了不同权重或否决项。</p></div>
    <div class="takeaway"><div class="tag">判断 03</div><h3>覆盖提升伴随现金流脆弱</h3>
    <p>置入组支出≥收入 {fmt(S["swap_in"]["burn_bands"]["支出达到或超过收入"], "pct")}、低缓冲 {fmt(S["swap_in"]["flags"]["flag_low_buffer"][0], "pct")}；若目标是降坏账，先验证薄缓冲带来的逾期风险。</p></div>
    </div>'''
    parts.append(section("summary", "01 · 结论", "先看结论：置入了谁，置出了谁",
                         "核心判断与两侧画像；先给答案，再看证据。", content))

    # 02 置入侧
    content = (f'<p class="lead">Swap In 并不是“没有收入”。182 天总收入中位数 '
               f'{fmt(S["swap_in"]["raw"]["inc_global_sum"], "k")}，与置出组 '
               f'{fmt(S["swap_out"]["raw"]["inc_global_sum"], "k")} 接近；但工资占比只有 '
               f'{fmt(S["swap_in"]["median"]["wage_share"], "pct")}，多源/非工资收入占 '
               f'{fmt(S["swap_in"]["archetypes"]["多源/非工资"], "pct")}，无工资入账 '
               f'{fmt(S["swap_in"]["flags"]["flag_no_wage"][0], "pct")}。同时，历史负债和拒付都更轻。</p>')
    content += paired("收入与现金流：工资可见性更弱，缓冲更薄", [
        ("工资收入占比", pv("median.wage_share"), "中位数"),
        ("多源/非工资收入", pv("archetypes.多源/非工资"), "收入形态占比"),
        ("无工资入账", pv("flags.flag_no_wage.0"), "风险标签"),
        ("支出达到或超过收入", pv("burn_bands.支出达到或超过收入"), "近 28 天"),
        ("流动性缓冲不足", pv("flags.flag_low_buffer.0"), "风险标签"),
        ("现金贷/工资贷使用", pv("flags.flag_payday.0"), "风险标签"),
    ], caption="统一 0–100% 刻度 · 有流水申请", kind="pct",
        note="所有比例在各自客群内计算；IN = Swap In（圆点），OUT = Swap Out（方点），差值 = Swap Out − Swap In。")
    content += '<p class="table-note">以下数字均为置入 / 置出。</p>'
    content += f'''<div class="reasons">
    <div class="reason"><div class="num">01</div><div><h3>非工资/多源收入被识别</h3>
    <p>多源/非工资 {pct_pair("archetypes.多源/非工资")}；工资占比 {pct_pair("median.wage_share")}；无工资入账 {pct_pair("flags.flag_no_wage.0")}。模型可能更依赖交易流水识别收入，而不是工资标签。</p></div></div>
    <div class="reason"><div class="num">02</div><div><h3>历史负债更轻</h3>
    <p>有放款 {pct_pair("disburse_rate")}；放款笔数中位 {fmt(S["swap_in"]["median"]["disburse_cnt"], "int")} / {fmt(S["swap_out"]["median"]["disburse_cnt"], "int")}；金额中位 {fmt(S["swap_in"]["raw"]["db_disburse_sum"], "money")} / {fmt(S["swap_out"]["raw"]["db_disburse_sum"], "money")}；现金贷/工资贷 {pct_pair("flags.flag_payday.0")}。</p></div></div>
    <div class="reason"><div class="num">03</div><div><h3>用薄缓冲换低负债</h3>
    <p>未结清拒付 {pct_pair("flags.flag_unresolved_dishonour.0")}，笔数中位 {fmt(S["swap_in"]["median"]["unresolved_cnt"], "int")} / {fmt(S["swap_out"]["median"]["unresolved_cnt"], "int")}，金额中位 {fmt(S["swap_in"]["raw"]["unresolved_amt_p50"], "money")} / {fmt(S["swap_out"]["raw"]["unresolved_amt_p50"], "money")}；但支出≥收入 {pct_pair("burn_bands.支出达到或超过收入")}，低缓冲 {pct_pair("flags.flag_low_buffer.0")}。这是取舍，不是全面更优。</p></div></div>
    </div>'''
    content += ('<div class="notes">'
                '<p><strong>变量线索：</strong>新模型仍以债务类交易占比和拒付笔数开头，但拒付窗口从 168 天缩短到 56 天，'
                '并加入余额波动、内部/外部转账、工资消费率、非 SACC 贷款和 BNPL 等近期信号。'
                '置入组的非工资流水、余额波动和现金流缺口落在这些变量视野内；工资痕迹弱，不再等于收入不可见。'
                '这是相容解释，不是模型实际打分方向。</p>'
                f'<p><strong>排除：</strong>收入规模 {pct_pair("raw.inc_global_sum", "k")}、'
                f'活跃天数 {pct_pair("median.active_days", "int")}、现有健康分 {pct_pair("health_median", "one")}，'
                '都不支持“收入更少 / 更活跃 / 更健康”的解释。</p>'
                '</div>')
    parts.append(section("swap_in", "02 · 置入侧", "置入的人：工资更不可见，负债更轻",
                         "收入、现金流与负债特征，以及可能被置入的原因。", content))

    # 03 置出侧
    content = (f'<p class="lead">Swap Out 的工资更稳定：工资占比 '
               f'{fmt(S["swap_out"]["median"]["wage_share"], "pct")}，稳定工资形态 '
               f'{fmt(S["swap_out"]["archetypes"]["稳定工资"], "pct")}，收入稳定性得分 '
               f'{fmt(S["swap_out"]["dims"]["income"], "one")}。它的现金流压力反而更低——'
               f'支出≥收入 {fmt(S["swap_out"]["burn_bands"]["支出达到或超过收入"], "pct")}'
               f'（置入 {fmt(S["swap_in"]["burn_bands"]["支出达到或超过收入"], "pct")}）、低缓冲 '
               f'{fmt(S["swap_out"]["flags"]["flag_low_buffer"][0], "pct")}'
               f'（置入 {fmt(S["swap_in"]["flags"]["flag_low_buffer"][0], "pct")}）。'
               '历史借贷存量与拒付恢复，才是真正拉开差距的地方。</p>')
    content += paired("负债与拒付：置出组更重", [
        ("有放款记录", pv("disburse_rate"), "历史 182 天"),
        ("现金贷/工资贷使用", pv("flags.flag_payday.0"), "风险标签"),
        ("负债压力高", pv("flags.flag_high_debt.0"), "风险标签"),
        ("存在未结清拒付", pv("flags.flag_unresolved_dishonour.0"), "182 天窗口"),
        ("拒付部分解决", pv("dishonour_status.部分解决"), "拒付恢复状态"),
    ], caption="统一 0–100% 刻度 · 有流水申请", kind="pct",
        note="所有比例在各自客群内计算；IN = Swap In（圆点），OUT = Swap Out（方点），差值 = Swap Out − Swap In。")
    content += '<p class="table-note">以下数字均为置入 / 置出。</p>'
    content += f'''<div class="reasons">
    <div class="reason"><div class="num">01</div><div><h3>历史借贷更重</h3>
    <p>有放款 {pct_pair("disburse_rate")}；放款笔数中位 {fmt(S["swap_in"]["median"]["disburse_cnt"], "int")} / {fmt(S["swap_out"]["median"]["disburse_cnt"], "int")}；金额中位 {fmt(S["swap_in"]["raw"]["db_disburse_sum"], "money")} / {fmt(S["swap_out"]["raw"]["db_disburse_sum"], "money")}；现金贷/工资贷 {pct_pair("flags.flag_payday.0")}；高负债 {pct_pair("flags.flag_high_debt.0")}。</p></div></div>
    <div class="reason"><div class="num">02</div><div><h3>拒付恢复更差</h3>
    <p>未结清拒付 {pct_pair("flags.flag_unresolved_dishonour.0")}；笔数中位 {fmt(S["swap_in"]["median"]["unresolved_cnt"], "int")} / {fmt(S["swap_out"]["median"]["unresolved_cnt"], "int")}；金额中位 {fmt(S["swap_in"]["raw"]["unresolved_amt_p50"], "money")} / {fmt(S["swap_out"]["raw"]["unresolved_amt_p50"], "money")}；部分解决 {pct_pair("dishonour_status.部分解决")}。整体比例区间有重叠，但笔数、金额和恢复状态方向一致。</p></div></div>
    <div class="reason"><div class="num">03</div><div><h3>工资稳定没有抵消上述信号</h3>
    <p>收入稳定性得分 {pct_pair("dims.income", "one")}；现有健康分中位 {pct_pair("health_median", "one")}；高风险承压 {pct_pair("segments.高风险承压")}。置出组靠收入维度把总分拉高；如果按现有健康分排序，置出组反而更高，新模型可能用了不同权重或额外否决项。</p></div></div>
    </div>'''
    content += ('<div class="notes">'
                '<p><strong>变量线索：</strong>两版模型都看债务和拒付，差别更可能在“看多近、看多细”。'
                '新模型把拒付窗口缩短到 56 天，并加入非 SACC 贷款、BNPL 和 MyPayNow 竞品使用等更近期的负债信号；'
                '旧模型保留更多福利金、公用事业、医疗、金额总量等长期结构。'
                '置出组“工资更稳定、但历史借贷存量更重、拒付恢复更差”的画像，与这种变量重心变化相容；'
                '但不能证明模型的实际方向。另一个不能排除的解释是策略/政策规则而非纯模型分数。</p>'
                f'<p><strong>一个反向信号：</strong>置出组近 28 天净偿还占 '
                f'{fmt(S["swap_out"]["lender_behaviors"]["净偿还"], "pct")}，高于置入组 '
                f'{fmt(S["swap_in"]["lender_behaviors"]["净偿还"], "pct")}；'
                '“借得更多”更像历史存量，而不是当前加速扩张。</p>'
                '</div>')
    parts.append(section("swap_out", "03 · 置出侧", "置出的人：工资更稳，借贷与拒付更重",
                         "收入、负债与拒付特征，以及可能被置出的原因。", content))

    # 04 交换逻辑与边界
    content = ('<p class="lead">五维评分把交换方向量化：置出组收入稳定性更高，置入组负债压力得分更高'
               '（负债压力得分越高表示压力越低）。这不是现有健康分的简单排序，而是按风险类型重新排序。</p>')
    dim_pairs = [(DIM_LABELS[d], [S[k]["dims"][d] for k in KEYS], "") for d in DIM_LABELS]
    content += paired("五维评分：收入与负债方向相反", dim_pairs,
                      "合并样本统一断点 · 0–100 分，越高越好",
                      kind="score",
                      note="IN = Swap In（圆点）；OUT = Swap Out（方点）。差值 = Swap Out − Swap In，单位：分。负债压力得分越高表示压力越低。")
    content += (f'<div class="callout"><strong>关键结论：</strong>'
                '新模型的置换边界与现有健康分不完全一致。它更像是在以“负债与拒付”为主要区分轴，'
                '而不是按现有健康分排序；两版模型的变量窗口与重心差异，为这种边界变化提供了相容解释，但不能证明方向；'
                '代价是置入组的现金流更脆弱。</div>')
    content += ('<div class="boundary"><h3>解释边界与后续验证</h3><ul class="clean-list">'
                '<li><b>只描述差异：</b>两组按置换结果划分，不是随机实验；Swap Out 仅 139 条，比例类指标 95% CI 较宽；'
                'duedate_1m_5 仅 83/641 与 31/139 有值（阳性 11/5），不能做结果对比。</li>'
                '<li><b>原因仍是假设：</b>变量清单只说明模型能看什么，方向需模型评分、SHAP/特征重要度和拒绝原因码验证；'
                '也不能排除策略、名单或人工规则。</li>'
                '<li><b>业务判断：</b>若目标是降低坏账，重点跟踪置入组的现金流断裂与小额拒付滚动；'
                '若目标是扩大覆盖，价值在覆盖非工资/多源收入人群，而不是风险更低。</li>'
                '</ul></div>')
    content += (f'<p class="table-note">补充：按收入形态标准化后，未结清拒付原始差 '
                f'{pp("flags.flag_unresolved_dishonour.0")}，加权后 '
                f'{cross_std["flag_unresolved_dishonour"][1] - cross_std["flag_unresolved_dishonour"][0]:+.1f}pp；'
                f'现金贷/工资贷原始差 {pp("flags.flag_payday.0")}，加权后仍 '
                f'{cross_std["flag_payday"][1] - cross_std["flag_payday"][0]:+.1f}pp。'
                '收入结构解释一部分拒付差，但解释不了借贷差。</p>')
    content += (f'<details class="details"><summary>数据来源与口径</summary><div class="details-body">'
                f'<p>输入两份 v1_2 特征文件，合并 {total_apps:,} 条申请；沿用 products/fundo.json 的 122 个注册指标与 46 个计分指标，'
                '断点由合并样本统一计算。金额类指标做同一哨兵值清洗；分群、维度分与标签直接读取 user_scores.csv。</p>'
                '<p>收入形态标准化以稳定工资、不稳定工资、多源/非工资三类在合并有流水样本中的占比为权重；'
                f'福利依赖组 Swap Out 为 0 条，未纳入。样本期 {period_start:%Y-%m-%d} 至 {period_end:%Y-%m-%d}，均为 Old 用户；'
                f'Swap Out 的 {S["swap_out"]["apps"]:,} 条申请来自 {S["swap_out"]["users"]:,} 个 user_id，另 1 人 2 条申请。</p>'
                '</div></details>')
    parts.append(section("logic", "04 · 交换逻辑与边界", "交换逻辑：用工资稳定性换负债安全",
                         "五维评分、解释边界与后续验证。", content))

    footer = ('<footer class="footer"><div class="wrap footer-inner">'
              f'<span>Swap In × Swap Out 置换画像 · 合并统一评分</span>'
              f'<span>样本 {total_apps:,} 条 · 生成 {pd.Timestamp.now():%Y-%m-%d %H:%M} · '
              '<a href="#summary">数据与口径</a></span></div></footer>')

    output = ('<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1">'
              '<meta name="description" content="新模型置入了谁、置出了谁：Swap In / Swap Out 的核心结论、画像证据、可能原因与解释边界。">'
              f'<title>新模型置入了谁，又置出了谁 · Swap In × Swap Out</title><style>{css}</style></head>'
              f'<body>{header}{nav}<main>{"".join(parts)}</main>{footer}</body></html>\n')
    REPORT.write_text(output, encoding="utf-8")
    print(f"report: {REPORT} ({REPORT.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
