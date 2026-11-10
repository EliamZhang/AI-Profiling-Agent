#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build PPT-style slide decks for the individual profile reports.

Usage:
    python build_profile_decks.py                # all four datasets
    python build_profile_decks.py --only fundo,swap_in

Outputs (in output/html/):
    fundo_profile_report_slides.html
    wagego_profile_report_slides.html
    existing_swap_in_profile_report_slides.html
    existing_swap_out_profile_report_slides.html
"""

from __future__ import annotations

import argparse
import gc
import html
import re
import sys
from datetime import datetime
from pathlib import Path

import existing_swap_in_profile_report as swap_in_mod
import existing_swap_out_profile_report as swap_out_mod
import fundo_profile_report as fundo_mod
import wagego_profile_report as wagego_mod
import deck_framework as ui
from plotly.offline import get_plotlyjs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "html"

DATASETS = {
    "fundo": {
        "module": fundo_mod,
        "csv": ROOT / "data" / "fundo_sample_with_feature_all.csv",
        "filename": "fundo_profile_report_slides.html",
        "source": "fundo_sample_with_feature_all.csv",
        "name": "Fundo",
        "brand": "FUNDO <em>·</em> USER PROFILE",
        "tag": "USER PROFILE REPORT",
    },
    "wagego": {
        "module": wagego_mod,
        "csv": ROOT / "data" / "wagego_sample_with_feature_all.csv",
        "filename": "wagego_profile_report_slides.html",
        "source": "wagego_sample_with_feature_all.csv",
        "name": "WageGo",
        "brand": "WAGEGO <em>·</em> USER PROFILE",
        "tag": "USER PROFILE REPORT",
    },
    "swap_in": {
        "module": swap_in_mod,
        "csv": ROOT / "data" / "existing_swap_in_v1_2_with_feature_all.csv",
        "filename": "existing_swap_in_profile_report_slides.html",
        "source": "existing_swap_in_v1_2_with_feature_all.csv",
        "name": "Existing Swap In",
        "brand": "SWAP IN <em>·</em> USER PROFILE",
        "tag": "EXISTING CUSTOMER PROFILE",
    },
    "swap_out": {
        "module": swap_out_mod,
        "csv": ROOT / "data" / "existing_swap_out_v1_2_with_feature_all.csv",
        "filename": "existing_swap_out_profile_report_slides.html",
        "source": "existing_swap_out_v1_2_with_feature_all.csv",
        "name": "Existing Swap Out",
        "brand": "SWAP OUT <em>·</em> USER PROFILE",
        "tag": "EXISTING CUSTOMER PROFILE",
    },
}

SEG_COLORS = {
    "盈余型": "#0E7558",
    "平衡型": "#5E8BB4",
    "赤字型": "#C98A2E",
    "重压型": "#B95F5F",
}


def split_finding(text: str) -> tuple[str, str]:
    m = re.match(r"\s*<b>(.*?)</b>(.*)", text, re.S)
    if m:
        return m.group(1), m.group(2).strip()
    return "", text


def build_profile_findings(mod, m: dict, seg) -> list[str]:
    fm = mod.fmt_money
    fp = mod.fmt_pct

    def sv(name: str, col: str):
        try:
            return float(seg.loc[name, col])
        except Exception:
            return float("nan")

    def num(x, digits: int = 2):
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return "—"

    return [
        f"<b>收支缺口是常态，支出端为主、偿债进一步扩大缺口。</b> {fp(m['pre_debt_neg'])} 的用户在偿债前已收不抵支，计入还本付息后升至 {fp(m['sur_neg'])}，中位月盈余 {fm(m['sur_p50'])}。重压型收入中位 {fm(sv('重压型', 'inc_m'))}，高于平衡型的 {fm(sv('平衡型', 'inc_m'))}，缺口并非单纯收入不足。",
        f"<b>工资覆盖广，但入账金额高度依赖其他来源。</b> {fp(m['wage_incidence'])} 有工资收入，工资中位 {fm(m['wage_median_m'])}/月，但工资仅占入账金额 {fp(m['wage_share_agg'])}，“其他入账”占 {fp(m['other_share_agg'])}，其中外部+内部转账合计占分项收入 {fp(m['transfer_inc_share'])}。总入账 CV 中位 {num(m['income_cv_p50'])}，收入来源数中位 {num(m['income_source_p50'], 0)} 个；{fp(m['wage_3m6m_decline'])} 的用户近 3 月工资较近 6 月下降超 10%。",
        f"<b>流动性缓冲接近于零。</b> 中位账户余额仅 {fm(m['bal_p50'])}，只能覆盖 {m['buffer_p50']:.1f} 天支出；{fp(m['buffer_lt7'])} 的用户缓冲不足 7 天，小额意外支出即可直接转为透支；6 个月内余额正负切换中位 {num(m['sign_change_p50'], 0)} 次，反复进出透支。",
        f"<b>支出刚性强，可压缩空间有限。</b> 按全部借记流水口径，外部转账占 {fp(m['ext_transfer_share_global'])}、内部转账占 {fp(m['internal_transfer_share_global'])}；债务相关支出占 {fp(m['debt_share_p50'])}，高承诺支出占收入中位 {fp(m['high_commit_share_p50'])}；可识别消费中位仅占总支出 {fp(m['cons_share_exp_p50'])}，支出端可压缩余地不大。",
        f"<b>借贷以短期现金贷为主，借款扩张快于还款。</b> 放款金额结构为 {m['product_mix_text'] or '—'}；近 1 月/近 3 月新增借款比中位 {num(m['draw_trend_p50'])}，高于还款比的 {num(m['rep_trend_p50'])}，{fp(m['net_borrow_share'])} 为净借款人，债务滚动特征明显。",
        f"<b>拒付处置能力弱，风险可能继续累积。</b> {fp(m['dish_unres'])} 的用户存在未解决拒付，未解决金额合计 {fm(m['dish_unres_amt_total'], compact=True)}，未解决金额/收入 P90 达 {fp(m['dish_unres_income_p90'])}，解决率仅 {fp(m['dish_resolution_rate'])}；未解决机构数中位 {num(m['dish_institution_p50'], 0)} 家，处置渠道集中但进展有限。",
        f"<b>赌博与手续费渗透广，行为风险不可忽视。</b> {fp(m['gamb_incidence'])} 有赌博交易，占可识别消费 {fp(m['gamb_share_cons'])}，P99 单笔达 {fm(m['gamb_p99'])}；{fp(m['fees_incidence'])} 有手续费支出，手续费/透支类交易发生率达 {fp(m['fees_ticket_incidence'])}。",
        f"<b>人群分化明显，尾部压力集中。</b> 重压型占 {fp(sv('重压型', 'share'))}，月支出中位 {fm(sv('重压型', 'exp_m'))} 高于收入 {fm(sv('重压型', 'inc_m'))}，现金贷使用率 {fp(sv('重压型', 'adv_use'))}、拒付率 {fp(sv('重压型', 'dishon'))}；盈余型仅占 {fp(sv('盈余型', 'share'))}，整体均值掩盖了尾部风险。",
    ]


def build_deck(mod, ds: dict, m: dict, seg, d, figs: dict[str, str], generated_at: str) -> str:
    name = ds["name"]
    fm = mod.fmt_money
    fp = mod.fmt_pct
    brand = ds["brand"]
    section_1 = "01 · 收入与支出"
    section_2 = "02 · 负债、流动性与风险"
    sample_note = f"{html.escape(str(m['sample_month']))} {html.escape(str(m['user_type_label']))}样本"

    findings = build_profile_findings(mod, m, seg)

    def seg_val(seg_name: str, col: str):
        try:
            return float(seg.loc[seg_name, col])
        except Exception:
            return float("nan")

    top_cats = sorted(m["cat_shares"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_cats_text = "、".join(f"{mod.CONSUMPTION_CATS[k]} {fp(v)}" for k, v in top_cats)

    lead_title, lead_body = split_finding(findings[0]) if findings else ("", "")

    slides: list[str] = []

    # 01 Cover
    slides.append(
        ui.slide_cover(
            brand=brand,
            tag=ds["tag"],
            eyebrow=f"用户画像报告 · {generated_at}",
            title=f"{html.escape(name)}<br>银行流水用户画像",
            desc=(
                f"基于 {ds['source']} 的银行流水特征，观察窗口 {mod.WINDOW_DAYS} 天，"
                "从收入、支出、负债、现金流、用户分层与风险信号六个维度呈现人群画像。"
            ),
            statuses=[
                ("green", f"有效流水用户 {m['n_valid']:,}"),
                ("blue", f"覆盖率 {fp(m['coverage'])}"),
            ],
            metrics=[
                ("Applications", f"{m['n_apps']:,}", "申请数"),
                ("Users", f"{m['n_users']:,}", "去重用户"),
                ("Valid", f"{m['n_valid']:,}", "有效流水"),
                ("Sample", html.escape(str(m["sample_month"])), str(m["user_type_label"]) + "样本"),
            ],
        )
    )

    # 02 Executive summary
    exec_body = (
        ui.kpi_cards(
            [
                ("中位月收入", fm(m["inc_p50"]), f"均值 {fm(m['inc_mean'])}"),
                ("中位月支出", fm(m["exp_p50"]), f"均值 {fm(m['exp_mean'])}"),
                ("中位月盈余", fm(m["sur_p50"]), f"负盈余用户 {fp(m['sur_neg'])}"),
                ("有借贷放款", fp(m["any_disburse"]), f"净借款人 {fp(m['net_borrow_share'])}"),
            ]
        )
        + f"""
        <div class="split">
          <div class="statement reveal d2">
            <span class="status green"><i></i>描述性画像</span>
            <h3>{lead_title}</h3>
            <p>{lead_body}</p>
            <div class="mini">
              <div><b>{fp(m['wage_incidence'])}</b><span>工资覆盖率</span></div>
              <div><b>{m['buffer_p50']:.1f} 天</b><span>中位缓冲天数</span></div>
              <div><b>{fp(m['dish_any'])}</b><span>有拒付记录</span></div>
            </div>
          </div>
          <div class="note-card reveal d3">
            <h3>关键判断</h3>
            <p>{findings[1] if len(findings) > 1 else ""}</p>
            <div class="divider-line"></div>
            <p style="font-size:12.5px">本报告为描述性画像，不构成信用评估结论。分层为规则判定，阈值变更会改变人群边界。</p>
          </div>
        </div>
        """
        + ui.note_strip(
            "样本口径",
            f"{sample_note}；有效流水要求 txn_raw_input_cnt &gt; 0 且收入 / 支出 / 盈余非缺失、收入 &gt; 0；"
            f"同一 user_id 保留交易笔数最多的申请。无有效流水用户 {m['n_no_data']:,} 人未纳入画像指标。",
        )
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section="Executive Summary",
            kicker="核心结论",
            title="六个维度看人群底盘",
            sub="收支缺口与薄缓冲是共性底盘；收入来源、借贷依赖与风险信号决定人群分化。",
            body=exec_body,
        )
    )

    # 03 Divider 01
    slides.append(
        ui.slide_divider(
            brand=brand,
            section=section_1,
            num="01",
            kicker="Income & Expense",
            title="收入与支出",
            desc="先看钱从哪里来、花到哪里去：工资覆盖与收入来源决定还款基础，支出刚性与转账流向决定可压缩空间。",
            status_html='<span class="status green"><i></i>画像已完成</span>',
            meta=[
                (fm(m["inc_p50"]), "中位月收入"),
                (fm(m["exp_p50"]), "中位月支出"),
                (fp(m["cons_share_exp_p50"]), "可识别消费 / 总支出"),
            ],
        )
    )

    # 04 Income
    income_body = (
        '<div class="duo">'
        + ui.chart_box(figs["income"], f"n = {m['n_valid']:,}；入账构成以分项收入为分母，借记构成以全部借记流水为分母。")
        + ui.side_cards(
            [
                ("工资覆盖率", fp(m["wage_incidence"]), f"有工资用户中位 {fm(m['wage_median_m'])}/月"),
                ("其他入账占入账金额", fp(m["other_share_agg"]), f"外部 + 内部转账占分项收入 {fp(m['transfer_inc_share'])}"),
                ("入账波动", f"CV {m['income_cv_p50']:.2f}", f"收入来源中位 {m['income_source_p50']:.0f} 个 · 近3月/近6月工资比 {m['wage_3m6m_p50']:.2f}"),
            ]
        )
        + "</div>"
        + ui.fact_strip(
            [
                (f"{m['employer_n'] / m['n_valid']:.1%}" if m["n_valid"] else "—", "可识别雇主占比"),
                (html.escape(str(m["industry_top_text"] or "—")), "主行业"),
                (fp(m["wage_3m6m_decline"]), "近 3 月工资下降 >10%"),
                (f"{m['wage_count_pm_p50']:.1f} 笔", "工资到账 / 月"),
            ]
        )
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_1,
            kicker="1.1 · Income & Employment",
            title="收入与就业",
            sub="工资覆盖面广，但入账金额高度依赖转账类“其他入账”；收入波动偏高、来源分散。",
            body=income_body,
        )
    )

    # 05 Expense
    expense_body = (
        '<div class="duo">'
        + ui.chart_box(figs["consumption"], f"n = {m['n_valid']:,}；占比以可识别消费为分母，可识别消费覆盖总支出 {fp(m['cons_share_exp_agg'])}。")
        + ui.side_cards(
            [
                ("可识别消费 / 总支出", fp(m["cons_share_exp_p50"]), f"可识别消费中位 {fm(m['cons_p50'])}/月"),
                ("转账是资金流出主通道", fp(m["ext_transfer_share_global"]), f"外部转账 / 全部借记 · 内部转账 {fp(m['internal_transfer_share_global'])}"),
                ("债务相关支出", fp(m["debt_share_p50"]), f"高承诺支出 / 收入 {fp(m['high_commit_share_p50'])}"),
            ]
        )
        + "</div>"
        + ui.fact_strip(
            [
                (fm(m["exp_p50"]), "中位月支出"),
                (fm(m["exp_ticket_p50"]), "中位单笔支出"),
                (fp(m["cat_top3_p50"]), "前 3 类消费占比"),
                (top_cats_text, "主要消费类目"),
            ]
        )
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_1,
            kicker="1.2 · Expense & Consumption",
            title="支出与消费",
            sub="真正占用资金的是转账与刚性负担；可识别消费占比有限，支出端可压缩空间不大。",
            body=expense_body,
        )
    )

    # 06 Divider 02
    slides.append(
        ui.slide_divider(
            brand=brand,
            section=section_2,
            num="02",
            kicker="Liability, Liquidity & Risk",
            title="负债、流动性与风险",
            desc="再看钱如何周转、能否缓冲：借贷使用与偿债压力、账户缓冲厚度、拒付与高风险行为共同决定尾部风险。",
            status_html='<span class="status amber"><i></i>尾部风险需重点跟踪</span>',
            meta=[
                (fp(m["any_disburse"]), "有借贷放款"),
                (fm(m["bal_p50"]), "中位账户余额"),
                (fp(m["dish_any"]), "有拒付记录"),
            ],
        )
    )

    # 07 Liability
    liability_body = (
        '<div class="duo">'
        + ui.chart_box(figs["liability"], f"n = {m['n_valid']:,}；使用率与拒付占比以有效用户为分母，金额占比以全部放款金额为分母。")
        + ui.side_cards(
            [
                ("有借贷放款", fp(m["any_disburse"]), f"有还款 {fp(m['any_repay'])} · 放款笔数中位 {m['disb_count_p50']:.0f} 笔"),
                ("偿债 / 收入", fp(m["rep_rate_p50"]), f"P90 {fp(m['rep_rate_p90'])} · 月还款中位 {fm(m['rep_m_p50'])}"),
                ("有拒付记录", fp(m["dish_any"]), f"未解决 {fp(m['dish_unres'])} · 解决率 {fp(m['dish_resolution_rate'])}"),
            ]
        )
        + "</div>"
        + ui.fact_strip(
            [
                (fp(m["adv_use"]), "现金贷 / 工资预支使用率"),
                (fp(m["net_borrow_share"]), "净借款人占比"),
                (fm(m["draw_m_p50"]), "中位月新增借款"),
                (fm(m["dish_unres_amt_total"], compact=True), "未解决拒付金额"),
            ]
        )
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_2,
            kicker="2.1 · Liability & Repayment",
            title="负债与信用履约",
            sub="借贷以短期现金贷为主，新增借款扩张快于还款；拒付规模大、解决率低，是负债端最突出的风险点。",
            body=liability_body,
        )
    )

    # 08 Cashflow
    cashflow_body = (
        '<div class="duo">'
        + ui.chart_box(figs["health"], f"n = {m['n_valid']:,}；盈余直方图显示 P1–P99 范围，余额取最近快照。")
        + ui.side_cards(
            [
                ("负盈余占比", fp(m["sur_neg"]), f"偿债前已有 {fp(m['pre_debt_neg'])} 收不抵支"),
                ("中位账户余额", fm(m["bal_p50"]), f"缓冲 {m['buffer_p50']:.1f} 天 · 不足 7 天 {fp(m['buffer_lt7'])}"),
                ("透支与低余额", fp(m["bal_neg"]), f"余额 < $100 占 {fp(m['bal_lt100'])} · 正负切换 {m['sign_change_p50']:.0f} 次 / 6 个月"),
            ]
        )
        + "</div>"
        + ui.fact_strip(
            [
                (fm(m["sur_p50"]), "中位月盈余"),
                (fp(m["fortnight_neg"]), "双周盈余为负"),
                (fp(m["savings_neg"]), "账户储蓄率为负"),
                (fp(m["buffer_lt30"]), "缓冲不足 30 天"),
            ]
        )
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_2,
            kicker="2.2 · Cashflow & Liquidity",
            title="现金流与流动性",
            sub="偿债前已大量用户收不抵支，计入还本付息后缺口进一步扩大；账户缓冲极薄，收入中断即可转为透支。",
            body=cashflow_body,
        )
    )

    # 09 Segments chart + table
    seg_rows = []
    for s in ("盈余型", "平衡型", "赤字型", "重压型"):
        if s not in seg.index:
            continue
        seg_rows.append(
            [
                f'<span class="chip" style="border-color:{SEG_COLORS[s]}33;color:{SEG_COLORS[s]}">{s}</span>',
                fp(seg_val(s, "share")),
                fm(seg_val(s, "inc_m")),
                fm(seg_val(s, "exp_m")),
                fm(seg_val(s, "sur_m")),
                fm(seg_val(s, "bal")),
                fp(seg_val(s, "rep_rate")),
                fp(seg_val(s, "adv_use")),
                fp(seg_val(s, "dishon")),
            ]
        )
    seg_table = ui.table_box(
        ["分层", "占比", "月收入", "月支出", "月盈余", "中位余额", "偿债/收入", "现金贷使用率", "拒付率"],
        seg_rows,
        note="盈余率已截尾至 [-100%, +100%]；完整分层指标见 segment_summary.csv。",
        num_cols=(1, 2, 3, 4, 5, 6, 7, 8),
    )
    segments_body = (
        '<div class="chart-stack">'
        + ui.chart_box(
            figs["segments"],
            f"n = {m['n_valid']:,}；按偿债后盈余率分层：盈余型 ≥5%、平衡型 -5%~5%、赤字型 -20%~-5%、重压型 &lt;-20%。",
        )
        + seg_table
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_2,
            kicker="2.3 · Segmentation",
            title="用户分层",
            sub="整体均值掩盖了尾部风险：重压型人群规模大、收支缺口深，是风险识别与策略分层的重点。",
            body=segments_body,
        )
    )

    # 11 Risk signals
    risk_body = f"""
    <div class="risk-grid">
      <div class="risk-card reveal d1"><div class="k">赌博交易发生率</div><b>{fp(m['gamb_incidence'])}</b><p>占可识别消费 {fp(m['gamb_share_cons'])} · 单笔 P99 {fm(m['gamb_p99'])}</p></div>
      <div class="risk-card reveal d2"><div class="k">手续费支出发生率</div><b>{fp(m['fees_incidence'])}</b><p>手续费 / 透支类交易发生率 {fp(m['fees_ticket_incidence'])}</p></div>
      <div class="risk-card reveal d3"><div class="k">最大单笔支出（中位）</div><b>{fm(m['large_expense_p50'])}</b><p>P90 {fm(m['large_expense_p90'])} · 大额支出 {m['large_expense_cnt_pm_p50']:.1f} 笔/月</p></div>
      <div class="risk-card reveal d2"><div class="k">月交易笔数（中位）</div><b>{m['median_txn_pm']:.0f}<small>笔</small></b><p>交易间隔 P90 {m['txn_gap_p90_p50']:.0f} 天 · 周末消费占比 {fp(m['weekend_share_p50'])}</p></div>
      <div class="risk-card reveal d3"><div class="k">平均月拒付费</div><b>{fm(m['dishfee_mean'])}</b><p>未解决拒付 {fp(m['dish_unres'])} · 涉及机构 {m['dish_institution_p50']:.0f} 家（中位）</p></div>
      <div class="risk-card reveal d4"><div class="k">有赌博交易</div><b>{fp(m['gamb_incidence'])}</b><p>有记录用户 6 个月中位支出 {fm(m['gamb_median_pos'])}</p></div>
    </div>
    """
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_2,
            kicker="2.4 · Risk Signals",
            title="交易行为与风险信号",
            sub="赌博、手续费/透支与拒付是传统收支指标之外的行为风险信号，需要与现金流指标联合解读。",
            body=risk_body,
        )
    )

    # 12-13 Findings
    for chunk_start in (0, 4):
        chunk = findings[chunk_start:chunk_start + 4]
        if not chunk:
            continue
        slides.append(
            ui.slide_content(
                brand=brand,
                section="Findings",
                kicker=f"主要发现 {chunk_start + 1:02d}–{chunk_start + len(chunk):02d}",
                title="主要发现" if chunk_start == 0 else "主要发现（续）",
                sub="以下判断均来自本期样本的描述性统计。",
                body=ui.finding_cards(chunk, start=chunk_start),
            )
        )

    # 14 Closing
    slides.append(
        ui.slide_closing(
            brand=brand,
            kicker="结论 · Takeaway",
            title="画像结论",
            sub=f"{name} 人群的核心特征是：收支缺口常态化、账户缓冲极薄、借贷与拒付暴露高，人群分化集中在尾部。",
            takeaways=[
                ("收入", f"工资覆盖 <b>{fp(m['wage_incidence'])}</b>，但工资仅占入账 <b>{fp(m['wage_share_agg'])}</b>；其他入账占 <b>{fp(m['other_share_agg'])}</b>。"),
                ("流动性", f"中位余额 <b>{fm(m['bal_p50'])}</b>，仅覆盖 <b>{m['buffer_p50']:.1f} 天</b>支出，<b>{fp(m['buffer_lt7'])}</b> 不足 7 天。"),
                ("风险", f"<b>{fp(m['dish_any'])}</b> 有拒付记录、<b>{fp(m['dish_unres'])}</b> 未解决；<b>{fp(m['gamb_incidence'])}</b> 有赌博交易。"),
            ],
            source=(
                f"<b>口径</b>：观察窗口 {mod.WINDOW_DAYS} 天，月度指标按 6 个月折算；盈余 = 收入 − 支出 − 还本付息 − 拒付费 + 新增借款；"
                f"数据源 {ds['source']}。分位数为各自分布独立计算，不可直接相减。"
            ),
            date=f"生成时间 {generated_at}",
        )
    )

    return ui.deck_document(
        title=f"{name} 用户画像 · 演示文稿",
        slides=slides,
        plotly_js=get_plotlyjs(),
    )


def run_dataset(key: str, generated_at: str, out_dir: Path) -> Path:
    ds = DATASETS[key]
    mod = ds["module"]
    print(f"[{key}] loading {ds['csv'].name} ...")
    raw, missing, n_fields = mod.load_data(ds["csv"])
    raw, n_apps = mod.clean_frame(raw)
    df = mod.dedupe_users(raw)
    d, m, seg = mod.build_metrics(df, n_apps)
    figs = {
        "income": mod.chart_income(m),
        "consumption": mod.chart_consumption(m, top_n=10),
        "liability": mod.chart_liability(m),
        "health": mod.chart_health(d, m),
        "segments": mod.chart_segments(seg),
    }
    report = build_deck(mod, ds, m, seg, d, figs, generated_at)
    out_path = out_dir / ds["filename"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[{key}] deck -> {out_path} ({len(report)/1e6:.1f} MB)")
    del raw, df, d, seg, figs, report
    gc.collect()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成用户画像 PPT 式幻灯片 HTML")
    parser.add_argument("--only", default="", help="逗号分隔：fundo,wagego,swap_in,swap_out")
    parser.add_argument("--out", default="", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else OUT
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(DATASETS)
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise SystemExit(f"未知数据集: {unknown}")

    for key in keys:
        run_dataset(key, generated_at, out_dir)


if __name__ == "__main__":
    main()


