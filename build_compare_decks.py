#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build PPT-style slide decks for the comparison reports.

Usage:
    python build_compare_decks.py
    python build_compare_decks.py --pair fundo_wagego

Outputs (in output/html/):
    compare_profile_report_slides.html
    compare_swap_profile_report_slides.html
"""

from __future__ import annotations

import argparse
import gc
import html
import sys
from datetime import datetime
from pathlib import Path

from plotly.offline import get_plotlyjs

import compare_profile_report as cmp_fw
import compare_swap_profile_report as cmp_sw
import deck_framework as ui
import existing_swap_in_profile_report as swap_in_mod
import existing_swap_out_profile_report as swap_out_mod
import fundo_profile_report as fundo_mod
import wagego_profile_report as wagego_mod

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "html"

PAIRS = {
    "fundo_wagego": {
        "module": cmp_fw,
        "left": ("Fundo", fundo_mod, ROOT / "data" / "fundo_sample_with_feature_all.csv"),
        "right": ("WageGo", wagego_mod, ROOT / "data" / "wagego_sample_with_feature_all.csv"),
        "filename": "compare_profile_report_slides.html",
        "brand": "FUNDO <em>×</em> WAGEGO",
        "tag": "PROFILE COMPARISON",
        "title": "Fundo vs WageGo<br>人群画像对比",
        "subtitle": "同一套银行流水特征口径 · 182 天观察窗口 · 两组独立新客样本",
        "source": "fundo_sample_with_feature_all.csv · wagego_sample_with_feature_all.csv",
        "sample_note": "Fundo 样本全部来自 2026-09，WageGo 覆盖 2026-08 至 2026-10；月份构成差异可能影响部分指标。",
        "status_levels": ("green", "purple"),
    },
    "swap_in_out": {
        "module": cmp_sw,
        "left": ("Swap In", swap_in_mod, ROOT / "data" / "existing_swap_in_v1_2_with_feature_all.csv"),
        "right": ("Swap Out", swap_out_mod, ROOT / "data" / "existing_swap_out_v1_2_with_feature_all.csv"),
        "filename": "compare_swap_profile_report_slides.html",
        "brand": "SWAP IN <em>×</em> SWAP OUT",
        "tag": "PROFILE COMPARISON",
        "title": "Swap In vs Swap Out<br>人群画像对比",
        "subtitle": "同一套银行流水特征口径 · 182 天观察窗口 · 存量用户 · 描述性对比",
        "source": "existing_swap_in_v1_2_with_feature_all.csv · existing_swap_out_v1_2_with_feature_all.csv",
        "sample_note": "两组均为存量用户样本；未做统一加权或显著性检验，差异仅用于描述性对比。",
        "status_levels": ("green", "blue"),
    },
}

SEG_ORDER = ("盈余型", "平衡型", "赤字型", "重压型")


def spec_rows(specs, pf, pw, mod) -> list[list[str]]:
    rows = []
    for label, key, kind in specs:
        fv = mod.metric_value(pf, key)
        wv = mod.metric_value(pw, key)
        rows.append(
            [
                html.escape(label),
                mod.fmt_metric(fv, kind),
                mod.fmt_metric(wv, kind),
                mod.fmt_delta(fv, wv, kind),
            ]
        )
    return rows


def pair_value(left: str, right: str) -> str:
    return f"{left}<small> vs </small>{right}"


def build_compare_kpis(module, pf, pw, names: tuple[str, str], specs) -> str:
    left_name, right_name = names
    left_color = module.POP_COLORS[left_name]
    right_color = module.POP_COLORS[right_name]
    cards = []
    for label, key, kind in specs:
        fv = module.metric_value(pf, key)
        wv = module.metric_value(pw, key)
        cards.append(
            {
                "label": label,
                "left_name": left_name,
                "left_value": module.fmt_metric(fv, kind),
                "right_name": right_name,
                "right_value": module.fmt_metric(wv, kind),
                "delta": module.fmt_delta(fv, wv, kind),
                "left_color": left_color,
                "right_color": right_color,
            }
        )
    return ui.cmp_kpi_cards(cards)


def build_pair_deck(
    pair_key: str,
    pair: dict,
    module,
    pf,
    pw,
    figs: dict[str, str],
    generated_at: str,
) -> str:
    left_name, right_name = pair["left"][0], pair["right"][0]
    fm = module.fmt_money
    fp = module.fmt_pct
    fmetric = module.fmt_metric
    mv = module.metric_value
    sv = module.seg_val
    brand = pair["brand"]
    section_a = "01 · 样本与收入"
    section_b = "02 · 支出、负债与流动性"
    deep = "income_resilience" in figs
    k_liability = "2.4" if deep else "2.2"
    k_cashflow = "2.5" if deep else "2.3"
    k_segments = "2.6" if deep else "2.4"
    k_seg_detail = "2.7" if deep else "2.4"
    k_risk = "2.8" if deep else "2.5"
    k_diff = "2.9" if deep else "2.6"
    findings = [f for f in module.build_findings(pf, pw) if f]

    def delta(key, kind):
        return module.fmt_delta(mv(pf, key), mv(pw, key), kind)

    slides: list[str] = []

    # 01 Cover
    slides.append(
        ui.slide_cover(
            brand=brand,
            tag=pair["tag"],
            eyebrow=f"人群画像对比 · {generated_at}",
            title=pair["title"],
            desc=(
                f"使用同一套特征口径对比 {left_name} 与 {right_name} 两组样本，"
                f"覆盖收入、支出、负债、现金流、用户分层与风险信号，共用 {len(pf.m)} 项指标口径。"
            ),
            statuses=[
                (pair["status_levels"][0], f"{left_name} 有效 {pf.m['n_valid']:,}"),
                (pair["status_levels"][1], f"{right_name} 有效 {pw.m['n_valid']:,}"),
            ],
            metrics=[
                (left_name, f"{pf.m['n_valid']:,}", f"有效用户 · 覆盖 {fp(pf.m['coverage'])}"),
                (right_name, f"{pw.m['n_valid']:,}", f"有效用户 · 覆盖 {fp(pw.m['coverage'])}"),
                ("Window", "182 天", "观察窗口"),
                ("Sample", html.escape(str(pf.m["sample_month"])), "对比基准月份"),
            ],
        )
    )

    # 02 Executive summary
    left_stats = (
        f'<div class="exec-stat"><b>{pf.m["n_valid"]:,}</b><span>有效用户</span></div>'
        f'<div class="exec-stat"><b>{fm(pf.m["inc_p50"])}</b><span>中位月收入</span></div>'
        f'<div class="exec-stat"><b>{fm(pf.m["sur_p50"])}</b><span>中位月盈余</span></div>'
    )
    right_stats = (
        f'<div class="exec-stat"><b>{pw.m["n_valid"]:,}</b><span>有效用户</span></div>'
        f'<div class="exec-stat"><b>{fm(pw.m["inc_p50"])}</b><span>中位月收入</span></div>'
        f'<div class="exec-stat"><b>{fm(pw.m["sur_p50"])}</b><span>中位月盈余</span></div>'
    )
    exec_body = (
        build_compare_kpis(
            module,
            pf,
            pw,
            (left_name, right_name),
            [
                ("中位月收入", "inc_p50", "money"),
                ("中位月盈余", "sur_p50", "money"),
                ("偿债 / 收入", "rep_rate_p50", "pct"),
                ("有拒付记录", "dish_any", "pct"),
            ],
        )
        + f"""
        <div class="exec-grid">
          <article class="exec-card green reveal d2">
            <div class="exec-card-top">
              <span class="exec-label">{html.escape(left_name)} · 画像底盘</span>
              <span class="status green"><i></i>基准组</span>
            </div>
            <div class="exec-main">
              <p>{html.escape(left_name)} 收入中位 <b>{fm(pf.m['inc_p50'])}</b>、月盈余中位 <b>{fm(pf.m['sur_p50'])}</b>，工资覆盖 <b>{fp(pf.m['wage_incidence'])}</b>、有拒付记录 <b>{fp(pf.m['dish_any'])}</b>。</p>
              <div class="exec-stats">{left_stats}</div>
            </div>
          </article>
          <article class="exec-card amber reveal d3">
            <div class="exec-card-top">
              <span class="exec-label">{html.escape(right_name)} · 画像底盘</span>
              <span class="status amber"><i></i>对比组</span>
            </div>
            <div class="exec-main">
              <p>{html.escape(right_name)} 收入中位 <b>{fm(pw.m['inc_p50'])}</b>、月盈余中位 <b>{fm(pw.m['sur_p50'])}</b>，工资覆盖 <b>{fp(pw.m['wage_incidence'])}</b>、有拒付记录 <b>{fp(pw.m['dish_any'])}</b>。</p>
              <div class="exec-stats">{right_stats}</div>
            </div>
          </article>
        </div>
        """
        + ui.note_strip("样本口径", pair["sample_note"])
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section="Executive Summary",
            kicker="核心结论",
            title="两组人群，同一套口径",
            sub=findings[0] if findings else "两组样本的收入、支出、负债与风险信号对比。",
            body=exec_body,
        )
    )

    # 03 Divider 01
    slides.append(
        ui.slide_divider(
            brand=brand,
            section=section_a,
            num="01",
            kicker="Sample & Income",
            title="样本与收入",
            desc="先确认两组样本的可比性，再从收入水平、收入结构与就业形态判断还款基础。",
            status_html='<span class="status green"><i></i>同一套特征口径</span>',
            meta=[
                (f"{pf.m['n_valid']:,} / {pw.m['n_valid']:,}", "有效用户"),
                (fm(pf.m["inc_p50"]), f"{left_name} 中位月收入"),
                (fm(pw.m["inc_p50"]), f"{right_name} 中位月收入"),
            ],
        )
    )

    # 04 Sample overview
    overview = ui.table_box(
        ["指标", left_name, right_name, "差异"],
        spec_rows(module.OVERVIEW_ROWS, pf, pw, module),
        note="有效率要求 txn_raw_input_cnt &gt; 0，且收入 / 支出 / 盈余非缺失、收入 &gt; 0；同一 user_id 保留交易笔数最多的申请。",
        num_cols=(1, 2, 3),
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_a,
            kicker="1.1 · Sample Overview",
            title="样本概览",
            sub="两组样本规模与月份构成不同，后续指标差异需要放在这一背景下解读。",
            body=overview,
        )
    )

    # 05 Distributions
    dist_body = (
        '<div class="duo">'
        + ui.chart_box(figs["distributions"], "箱体为 P25–P75，横线为中位数，须线为 1.5×IQR；横轴裁至两组合并后的 P1–P99。")
        + ui.side_cards(
            [
                ("中位月收入", pair_value(fm(pf.m["inc_p50"]), fm(pw.m["inc_p50"])), f"月收入 P25–P75：{fm(pf.m['inc_p25'])}–{fm(pf.m['inc_p75'])} / {fm(pw.m['inc_p25'])}–{fm(pw.m['inc_p75'])}"),
                ("中位月支出", pair_value(fm(pf.m["exp_p50"]), fm(pw.m["exp_p50"])), f"均值 {fm(pf.m['exp_mean'])} / {fm(pw.m['exp_mean'])}"),
                ("中位月盈余", pair_value(fm(pf.m["sur_p50"]), fm(pw.m["sur_p50"])), f"负盈余占比 {fp(pf.m['sur_neg'])} / {fp(pw.m['sur_neg'])}"),
                ("中位账户余额", pair_value(fm(pf.m["bal_p50"]), fm(pw.m["bal_p50"])), f"P25 {fm(pf.m['bal_p25'])} / {fm(pw.m['bal_p25'])}"),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_a,
            kicker="1.2 · Financial Distribution",
            title="核心财务指标分布",
            sub="两组收入与支出分布接近，盈余整体位于零以下，余额集中在低位。",
            body=dist_body,
        )
    )

    # 06 Income
    income_body = (
        '<div class="duo">'
        + ui.chart_box(figs["income_mix"], f"n = {left_name} {pf.m['n_valid']:,} / {right_name} {pw.m['n_valid']:,}；发生率以有效用户为分母，金额占比以分项收入加总为分母。")
        + ui.side_cards(
            [
                ("工资覆盖率", pair_value(fp(pf.m["wage_incidence"]), fp(pw.m["wage_incidence"])), f"有工资用户中位 {fm(pf.m['wage_median_m'])} / {fm(pw.m['wage_median_m'])} 每月"),
                ("工资占入账", pair_value(fp(pf.m["wage_share_agg"]), fp(pw.m["wage_share_agg"])), f"其他入账占比 {fp(pf.m['other_share_agg'])} / {fp(pw.m['other_share_agg'])}"),
                ("入账 CV 中位", pair_value(fmetric(pf.m["income_cv_p50"], "num"), fmetric(pw.m["income_cv_p50"], "num")), f"收入来源中位 {fmetric(pf.m['income_source_p50'], 'num0')} / {fmetric(pw.m['income_source_p50'], 'num0')} 个"),
                ("近 3 月工资下降 >10%", pair_value(fp(pf.m["wage_3m6m_decline"]), fp(pw.m["wage_3m6m_decline"])), f"主行业：{pf.m['industry_top_text']} / {pw.m['industry_top_text']}"),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_a,
            kicker="1.3 · Income & Employment",
            title="收入与就业",
            sub="收入水平接近，工资覆盖面与收入结构存在差异；两组入账波动都偏高。",
            body=income_body,
        )
    )

    # 07 Income resilience
    if deep:
        f_ben = pf.m["benefit_type_inc"]
        w_ben = pw.m["benefit_type_inc"]
        resilience_body = (
            '<div class="duo">'
            + ui.chart_box(
                figs["income_resilience"],
                f"n = {left_name} {pf.m['n_valid']:,} / {right_name} {pw.m['n_valid']:,}；"
                "左图为福利人群中各类型覆盖率（可重叠），右图为各分层工资占入账。",
            )
            + ui.side_cards(
                [
                    (
                        "福利收入覆盖率",
                        pair_value(fp(pf.m["cent_incidence"]), fp(pw.m["cent_incidence"])),
                        f"福利 / 入账中位 {fp(pf.m['cent_share_rec_p50'])} / {fp(pw.m['cent_share_rec_p50'])}（福利人群）",
                    ),
                    (
                        "工资到账波动",
                        pair_value(
                            f"{fmetric(pf.m['wage_interval_std_p50'], 'num1')} 天",
                            f"{fmetric(pw.m['wage_interval_std_p50'], 'num1')} 天",
                        ),
                        f"最长连续 {pf.m['wage_max_run_p50']:.0f} 个月 / "
                        f"{pw.m['wage_max_run_p50']:.0f} 个月",
                    ),
                    (
                        "养老金覆盖率",
                        pair_value(fp(f_ben.get("pension")), fp(w_ben.get("pension"))),
                        f"JobSeeker {fp(f_ben.get('jobseeker'))} / {fp(w_ben.get('jobseeker'))}（福利人群）",
                    ),
                    (
                        "重压型工资占比",
                        pair_value(
                            fp(sv(pf, "重压型", "wage_share")),
                            fp(sv(pw, "重压型", "wage_share")),
                        ),
                        "两组均为各分层最低",
                    ),
                ]
            )
            + "</div>"
            + ui.fact_strip(
                [
                    (
                        f"{fp(f_ben.get('pension'))} / {fp(w_ben.get('pension'))}",
                        "养老金覆盖率（福利人群）",
                    ),
                    (
                        f"{fp(f_ben.get('jobseeker'))} / {fp(w_ben.get('jobseeker'))}",
                        "JobSeeker 覆盖率（福利人群）",
                    ),
                    (
                        f"{fp(pf.m['cent_share_rec_p50'])} / {fp(pw.m['cent_share_rec_p50'])}",
                        "福利 / 入账中位",
                    ),
                    (
                        f"{fp(sv(pf, '重压型', 'wage_share'))} / {fp(sv(pw, '重压型', 'wage_share'))}",
                        "重压型工资占比",
                    ),
                ]
            )
        )
        slides.append(
            ui.slide_content(
                brand=brand,
                section=section_a,
                kicker="1.4 · Income Resilience",
                title="收入结构与韧性",
                sub=(
                    f"{left_name} 福利覆盖更高、福利依赖更强；{right_name} 工资覆盖更广，"
                    "但重压型同样明显依赖非工资入账。"
                ),
                body=resilience_body,
            )
        )

    # 08 Divider 02
    slides.append(
        ui.slide_divider(
            brand=brand,
            section=section_b,
            num="02",
            kicker="Expense, Liability & Liquidity",
            title="支出、负债与流动性",
            desc=(
                "再看资金去向与周转方式：消费深度、刚性支出与真实可支配收入、借贷使用与偿债压力、账户缓冲厚度。"
                if deep
                else "再看资金去向与周转方式：支出刚性与转账流向、借贷使用与偿债压力、账户缓冲厚度。"
            ),
            status_html='<span class="status amber"><i></i>结构差异集中在负债与风险端</span>',
            meta=(
                [
                    (fp(pf.m["fixed_wage_p50"]), f"{left_name} 固定账单 / 工资"),
                    (fp(pw.m["fixed_wage_p50"]), f"{right_name} 固定账单 / 工资"),
                    (fp(pw.m["external_lender_incidence"]), f"{right_name} 外部借贷覆盖"),
                ]
                if deep
                else [
                    (fp(pf.m["ext_transfer_share_global"]), f"{left_name} 外部转账 / 借记"),
                    (fp(pw.m["ext_transfer_share_global"]), f"{right_name} 外部转账 / 借记"),
                    (fp(pw.m["rep_rate_p50"]), f"{right_name} 偿债 / 收入"),
                ]
            ),
        )
    )

    # 08 Expense structure
    expense_body = (
        '<div class="chart-pair">'
        + ui.chart_box(figs["expense_structure"], "外部 / 内部转账占比以全部借记流水为分母；其余占比分母见指标名。")
        + ui.chart_box(figs["consumption"], "仅展示份额最高的 12 个消费类目；Δ 为右组 − 左组（百分点）。")
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker="2.1 · Expense Structure",
            title="支出与消费",
            sub="支出规模接近，但资金流向不同：外部/内部转账结构与赌博类消费是主要差异。",
            body=expense_body,
        )
    )

    # 09 Consumption depth
    if deep:
        cons_depth_body = (
            '<div class="duo">'
            + ui.chart_box(
                figs["consumption_depth"],
                f"n = {left_name} {pf.m['n_valid']:,} / {right_name} {pw.m['n_valid']:,}；"
                "渗透率为有该笔交易的用户占比；品质消费含餐饮、旅游、零售、百货、娱乐、个人护理等。",
            )
            + ui.side_cards(
                [
                    (
                        "品质消费 / 工资中位",
                        pair_value(fp(pf.m["prem_wage_p50"]), fp(pw.m["prem_wage_p50"])),
                        f"P90 {fp(pf.m['prem_wage_p90'])} / {fp(pw.m['prem_wage_p90'])}",
                    ),
                    (
                        "品质消费渗透率",
                        pair_value(fp(pf.m["premium_incidence"]), fp(pw.m["premium_incidence"])),
                        f"重压型 {fp(sv(pf, '重压型', 'prem_wage'))} / {fp(sv(pw, '重压型', 'prem_wage'))}（/ 工资）",
                    ),
                    (
                        "超市渗透率",
                        pair_value(
                            fp(pf.m["cat_pen"].get("groceries")),
                            fp(pw.m["cat_pen"].get("groceries")),
                        ),
                        f"餐饮 {fp(pf.m['cat_pen'].get('dining_out'))} / "
                        f"{fp(pw.m['cat_pen'].get('dining_out'))} · 客单价 "
                        f"{fm(pf.m['cat_ticket'].get('groceries'))} / "
                        f"{fm(pw.m['cat_ticket'].get('groceries'))}（超市）",
                    ),
                    (
                        "赌博与手续费",
                        pair_value(fp(pf.m["gamb_incidence"]), fp(pw.m["gamb_incidence"])),
                        f"手续费 {fp(pf.m['fees_incidence'])} / {fp(pw.m['fees_incidence'])} · "
                        f"赌博占消费 {fp(pf.m['gamb_share_cons'])} / {fp(pw.m['gamb_share_cons'])}",
                    ),
                ]
            )
            + "</div>"
            + ui.fact_strip(
                [
                    (
                        f"{fp(pf.m['prem_wage_p50'])} / {fp(pw.m['prem_wage_p50'])}",
                        "品质消费 / 工资中位",
                    ),
                    (
                        f"{fp(pf.m['prem_wage_p90'])} / {fp(pw.m['prem_wage_p90'])}",
                        "品质消费 / 工资 P90",
                    ),
                    (
                        f"{fp(pf.m['gamb_incidence'])} / {fp(pw.m['gamb_incidence'])}",
                        "赌博渗透率",
                    ),
                    (
                        f"{fp(pf.m['gamb_share_cons'])} / {fp(pw.m['gamb_share_cons'])}",
                        "赌博占可识别消费",
                    ),
                ]
            )
        )
        slides.append(
            ui.slide_content(
                brand=brand,
                section=section_b,
                kicker="2.2 · Consumption Depth",
                title="消费深度与品质消费",
                sub=(
                    f"{right_name} 的消费渗透更广，尤其赌博与手续费；{left_name} 的品质消费强度更高，"
                    "重压型达到工资的 13.8%。"
                ),
                body=cons_depth_body,
            )
        )

    # 10 Fixed obligations
    if deep:
        fixed_body = (
            '<div class="duo">'
            + ui.chart_box(
                figs["fixed_burden"],
                f"n = {left_name} {pf.m['n_valid']:,} / {right_name} {pw.m['n_valid']:,}；"
                "固定账单为特征表中的周期扣款口径；固定账单 / 工资以工资为分母。",
            )
            + ui.side_cards(
                [
                    (
                        "固定账单覆盖率",
                        pair_value(fp(pf.m["fixed_bill_incidence"]), fp(pw.m["fixed_bill_incidence"])),
                        f"有账单用户中位 {fm(pf.m['fixed_bill_median_pos'])} / "
                        f"{fm(pw.m['fixed_bill_median_pos'])}（6 个月）",
                    ),
                    (
                        "房租覆盖率",
                        pair_value(fp(pf.m["rent_incidence"]), fp(pw.m["rent_incidence"])),
                        f"租金中位 {fm(pf.m['rent_median_pos'])} / {fm(pw.m['rent_median_pos'])}（6 个月）· "
                        f"租客房租 / 工资 {fp(pf.m['rent_wage_p50_renters'])} / {fp(pw.m['rent_wage_p50_renters'])}",
                    ),
                    (
                        "固定账单 / 工资中位",
                        pair_value(fp(pf.m["fixed_wage_p50"]), fp(pw.m["fixed_wage_p50"])),
                        f"P90 {fp(pf.m['fixed_wage_p90'])} / {fp(pw.m['fixed_wage_p90'])} · "
                        f">30% 占比 {fp(pf.m['fixed_wage_gt30'])} / {fp(pw.m['fixed_wage_gt30'])}",
                    ),
                    (
                        "收入 / 支出中位",
                        pair_value(
                            f"{pf.m['income_expense_ratio_p50']:.3f}",
                            f"{pw.m['income_expense_ratio_p50']:.3f}",
                        ),
                        f"重压型固定账单 / 工资 {fp(sv(pf, '重压型', 'fixed_wage'))} / "
                        f"{fp(sv(pw, '重压型', 'fixed_wage'))}",
                    ),
                ]
            )
            + "</div>"
            + ui.fact_strip(
                [
                    (
                        f"{fp(pf.m['fixed_bill_incidence'])} / {fp(pw.m['fixed_bill_incidence'])}",
                        "固定账单覆盖率",
                    ),
                    (
                        f"{fp(pf.m['fixed_wage_p50'])} / {fp(pw.m['fixed_wage_p50'])}",
                        "固定账单 / 工资中位",
                    ),
                    (
                        f"{fp(pf.m['fixed_wage_p90'])} / {fp(pw.m['fixed_wage_p90'])}",
                        "固定账单 / 工资 P90",
                    ),
                    (
                        f"{fp(sv(pf, '重压型', 'fixed_wage'))} / {fp(sv(pw, '重压型', 'fixed_wage'))}",
                        "重压型固定账单 / 工资",
                    ),
                ]
            )
        )
        slides.append(
            ui.slide_content(
                brand=brand,
                section=section_b,
                kicker="2.3 · Fixed Obligations",
                title="刚性支出与真实可支配收入",
                sub=(
                    "覆盖率接近，但 Fundo 的固定账单负担更重、尾部更高；"
                    "同样的账单结构下，Fundo 的真实可支配收入更薄。"
                ),
                body=fixed_body,
            )
        )

    # 11 Liability
    net_borrow_sub = (
        f"中位月新增借款 {fm(pf.m['draw_m_p50'])} / {fm(pw.m['draw_m_p50'])}"
    )
    if deep:
        net_borrow_sub += (
            f" · 外部借贷 {fp(pf.m['external_lender_incidence'])} / "
            f"{fp(pw.m['external_lender_incidence'])}"
        )
    liability_body = (
        '<div class="duo">'
        + ui.chart_box(figs["liability"], "使用率与拒付发生率均以有效用户为分母。")
        + ui.side_cards(
            [
                ("有借贷放款", pair_value(fp(pf.m["any_disburse"]), fp(pw.m["any_disburse"])), f"现金贷使用率 {fp(pf.m['adv_use'])} / {fp(pw.m['adv_use'])}"),
                ("偿债 / 收入", pair_value(fp(pf.m["rep_rate_p50"]), fp(pw.m["rep_rate_p50"])), f"中位月还款 {fm(pf.m['rep_m_p50'])} / {fm(pw.m['rep_m_p50'])}"),
                ("净借款人占比", pair_value(fp(pf.m["net_borrow_share"]), fp(pw.m["net_borrow_share"])), net_borrow_sub),
                ("有拒付记录", pair_value(fp(pf.m["dish_any"]), fp(pw.m["dish_any"])), f"未解决 {fp(pf.m['dish_unres'])} / {fp(pw.m['dish_unres'])} · 解决率 {fp(pf.m['dish_resolution_rate'])} / {fp(pw.m['dish_resolution_rate'])}"),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_liability} · Liability & Repayment",
            title="负债与信用履约",
            sub="借贷使用、偿债负担与拒付暴露是两组分化最明显的维度之一。",
            body=liability_body,
        )
    )

    # 10 Cashflow
    cashflow_body = (
        '<div class="chart-pair">'
        + ui.chart_box(figs["cashflow"], "占比均以有效用户为分母；缓冲天数 = 账户余额 /（月支出 / 30）。")
        + ui.chart_box(figs["balance_buckets"], "余额取最近快照；两组各自归一化为 100%。")
        + "</div>"
        + ui.fact_strip(
            [
                (fp(pf.m["sur_neg"]), f"{left_name} 负盈余占比"),
                (fp(pw.m["sur_neg"]), f"{right_name} 负盈余占比"),
                (fp(pf.m["buffer_lt7"]), f"{left_name} 缓冲 < 7 天"),
                (fp(pw.m["buffer_lt7"]), f"{right_name} 缓冲 < 7 天"),
            ]
        )
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_cashflow} · Cashflow & Liquidity",
            title="现金流与流动性",
            sub="两组都以负盈余为主、账户缓冲极薄；流动性是共同底盘，而非区分两组的主轴。",
            body=cashflow_body,
        )
    )

    # 11 Segments charts
    segment_body = (
        '<div class="chart-pair">'
        + ui.chart_box(figs["segments"], "各分层占比，组内归一化为 100%。")
        + ui.chart_box(figs["segment_surplus"], "各分层中位月盈余。")
        + "</div>"
        + ui.fact_strip(
            [
                (fp(sv(pf, "盈余型", "share")), f"{left_name} 盈余型"),
                (fp(sv(pw, "盈余型", "share")), f"{right_name} 盈余型"),
                (fp(sv(pf, "重压型", "share")), f"{left_name} 重压型"),
                (fp(sv(pw, "重压型", "share")), f"{right_name} 重压型"),
            ]
        )
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_segments} · Segmentation",
            title="用户分层",
            sub="按偿债后盈余率分为四层：盈余型 ≥5%、平衡型 -5%~5%、赤字型 -20%~-5%、重压型 <-20%。",
            body=segment_body,
        )
    )

    # 12 Segment table
    seg_rows = []
    for s in SEG_ORDER:
        fv = sv(pf, s, "share")
        wv = sv(pw, s, "share")
        seg_rows.append(
            [
                s,
                fp(fv),
                fp(wv),
                module.fmt_delta(fv, wv, "pct"),
                fm(sv(pf, s, "sur_m")),
                fm(sv(pw, s, "sur_m")),
                fp(sv(pf, s, "dishon")),
                fp(sv(pw, s, "dishon")),
            ]
        )
    seg_table = ui.table_box(
        ["分层", f"{left_name} 占比", f"{right_name} 占比", "差异", f"{left_name} 月盈余", f"{right_name} 月盈余", f"{left_name} 拒付率", f"{right_name} 拒付率"],
        seg_rows,
        note="盈余率已截尾至 [-100%, +100%]；拒付率 = 拒付笔数 /（拒付笔数 + 还款笔数）。",
        num_cols=(1, 2, 3, 4, 5, 6, 7),
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_seg_detail} · Segmentation Detail",
            title="分层指标明细",
            sub="各分层内部的盈余水平与拒付率，决定尾部风险落在哪一类人群。",
            body=seg_table,
        )
    )

    # 13 Risk
    risk_body = (
        '<div class="duo">'
        + ui.chart_box(figs["risk"], "“赌博占消费”为样本加总口径，其余为有效用户发生率。")
        + ui.side_cards(
            [
                ("赌博交易发生率", pair_value(fp(pf.m["gamb_incidence"]), fp(pw.m["gamb_incidence"])), f"赌博占可识别消费 {fp(pf.m['gamb_share_cons'])} / {fp(pw.m['gamb_share_cons'])}"),
                ("手续费支出发生率", pair_value(fp(pf.m["fees_incidence"]), fp(pw.m["fees_incidence"])), f"手续费 / 透支交易发生率 {fp(pf.m['fees_ticket_incidence'])} / {fp(pw.m['fees_ticket_incidence'])}"),
                ("有拒付记录", pair_value(fp(pf.m["dish_any"]), fp(pw.m["dish_any"])), f"未解决 {fp(pf.m['dish_unres'])} / {fp(pw.m['dish_unres'])}"),
                ("催收类拒付", pair_value(fp(pf.m["collections_inc"]), fp(pw.m["collections_inc"])), f"债务整合 {fp(pf.m['debt_consolidation_inc'])} / {fp(pw.m['debt_consolidation_inc'])}"),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_risk} · Risk Signals",
            title="风险信号",
            sub="赌博、手续费与催收暴露构成收支指标之外的风险信号。",
            body=risk_body,
        )
    )

    # 14 Difference overview
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_diff} · Key Differences",
            title="主要差异",
            sub=f"差值 = {right_name} − {left_name}，单位为百分点；仅列可比的比例类指标，按差值绝对值排序。",
            body=ui.chart_box(figs["difference"], "图中仅展示部分指标；完整指标见各章节明细表。"),
        )
    )

    # 15-17 Findings
    for start in range(0, len(findings), 4):
        chunk = findings[start:start + 4]
        if not chunk:
            continue
        slides.append(
            ui.slide_content(
                brand=brand,
                section="Findings",
                kicker=f"主要差异 {start + 1:02d}–{start + len(chunk):02d}",
                title="主要差异与解读" if start == 0 else "主要差异与解读（续）",
                sub="以下判断来自本期两组样本的描述性对比。",
                body=ui.finding_cards(chunk, start=start),
            )
        )

    # 18 Closing
    takeaways = [
        ("收入", f"{left_name} 工资覆盖 <b>{fp(pf.m['wage_incidence'])}</b>、{right_name} <b>{fp(pw.m['wage_incidence'])}</b>；中位月收入 <b>{fm(pf.m['inc_p50'])}</b> vs <b>{fm(pw.m['inc_p50'])}</b>。"),
    ]
    if deep:
        takeaways.append(
            (
                "支出与刚性",
                f"固定账单 / 工资中位 <b>{fp(pf.m['fixed_wage_p50'])}</b> vs <b>{fp(pw.m['fixed_wage_p50'])}</b>，"
                f"品质消费 / 工资中位 <b>{fp(pf.m['prem_wage_p50'])}</b> vs <b>{fp(pw.m['prem_wage_p50'])}</b>；"
                f"重压型固定账单负担 <b>{fp(sv(pf, '重压型', 'fixed_wage'))}</b> vs "
                f"<b>{fp(sv(pw, '重压型', 'fixed_wage'))}</b>。",
            )
        )
    takeaways += [
        ("流动性与杠杆", f"两组负盈余占比 <b>{fp(pf.m['sur_neg'])}</b> vs <b>{fp(pw.m['sur_neg'])}</b>，缓冲不足 7 天 <b>{fp(pf.m['buffer_lt7'])}</b> vs <b>{fp(pw.m['buffer_lt7'])}</b>；偿债/收入 <b>{fp(pf.m['rep_rate_p50'])}</b> vs <b>{fp(pw.m['rep_rate_p50'])}</b>。"),
        ("风险", f"有拒付记录 <b>{fp(pf.m['dish_any'])}</b> vs <b>{fp(pw.m['dish_any'])}</b>；赌博发生率 <b>{fp(pf.m['gamb_incidence'])}</b> vs <b>{fp(pw.m['gamb_incidence'])}</b>。"),
    ]
    slides.append(
        ui.slide_closing(
            brand=brand,
            kicker="结论 · Takeaway",
            title="对比结论",
            sub=(
                f"{left_name} 与 {right_name} 各有短板，没有一方全面占优；"
                "差异主要集中在收入结构、支出刚性、借贷依赖与拒付暴露。"
                if deep
                else f"{left_name} 与 {right_name} 各有短板，没有一方全面占优；"
                "差异主要集中在收入结构、借贷依赖与拒付暴露。"
            ),
            takeaways=takeaways,
            source=(
                f"<b>数据源</b>：{pair['source']}。两组使用同一套字段清单、缺失哨兵与清洗规则；"
                "以上为描述性对比，不做因果推断。"
            ),
            date=f"生成时间 {generated_at}",
        )
    )

    return ui.deck_document(
        title=f"{left_name} vs {right_name} 人群画像对比 · 演示文稿",
        slides=slides,
        plotly_js=get_plotlyjs(),
    )


def run_pair(pair_key: str, generated_at: str, out_dir: Path) -> None:
    pair = PAIRS[pair_key]
    module = pair["module"]
    left_name, left_mod, left_csv = pair["left"]
    right_name, right_mod, right_csv = pair["right"]
    print(f"[{pair_key}] loading profiles ...")
    pf = module.load_profile(left_name, left_mod, left_csv)
    pw = module.load_profile(right_name, right_mod, right_csv)
    figs = {
        "distributions": module.chart_distributions(pf, pw),
        "income_mix": module.chart_income_mix(pf, pw),
        "expense_structure": module.chart_expense_structure(pf, pw),
        "consumption": module.chart_consumption_dumbbell(pf, pw, top_n=12),
        "liability": module.chart_liability(pf, pw),
        "cashflow": module.chart_cashflow(pf, pw),
        "balance_buckets": module.chart_balance_buckets(pf, pw),
        "segments": module.chart_segments(pf, pw),
        "segment_surplus": module.chart_segment_surplus(pf, pw),
        "risk": module.chart_risk(pf, pw),
        "difference": module.chart_difference_overview(pf, pw),
    }
    if hasattr(module, "chart_income_resilience"):
        figs["income_resilience"] = module.chart_income_resilience(pf, pw)
        figs["consumption_depth"] = module.chart_consumption_depth(pf, pw)
        figs["fixed_burden"] = module.chart_fixed_burden(pf, pw)
    report = build_pair_deck(pair_key, pair, module, pf, pw, figs, generated_at)
    out_path = out_dir / pair["filename"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[{pair_key}] deck -> {out_path} ({len(report)/1e6:.1f} MB)")

    del pf, pw, figs, report
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成人群对比 PPT 式幻灯片 HTML")
    parser.add_argument("--pair", choices=list(PAIRS) + ["all"], default="all")
    parser.add_argument("--out", default="", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else OUT
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    keys = list(PAIRS) if args.pair == "all" else [args.pair]
    for key in keys:
        run_pair(key, generated_at, out_dir)


if __name__ == "__main__":
    main()


