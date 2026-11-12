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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from plotly.offline import get_plotlyjs

import compare_profile_report as cmp_fw
import compare_swap_profile_report as cmp_sw
import deck_framework as ui
import existing_swap_in_profile_report as swap_in_mod
import existing_swap_out_profile_report as swap_out_mod
import fundo_profile_report as fundo_mod
import wagego_profile_report as wagego_mod
import compare_profile_enrichment as compare_extra
import profile_enrichment as profile_extra

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
        "subtitle": "同一套银行流水特征口径 · 182 天观察窗口 · 两组新客样本",
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


def build_compare_kpis(module, pf, pw, names: tuple[str, str], specs, with_vars: bool = False) -> str:
    left_name, right_name = names
    left_color = module.POP_COLORS[left_name]
    right_color = module.POP_COLORS[right_name]
    cards = []
    for label, key, kind in specs:
        fv = module.metric_value(pf, key)
        wv = module.metric_value(pw, key)
        card = {
            "label": label,
            "left_name": left_name,
            "left_value": module.fmt_metric(fv, kind),
            "right_name": right_name,
            "right_value": module.fmt_metric(wv, kind),
            "delta": module.fmt_delta(fv, wv, kind),
            "left_color": left_color,
            "right_color": right_color,
        }
        if with_vars:
            if key.startswith("enrich:"):
                card["var"] = "profile_enrichment." + key.split(":", 1)[1]
            elif key == "employer_share":
                card["var"] = "employer_n / n_valid"
            else:
                card["var"] = key
        cards.append(card)
    return ui.cmp_kpi_cards(cards)


def build_concise_pair_deck(
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
    brand = pair["brand"]
    f_enrich = pf.m["profile_enrichment"]
    w_enrich = pw.m["profile_enrichment"]
    f_ben = pf.m["benefit_type_inc"]
    w_ben = pw.m["benefit_type_inc"]
    sv = module.seg_val

    def delta(key, kind):
        return module.fmt_delta(mv(pf, key), mv(pw, key), kind)

    fc = module.POP_COLORS[left_name]
    wc = module.POP_COLORS[right_name]

    def cmp_cards(cards):
        inner = ""
        for label, fv, wv, note, var in cards:
            inner += (
                '<div class="side-card cmp">'
                f'<div class="k"><span>{label}</span><code class="var">{var}</code></div>'
                f'<div class="prow"><span class="who"><i style="background:{fc}"></i>{left_name}</span>'
                f"<b>{fv}</b></div>"
                f'<div class="prow"><span class="who"><i style="background:{wc}"></i>{right_name}</span>'
                f"<b>{wv}</b></div>"
                f"<p>{note}</p></div>"
            )
        return f'<div class="side-stack reveal d2">{inner}</div>'

    slides: list[str] = []

    # 01 收入
    income_body = (
        build_compare_kpis(
            module,
            pf,
            pw,
            (left_name, right_name),
            [
                ("中位月收入", "inc_p50", "money"),
                ("工资覆盖率", "wage_incidence", "pct"),
                ("福利 / 入账中位", "cent_share_rec_p50", "pct"),
                ("可识别雇主", "employer_share", "pct"),
            ],
            with_vars=True,
        )
        + '<div class="duo">'
        + ui.chart_box(
            figs["income_mix"],
            f"n = {left_name} {pf.m['n_valid']:,} / {right_name} {pw.m['n_valid']:,}；"
            "发生率以有效用户为分母，金额占比以分项收入加总为分母。"
            "变量：wage_incidence / cent_incidence / other_incidence / "
            "wage_share_agg / cent_share_agg / other_share_agg。",
        )
        + cmp_cards(
            [
                (
                    "工资占入账",
                    fp(pf.m["wage_share_agg"]),
                    fp(pw.m["wage_share_agg"]),
                    f"其他入账 {fp(pf.m['other_share_agg'])} / {fp(pw.m['other_share_agg'])}",
                    "wage_share_agg",
                ),
                (
                    "收入波动 CV 中位",
                    fmetric(pf.m["income_cv_p50"], "num"),
                    fmetric(pw.m["income_cv_p50"], "num"),
                    f"收入来源中位 {fmetric(pf.m['income_source_p50'], 'num0')} / "
                    f"{fmetric(pw.m['income_source_p50'], 'num0')} 个",
                    "income_cv_p50",
                ),
                (
                    "养老金覆盖率（福利人群）",
                    fp(f_ben.get("pension")),
                    fp(w_ben.get("pension")),
                    f"JobSeeker {fp(f_ben.get('jobseeker'))} / {fp(w_ben.get('jobseeker'))}；覆盖率可重叠",
                    "benefit_type_inc.pension",
                ),
                (
                    "近 3 月工资下降 >10%",
                    fp(pf.m["wage_3m6m_decline"]),
                    fp(pw.m["wage_3m6m_decline"]),
                    f"工资中位 {fm(pf.m['wage_median_m'])} / {fm(pw.m['wage_median_m'])}",
                    "wage_3m6m_decline",
                ),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section="01 · 收入",
            kicker="Income · 01",
            title="收入水平接近，差异在来源结构：WageGo 更靠工资，Fundo 更靠福利",
            body=income_body,
        )
    )

    # 02 负债
    liability_rows = [
        [
            '有借贷放款<small class="var">any_disburse</small>',
            fp(pf.m["any_disburse"]),
            fp(pw.m["any_disburse"]),
            delta("any_disburse", "pct"),
        ],
        [
            '现金贷使用率<small class="var">adv_use</small>',
            fp(pf.m["adv_use"]),
            fp(pw.m["adv_use"]),
            delta("adv_use", "pct"),
        ],
        [
            '中位月还款<small class="var">rep_m_p50</small>',
            fm(pf.m["rep_m_p50"]),
            fm(pw.m["rep_m_p50"]),
            delta("rep_m_p50", "money"),
        ],
        [
            '中位月新增借款<small class="var">draw_m_p50</small>',
            fm(pf.m["draw_m_p50"]),
            fm(pw.m["draw_m_p50"]),
            delta("draw_m_p50", "money"),
        ],
        [
            '还款 /（工资+福利）中位<small class="var">profile_enrichment.recurring_p50</small>',
            fp(f_enrich["recurring_p50"]),
            fp(w_enrich["recurring_p50"]),
            delta("enrich:recurring_p50", "pct"),
        ],
        [
            '≥5 家还款机构<small class="var">profile_enrichment.multi5</small>',
            fp(f_enrich["multi5"]),
            fp(w_enrich["multi5"]),
            delta("enrich:multi5", "pct"),
        ],
        [
            '有拒付记录<small class="var">dish_any</small>',
            fp(pf.m["dish_any"]),
            fp(pw.m["dish_any"]),
            delta("dish_any", "pct"),
        ],
        [
            '未解决拒付<small class="var">dish_unres</small>',
            fp(pf.m["dish_unres"]),
            fp(pw.m["dish_unres"]),
            delta("dish_unres", "pct"),
        ],
    ]
    liability_body = ui.table_box(
        ["指标", left_name, right_name, "差异"],
        liability_rows,
        note=(
            "机构数为 182 天可识别还款关系，不等于当前在贷机构数；拒付为用户占比，不是交易笔数。"
            "还款负担以工资+福利为分母，属收入代理口径，不能直接视为可支配收入。"
        ),
        num_cols=(1, 2, 3),
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section="02 · 负债",
            kicker="Liability · 02",
            title="WageGo 的借贷依赖、还款负担与拒付暴露都显著更高",
            body=liability_body,
        )
    )

    # 03 支出
    expense_body = (
        build_compare_kpis(
            module,
            pf,
            pw,
            (left_name, right_name),
            [
                ("中位月支出", "exp_p50", "money"),
                ("固定账单 / 工资", "fixed_wage_p50", "pct"),
                ("品质消费 / 工资", "prem_wage_p50", "pct"),
                ("赌博占可识别消费", "gamb_share_cons", "pct"),
            ],
            with_vars=True,
        )
        + '<div class="duo">'
        + ui.chart_box(
            figs["expense_structure"],
            "外部 / 内部转账占比以全部借记流水为分母；其余占比分母见指标名。"
            "变量：cons_share_exp_p50 / ess_share_cons / dis_share_cons / ext_share_agg / "
            "ext_transfer_share_global / internal_transfer_share_global / debt_share_p50 / "
            "high_commit_share_p50。",
        )
        + cmp_cards(
            [
                (
                    "固定账单覆盖率",
                    fp(pf.m["fixed_bill_incidence"]),
                    fp(pw.m["fixed_bill_incidence"]),
                    f"有账单用户中位 {fm(pf.m['fixed_bill_median_pos'])} / "
                    f"{fm(pw.m['fixed_bill_median_pos'])}（6 个月）",
                    "fixed_bill_incidence",
                ),
                (
                    "固定账单 / 工资 P90",
                    fp(pf.m["fixed_wage_p90"]),
                    fp(pw.m["fixed_wage_p90"]),
                    f"中位 {fp(pf.m['fixed_wage_p50'])} / {fp(pw.m['fixed_wage_p50'])}",
                    "fixed_wage_p90",
                ),
                (
                    "品质消费渗透率",
                    fp(pf.m["premium_incidence"]),
                    fp(pw.m["premium_incidence"]),
                    f"品质消费 / 工资 P90 {fp(pf.m['prem_wage_p90'])} / {fp(pw.m['prem_wage_p90'])}",
                    "premium_incidence",
                ),
                (
                    "手续费发生率",
                    fp(pf.m["fees_incidence"]),
                    fp(pw.m["fees_incidence"]),
                    f"赌博发生率 {fp(pf.m['gamb_incidence'])} / {fp(pw.m['gamb_incidence'])}；"
                    f"手续费 / 透支 {fp(pf.m['fees_ticket_incidence'])} / {fp(pw.m['fees_ticket_incidence'])}",
                    "fees_incidence",
                ),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section="03 · 支出",
            kicker="Spending · 03",
            title="支出规模接近，WageGo 固定账单负担更低、赌博与手续费更高",
            body=expense_body,
        )
    )

    # 04 流动性
    liquidity_body = (
        build_compare_kpis(
            module,
            pf,
            pw,
            (left_name, right_name),
            [
                ("负盈余占比", "sur_neg", "pct"),
                ("缓冲 < 7 天", "buffer_lt7", "pct"),
                ("中位账户余额", "bal_p50", "money"),
                ("历史曾有负余额", "enrich:ever_negative", "pct"),
            ],
            with_vars=True,
        )
        + '<div class="duo">'
        + ui.chart_box(
            figs["cashflow"],
            "占比均以有效用户为分母；缓冲天数 = 账户余额 /（月支出 / 30）。"
            "变量：pre_debt_neg / sur_neg / fortnight_neg / savings_neg / "
            "bal_lt100 / buffer_lt7 / buffer_lt30。",
        )
        + cmp_cards(
            [
                (
                    "中位月盈余",
                    fm(pf.m["sur_p50"]),
                    fm(pw.m["sur_p50"]),
                    "两组均为负；负盈余占比见上方",
                    "sur_p50",
                ),
                (
                    "中位缓冲天数",
                    f"{fmetric(pf.m['buffer_p50'], 'num1')} 天",
                    f"{fmetric(pw.m['buffer_p50'], 'num1')} 天",
                    "缓冲 = 余额 /（月支出 / 30）",
                    "buffer_p50",
                ),
                (
                    "重压型占比",
                    fp(sv(pf, "重压型", "share")),
                    fp(sv(pw, "重压型", "share")),
                    f"盈余型仅 {fp(sv(pf, '盈余型', 'share'))} / {fp(sv(pw, '盈余型', 'share'))}；"
                    "分层含新增借款净流量",
                    "seg[重压型].share",
                ),
                (
                    "快照非负仍有历史压力",
                    fp(f_enrich["hidden_share_positive"]),
                    fp(w_enrich["hidden_share_positive"]),
                    "历史负余额占比 ≥10%；分母为快照非负用户",
                    "profile_enrichment.hidden_share_positive",
                ),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section="04 · 流动性",
            kicker="Liquidity · 04",
            title="负盈余与薄缓冲是共同底盘，WageGo 的账户余额更薄",
            body=liquidity_body,
        )
    )

    return ui.deck_document(
        title=f"{left_name} vs {right_name} 人群画像对比 · 4 页版",
        slides=slides,
        plotly_js=get_plotlyjs(),
        extra_css=ui.CONCISE_CSS,
    )


def build_concise_swap_deck(
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
    fc = module.POP_COLORS[left_name]
    wc = module.POP_COLORS[right_name]

    def delta(key, kind):
        return module.fmt_delta(mv(pf, key), mv(pw, key), kind)

    def cmp_cards(cards):
        inner = ""
        for label, fv, wv, note, var in cards:
            inner += (
                '<div class="side-card cmp">'
                f'<div class="k"><span>{label}</span><code class="var">{var}</code></div>'
                f'<div class="prow"><span class="who"><i style="background:{fc}"></i>{left_name}</span>'
                f"<b>{fv}</b></div>"
                f'<div class="prow"><span class="who"><i style="background:{wc}"></i>{right_name}</span>'
                f"<b>{wv}</b></div>"
                f"<p>{note}</p></div>"
            )
        return f'<div class="side-stack reveal d2">{inner}</div>'

    n_note = f"n = {left_name} {pf.m['n_valid']:,} / {right_name} {pw.m['n_valid']:,}"
    slides: list[str] = []

    # 01 收入
    slides.append(
        ui.slide_content(
            brand=brand,
            section="01 · 收入",
            kicker="Income · 01",
            title="Swap Out 收入更高、工资覆盖更完整，Swap In 更依赖其他入账",
            body=build_compare_kpis(
                module,
                pf,
                pw,
                (left_name, right_name),
                [
                    ("中位月收入", "inc_p50", "money"),
                    ("工资覆盖率", "wage_incidence", "pct"),
                    ("福利 / 入账中位", "cent_share_agg", "pct"),
                    ("可识别雇主", "employer_share", "pct"),
                ],
                with_vars=True,
            )
            + '<div class="duo">'
            + ui.chart_box(
                figs["income_mix"],
                f"{n_note}；发生率以有效用户为分母，金额占比以分项收入加总为分母。"
                "变量：wage_incidence / cent_incidence / other_incidence / "
                "wage_share_agg / cent_share_agg / other_share_agg。",
            )
            + cmp_cards(
                [
                    (
                        "工资占入账",
                        fp(pf.m["wage_share_agg"]),
                        fp(pw.m["wage_share_agg"]),
                        f"其他入账 {fp(pf.m['other_share_agg'])} / {fp(pw.m['other_share_agg'])}",
                        "wage_share_agg",
                    ),
                    (
                        "收入波动 CV 中位",
                        fmetric(pf.m["income_cv_p50"], "num"),
                        fmetric(pw.m["income_cv_p50"], "num"),
                        f"收入来源中位 {fmetric(pf.m['income_source_p50'], 'num0')} / "
                        f"{fmetric(pw.m['income_source_p50'], 'num0')} 个",
                        "income_cv_p50",
                    ),
                    (
                        "近 3 月工资下降 >10%",
                        fp(pf.m["wage_3m6m_decline"]),
                        fp(pw.m["wage_3m6m_decline"]),
                        f"工资中位 {fm(pf.m['wage_median_m'])} / {fm(pw.m['wage_median_m'])}",
                        "wage_3m6m_decline",
                    ),
                    (
                        "福利中位（有福利）",
                        fm(pf.m["cent_median_m"]),
                        fm(pw.m["cent_median_m"]),
                        f"福利覆盖 {fp(pf.m['cent_incidence'])} / {fp(pw.m['cent_incidence'])}",
                        "cent_median_m",
                    ),
                ]
            )
            + "</div>",
        )
    )

    # 02 负债
    liability_rows = [
        [
            '有借贷放款<small class="var">any_disburse</small>',
            fp(pf.m["any_disburse"]),
            fp(pw.m["any_disburse"]),
            delta("any_disburse", "pct"),
        ],
        [
            '现金贷使用率<small class="var">adv_use</small>',
            fp(pf.m["adv_use"]),
            fp(pw.m["adv_use"]),
            delta("adv_use", "pct"),
        ],
        [
            '中位月还款<small class="var">rep_m_p50</small>',
            fm(pf.m["rep_m_p50"]),
            fm(pw.m["rep_m_p50"]),
            delta("rep_m_p50", "money"),
        ],
        [
            '中位月新增借款<small class="var">draw_m_p50</small>',
            fm(pf.m["draw_m_p50"]),
            fm(pw.m["draw_m_p50"]),
            delta("draw_m_p50", "money"),
        ],
        [
            '偿债 / 收入中位<small class="var">rep_rate_p50</small>',
            fp(pf.m["rep_rate_p50"]),
            fp(pw.m["rep_rate_p50"]),
            delta("rep_rate_p50", "pct"),
        ],
        [
            '还款机构数中位<small class="var">lender_cnt_p50</small>',
            fmetric(pf.m["lender_cnt_p50"], "num0"),
            fmetric(pw.m["lender_cnt_p50"], "num0"),
            delta("lender_cnt_p50", "num0"),
        ],
        [
            '有拒付记录<small class="var">dish_any</small>',
            fp(pf.m["dish_any"]),
            fp(pw.m["dish_any"]),
            delta("dish_any", "pct"),
        ],
        [
            '未解决拒付<small class="var">dish_unres</small>',
            fp(pf.m["dish_unres"]),
            fp(pw.m["dish_unres"]),
            delta("dish_unres", "pct"),
        ],
    ]
    slides.append(
        ui.slide_content(
            brand=brand,
            section="02 · 负债",
            kicker="Liability · 02",
            title="Swap Out 借贷渗透与还款负担更高，两组拒付暴露都在高位",
            body=ui.table_box(
                ["指标", left_name, right_name, "差异"],
                liability_rows,
                note=(
                    "机构数为 182 天可识别还款关系，不等于当前在贷机构数；拒付为用户占比，不是交易笔数。"
                    "偿债/收入以现金流收入为分母，属描述性口径。"
                ),
                num_cols=(1, 2, 3),
            ),
        )
    )

    # 03 支出
    slides.append(
        ui.slide_content(
            brand=brand,
            section="03 · 支出",
            kicker="Spending · 03",
            title="Swap Out 支出更高、必需品占比更高；Swap In 赌博占比更高",
            body=build_compare_kpis(
                module,
                pf,
                pw,
                (left_name, right_name),
                [
                    ("中位月支出", "exp_p50", "money"),
                    ("可识别消费 / 总支出", "cons_share_exp_p50", "pct"),
                    ("必需品 / 消费", "ess_share_cons", "pct"),
                    ("赌博占可识别消费", "gamb_share_cons", "pct"),
                ],
                with_vars=True,
            )
            + '<div class="duo">'
            + ui.chart_box(
                figs["expense_structure"],
                "外部 / 内部转账占比以全部借记流水为分母；其余占比分母见指标名。"
                "变量：cons_share_exp_p50 / ess_share_cons / dis_share_cons / ext_share_agg / "
                "ext_transfer_share_global / internal_transfer_share_global / debt_share_p50 / "
                "high_commit_share_p50。",
            )
            + cmp_cards(
                [
                    (
                        "外部转账 / 总支出",
                        fp(pf.m["ext_share_agg"]),
                        fp(pw.m["ext_share_agg"]),
                        f"内部转账 {fp(pf.m['internal_transfer_share_global'])} / "
                        f"{fp(pw.m['internal_transfer_share_global'])}",
                        "ext_share_agg",
                    ),
                    (
                        "手续费发生率",
                        fp(pf.m["fees_incidence"]),
                        fp(pw.m["fees_incidence"]),
                        f"手续费 / 透支 {fp(pf.m['fees_ticket_incidence'])} / "
                        f"{fp(pw.m['fees_ticket_incidence'])}",
                        "fees_incidence",
                    ),
                    (
                        "赌博发生率",
                        fp(pf.m["gamb_incidence"]),
                        fp(pw.m["gamb_incidence"]),
                        f"赌博中位金额 {fm(pf.m['gamb_median_pos'])} / {fm(pw.m['gamb_median_pos'])}",
                        "gamb_incidence",
                    ),
                    (
                        "债务相关支出中位",
                        fp(pf.m["debt_share_p50"]),
                        fp(pw.m["debt_share_p50"]),
                        f"高承诺支出 / 收入 {fp(pf.m['high_commit_share_p50'])} / "
                        f"{fp(pw.m['high_commit_share_p50'])}",
                        "debt_share_p50",
                    ),
                ]
            )
            + "</div>",
        )
    )

    # 04 流动性
    slides.append(
        ui.slide_content(
            brand=brand,
            section="04 · 流动性",
            kicker="Liquidity · 04",
            title="负盈余与薄缓冲是共同底盘，Swap In 的重压型占比更高",
            body=build_compare_kpis(
                module,
                pf,
                pw,
                (left_name, right_name),
                [
                    ("负盈余占比", "sur_neg", "pct"),
                    ("缓冲 < 7 天", "buffer_lt7", "pct"),
                    ("中位账户余额", "bal_p50", "money"),
                    ("重压型占比", "seg:重压型:share", "pct"),
                ],
                with_vars=True,
            )
            + '<div class="duo">'
            + ui.chart_box(
                figs["cashflow"],
                "占比均以有效用户为分母；缓冲天数 = 账户余额 /（月支出 / 30）。"
                "变量：pre_debt_neg / sur_neg / fortnight_neg / savings_neg / "
                "bal_lt100 / buffer_lt7 / buffer_lt30。",
            )
            + cmp_cards(
                [
                    (
                        "中位月盈余",
                        fm(pf.m["sur_p50"]),
                        fm(pw.m["sur_p50"]),
                        "两组均为负；负盈余占比见上方",
                        "sur_p50",
                    ),
                    (
                        "中位缓冲天数",
                        f"{fmetric(pf.m['buffer_p50'], 'num1')} 天",
                        f"{fmetric(pw.m['buffer_p50'], 'num1')} 天",
                        "缓冲 = 余额 /（月支出 / 30）",
                        "buffer_p50",
                    ),
                    (
                        "双周盈余为负",
                        fp(pf.m["fortnight_neg"]),
                        fp(pw.m["fortnight_neg"]),
                        f"储蓄率为负 {fp(pf.m['savings_neg'])} / {fp(pw.m['savings_neg'])}",
                        "fortnight_neg",
                    ),
                    (
                        "盈余型占比",
                        fp(sv(pf, "盈余型", "share")),
                        fp(sv(pw, "盈余型", "share")),
                        f"平衡型 {fp(sv(pf, '平衡型', 'share'))} / {fp(sv(pw, '平衡型', 'share'))}；"
                        "分层含新增借款净流量",
                        "seg[盈余型].share",
                    ),
                ]
            )
            + "</div>",
        )
    )

    return ui.deck_document(
        title=f"{left_name} vs {right_name} 人群画像对比 · 4 页版",
        slides=slides,
        plotly_js=get_plotlyjs(),
        extra_css=ui.CONCISE_CSS,
    )


def build_pair_deck(
    pair_key: str,
    pair: dict,
    module,
    pf,
    pw,
    figs: dict[str, str],
    generated_at: str,
) -> str:
    if pair_key == "fundo_wagego":
        return build_concise_pair_deck(pair, module, pf, pw, figs, generated_at)
    if pair_key == "swap_in_out":
        return build_concise_swap_deck(pair, module, pf, pw, figs, generated_at)

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
    enhanced = pair_key == "fundo_wagego" and all("profile_enrichment" in p.m for p in (pf,pw))
    extras = compare_extra.slides(pair,pf,pw) if enhanced else {}
    has_employment = "employment" in extras
    f_enrich = pf.m.get("profile_enrichment", {})
    w_enrich = pw.m.get("profile_enrichment", {})
    k_liability = "2.4" if deep else "2.2"
    k_cashflow = "2.5" if deep else "2.3"
    k_segments = "2.6" if deep else "2.4"
    k_seg_detail = "2.7" if deep else "2.4"
    k_risk = "2.8" if deep else "2.5"
    k_diff = "2.9" if deep else "2.6"
    findings = [f for f in module.build_findings(pf, pw) if f]
    income_gap = pw.m["inc_p50"] - pf.m["inc_p50"]

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
                f"使用同一套特征口径对比 {left_name} 与 {right_name} 两组样本，" +
                ("覆盖收入来源、偿债负担、多头还款、近期资金变化、历史余额与消费净流出；新增指标在两组使用相同公式与阈值。" if enhanced else f"覆盖收入、支出、负债、现金流、用户分层与风险信号，共用 {len(pf.m)} 项指标口径。")
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
            title="收入接近，差异集中在负债与履约" if enhanced else "两组人群，同一套口径",
            sub=(
                f"两组的收入中位仅差 {fm(abs(income_gap))}（{abs(income_gap)/pf.m['inc_p50']:.0%}），"
                "但 WageGo 的偿债负担、拒付暴露与多头关系明显更高；先看结构差异，不把总体差距归因于收入。"
                if enhanced else findings[0] if findings else "两组样本的收入、支出、负债与风险信号对比。"
            ),
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
            title="WageGo 样本更大，有效率也更高" if enhanced else "样本概览",
            sub=(
                f"Fundo {pf.m['n_valid']:,} 人、有效率 {fp(pf.m['coverage'])}；"
                f"WageGo {pw.m['n_valid']:,} 人、{fp(pw.m['coverage'])}。规模与月份构成不同，后续差异需保留这一背景。"
                if enhanced else "两组样本规模与月份构成不同，后续指标差异需要放在这一背景下解读。"
            ),
            body=overview,
        )
    )

    if enhanced:
        slides.append(extras["scope"])

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
            title="收入与支出接近，盈余普遍为负" if enhanced else "核心财务指标分布",
            sub=(
                f"两组中位收入相差 {fm(abs(income_gap))}、中位支出相差 {fm(abs(pw.m['exp_p50']-pf.m['exp_p50']))}；"
                f"负盈余占比 {fp(pf.m['sur_neg'])} / {fp(pw.m['sur_neg'])}，余额中位 {fm(pf.m['bal_p50'])} / {fm(pw.m['bal_p50'])}，缓冲都很薄。"
                if enhanced else "两组收入与支出分布接近，盈余整体位于零以下，余额集中在低位。"
            ),
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
                ("可识别雇主", pair_value(fp(mv(pf, "employer_share")), fp(mv(pw, "employer_share"))), f"主要行业码（占已分类）：{pf.m['industry_top_text']} / {pw.m['industry_top_text']}；WageGo 高度集中，行业结构不直接对比"),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_a,
            kicker="1.3 · Income & Employment",
            title="工资覆盖更高，但七成入账来自其他来源" if enhanced else "收入与就业",
            sub=(
                f"WageGo 工资覆盖 {fp(pw.m['wage_incidence'])}、Fundo {fp(pf.m['wage_incidence'])}；"
                f"两组其他入账都约占 {fp(pf.m['other_share_agg'])}，收入波动 CV {fmetric(pf.m['income_cv_p50'], 'num')} / {fmetric(pw.m['income_cv_p50'], 'num')}。"
                if enhanced else "收入水平接近，工资覆盖面与收入结构存在差异；两组入账波动都偏高。"
            ),
            body=income_body,
        )
    )

    # 07 Income resilience
    if has_employment:
        slides.append(extras["employment"])

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
                kicker=("1.5" if has_employment else "1.4") + " · Income Resilience",
                title="Fundo 更依赖福利，WageGo 更依赖工资" if enhanced else "收入结构与韧性",
                sub=(
                    f"福利覆盖 {fp(pf.m['cent_incidence'])} / {fp(pw.m['cent_incidence'])}，"
                    f"福利/入账中位 {fp(pf.m['cent_share_rec_p50'])} / {fp(pw.m['cent_share_rec_p50'])}；"
                    f"重压型工资占比反而最低（{fp(sv(pf, '重压型', 'wage_share'))} / {fp(sv(pw, '重压型', 'wage_share'))}），压力组更依赖非工资入账。"
                    if enhanced else
                    f"{left_name} 福利覆盖更高、福利依赖更强；{right_name} 工资覆盖更广，"
                    "但重压型同样明显依赖非工资入账。"
                ),
                body=resilience_body,
            )
        )

    if enhanced:
        slides.append(extras["capacity"])

    # 08 Divider 02
    slides.append(
        ui.slide_divider(
            brand=brand,
            section=section_b,
            num="02",
            kicker="Expense, Liability & Liquidity",
            title="支出、负债与流动性",
            desc=(
                "再看资金去向与周转方式：消费深度、固定账单与工资负担、借贷使用与偿债压力、账户缓冲厚度。"
                if deep
                else "再看资金去向与周转方式：支出刚性与转账流向、借贷使用与偿债压力、账户缓冲厚度。"
            ),
            status_html='<span class="status amber"><i></i>结构差异集中在负债与风险端</span>',
            meta=(
                [
                    (fp(pf.m["fixed_wage_p50"]), f"{left_name} 固定账单 / 工资"),
                    (fp(pw.m["fixed_wage_p50"]), f"{right_name} 固定账单 / 工资"),
                    (fp(pw.m["external_lender_incidence"]), f"{right_name} 选定机构覆盖"),
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
            title="支出规模接近，资金流向差异更大" if enhanced else "支出与消费",
            sub=(
                f"两组中位支出仅差 {fm(abs(pw.m['exp_p50']-pf.m['exp_p50']))}；"
                "差异主要在外部/内部转账结构与赌博类消费，而不是总支出水平。"
                if enhanced else "支出规模接近，但资金流向不同：外部/内部转账结构与赌博类消费是主要差异。"
            ),
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
                title="品质消费渗透接近，负担差异受工资影响" if enhanced else "消费深度与品质消费",
                sub=(
                    f"品质消费渗透率 {fp(pf.m['premium_incidence'])} / {fp(pw.m['premium_incidence'])}，"
                    f"但品质消费/工资中位 {fp(pf.m['prem_wage_p50'])} / {fp(pw.m['prem_wage_p50'])}、"
                    f"P90 {fp(pf.m['prem_wage_p90'])} / {fp(pw.m['prem_wage_p90'])}；工资分母差异会放大比率差距。"
                    if enhanced else
                    "比较类目覆盖与相对工资的负担；比率差异可能受工资分母影响，不直接等同于消费水平或可压缩空间。"
                ),
                body=cons_depth_body,
            )
        )

    if enhanced:
        slides.append(extras["gambling"])

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
                        "标签收入 / 标签支出中位",
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
                title="账单覆盖接近，工资负担与尾部差异更大" if enhanced else "刚性支出与工资负担",
                sub=(
                    f"覆盖率 {fp(pf.m['fixed_bill_incidence'])} / {fp(pw.m['fixed_bill_incidence'])}，"
                    f"固定账单/工资中位 {fp(pf.m['fixed_wage_p50'])} / {fp(pw.m['fixed_wage_p50'])}、"
                    f"P90 {fp(pf.m['fixed_wage_p90'])} / {fp(pw.m['fixed_wage_p90'])}；"
                    f"重压型负担 {fp(sv(pf, '重压型', 'fixed_wage'))} / {fp(sv(pw, '重压型', 'fixed_wage'))}。"
                    "标签收支比属另一体系，不能据此推算可支配收入。"
                    if enhanced else
                    "分别比较账单覆盖率与相对工资的金额负担；标签收支比属于另一统计体系，"
                    "不用于解释现金流盈余，也不能直接推算真实可支配收入。"
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
            f" · 选定机构覆盖 {fp(pf.m['external_lender_incidence'])} / "
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
            title="借贷使用与拒付暴露是最大分化维度" if enhanced else "负债与信用履约",
            sub=(
                f"有借贷放款 {fp(pf.m['any_disburse'])} / {fp(pw.m['any_disburse'])}，"
                f"偿债/收入 {fp(pf.m['rep_rate_p50'])} / {fp(pw.m['rep_rate_p50'])}，"
                f"有拒付记录 {fp(pf.m['dish_any'])} / {fp(pw.m['dish_any'])}；WageGo 的借贷与履约暴露更高。"
                if enhanced else "借贷使用、偿债负担与拒付暴露是两组分化最明显的维度之一。"
            ),
            body=liability_body,
        )
    )

    if enhanced:
        slides.append(extras["multi"])

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
            title="两组都以负盈余为主、缓冲极薄" if enhanced else "现金流与流动性",
            sub=(
                f"负盈余 {fp(pf.m['sur_neg'])} / {fp(pw.m['sur_neg'])}，"
                f"缓冲<7 天 {fp(pf.m['buffer_lt7'])} / {fp(pw.m['buffer_lt7'])}；"
                "流动性是共同底盘，不是区分品牌的主轴。"
                if enhanced else "两组都以负盈余为主、账户缓冲极薄；流动性是共同底盘，而非区分两组的主轴。"
            ),
            body=cashflow_body,
        )
    )

    if enhanced:
        slides.extend([extras["recent"],extras["balance"]])

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
            title="两组都以赤字型和重压型为主" if enhanced else "用户分层",
            sub=(
                f"Fundo 重压型 {fp(sv(pf, '重压型', 'share'))}、WageGo {fp(sv(pw, '重压型', 'share'))}；"
                f"盈余型仅 {fp(sv(pf, '盈余型', 'share'))} / {fp(sv(pw, '盈余型', 'share'))}。"
                "分层按含借款净流量，不等于信用等级。"
                if enhanced else "按含新增借款的盈余率分层：盈余型 ≥5%、平衡型 -5%~5%、赤字型 -20%~-5%、重压型 <-20%；分层不等于信用等级。"
            ),
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
        ["分层", f"{left_name} 占比", f"{right_name} 占比", "差异", f"{left_name} 月盈余", f"{right_name} 月盈余", f"{left_name} 拒付用户占比", f"{right_name} 拒付用户占比"],
        seg_rows,
        note="盈余率含新增借款且截尾至 [-100%, +100%]；此表拒付占比为组内有拒付记录的用户占比，不是按交易笔数计算的拒付率。",
        num_cols=(1, 2, 3, 4, 5, 6, 7),
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_seg_detail} · Segmentation Detail",
            title="重压型拒付最高，盈余型并非最低" if enhanced else "分层指标明细",
            sub=(
                f"重压型有拒付记录 {fp(sv(pf, '重压型', 'dishon'))} / {fp(sv(pw, '重压型', 'dishon'))}，"
                f"盈余型 {fp(sv(pf, '盈余型', 'dishon'))} / {fp(sv(pw, '盈余型', 'dishon'))}；"
                "拒付占比按用户数，不是按交易笔数。分层仍不是信用等级。"
                if enhanced else "分别观察各规则分层的净流量与有拒付记录的用户占比；不把规则分层直接视为风险等级。"
            ),
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
                ("催收类支出覆盖", pair_value(fp(pf.m["collections_inc"]), fp(pw.m["collections_inc"])), f"债务整合类支出 {fp(pf.m['debt_consolidation_inc'])} / {fp(pw.m['debt_consolidation_inc'])}"),
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_risk} · Risk Signals",
            title="WageGo 的赌博、手续费与拒付暴露更高" if enhanced else "风险信号",
            sub=(
                f"赌博发生率 {fp(pf.m['gamb_incidence'])} / {fp(pw.m['gamb_incidence'])}，"
                f"手续费 {fp(pf.m['fees_incidence'])} / {fp(pw.m['fees_incidence'])}，"
                f"有拒付记录 {fp(pf.m['dish_any'])} / {fp(pw.m['dish_any'])}；这些是同期信号，不作因果判断。"
                if enhanced else "赌博、手续费与催收暴露构成收支指标之外的风险信号。"
            ),
            body=risk_body,
        )
    )

    # 14 Difference overview
    slides.append(
        ui.slide_content(
            brand=brand,
            section=section_b,
            kicker=f"{k_diff} · Key Differences",
            title="差异集中在借贷依赖与拒付暴露" if enhanced else "主要差异",
            sub=(
                f"全期未加权差值 = {right_name} − {left_name}，按百分点绝对值排序；"
                "收入水平差异较小，负债、履约与多头关系差异更大。"
                if enhanced else
                f"全期未加权差值 = {right_name} − {left_name}，单位为百分点；比例及比率指标按差值绝对值排序，各自有效分母见对应分析。"
            ),
            body=ui.chart_box(figs["difference"], "图中仅展示部分指标；完整指标见各章节明细表。"),
        )
    )

    # 15 Findings
    if enhanced:
        checks = compare_extra.prepare(pf, pw)
        common = "、".join(checks["common_months"]) or "无"
        cf, cw = checks["common"]["Fundo"], checks["common"]["WageGo"]
        condensed = [
            f"<b>两套收支口径分别核验。</b> 现金流收入/支出中位比 Fundo {f_enrich['surplus_ratio_p50']:.3f}、"
            f"WageGo {w_enrich['surplus_ratio_p50']:.3f}；标签体系 {f_enrich['tag_ratio_p50']:.3f}、"
            f"{w_enrich['tag_ratio_p50']:.3f}。不能把标签比率解释为现金流缺口。",
            f"<b>换成工资+福利分母，负担接近翻倍。</b> Fundo {fp(f_enrich['same_total_p50'])}→"
            f"{fp(f_enrich['recurring_p50'])}、WageGo {fp(w_enrich['same_total_p50'])}→"
            f"{fp(w_enrich['recurring_p50'])}；负担>30% 占 "
            f"{fp(f_enrich['burden_gt30'])} / {fp(w_enrich['burden_gt30'])}。",
            f"<b>多头覆盖与同组拒付都更高。</b> ≥5 家机构 "
            f"{fp(f_enrich['multi5'])} / {fp(w_enrich['multi5'])}；该组未解决拒付 "
            f"{fp(f_enrich['lender_rows'][-1]['unresolved'])} / {fp(w_enrich['lender_rows'][-1]['unresolved'])}，"
            f"1 家组 {fp(f_enrich['lender_rows'][1]['unresolved'])} / {fp(w_enrich['lender_rows'][1]['unresolved'])}。",
            f"<b>快照为正仍可能隐藏历史压力。</b> 历史曾有负余额 "
            f"{fp(f_enrich['ever_negative'])} / {fp(w_enrich['ever_negative'])}；"
            f"快照非负但历史负余额≥10% 占 "
            f"{fp(f_enrich['hidden_share_positive'])} / {fp(w_enrich['hidden_share_positive'])}。",
            f"<b>近期双重压力组未显示更高拒付。</b> 收入日均下降>20% 且还款上升>20% 的用户，"
            f"未解决拒付 {fp(f_enrich['double_rate'])} / {fp(w_enrich['double_rate'])}，"
            f"其余可比组 {fp(f_enrich['other_comparable_rate'])} / {fp(w_enrich['other_comparable_rate'])}。",
            f"<b>赌博净流出更高，但组内不单调。</b> 正净流出/工资福利>5% 为 "
            f"{fp(f_enrich['gamble_gt5'])} / {fp(w_enrich['gamble_gt5'])}；"
            "控制机构数后未形成一致梯度，不直接解释为拒付原因。",
            f"<b>月份对齐后结论不变，身份命名空间未核实。</b> 共同月份 {common} 下负担 "
            f"{fp(cf['burden'])} / {fp(cw['burden'])}、≥5 家机构 "
            f"{fp(cf['multi5'])} / {fp(cw['multi5'])}；两组 user_id 键重合 "
            f"{checks['user_key_overlap']:,} 个，不预设独立。",
            f"<b>账单与分层指标不能直接用于审批。</b> 固定账单/工资 "
            f"{fp(pf.m['fixed_wage_p50'])} / {fp(pw.m['fixed_wage_p50'])}；"
            f"负盈余 {fp(pf.m['sur_neg'])} / {fp(pw.m['sur_neg'])}。"
            "标签收支比、含借款净流量与催收口径均有边界，全部为同期描述。",
        ]
        slides.append(
            ui.slide_content(
                brand=brand,
                section="Findings",
                kicker="主要差异 01–08",
                title="主要差异集中在负担、多头与履约",
                sub="8 条判断：前六条是差异结论，后两条是使用边界。",
                body=ui.finding_cards(condensed),
            )
        )
    else:
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
    if enhanced:
        f,p = pf.m["profile_enrichment"],pw.m["profile_enrichment"]
        takeaways = [
            ("还款基础",f"还款/(工资+福利)中位 <b>{fp(f['recurring_p50'])}</b> / <b>{fp(p['recurring_p50'])}</b>；收入代理口径，不是核实可支配收入。"),
            ("借贷关系",f"≥5 家还款机构覆盖 <b>{fp(f['multi5'])}</b> / <b>{fp(p['multi5'])}</b>；机构数与组内拒付表现分别解读。"),
            ("历史流动性",f"历史曾有负余额 <b>{fp(f['ever_negative'])}</b> / <b>{fp(p['ever_negative'])}</b>；最近余额非负仍有历史压力的用户需单独识别。"),
            ("可比性",f"共同月份 <b>{'、'.join(compare_extra.prepare(pf,pw)['common_months'])}</b>，敏感性结果已单列；两组不预设独立，不作因果或审批优劣判断。"),
        ]
    slides.append(
        ui.slide_closing(
            brand=brand,
            kicker="结论 · Takeaway",
            title="没有一方全面占优，差异集中在结构" if enhanced else "对比结论",
            sub=(
                f"WageGo 收入更高、负盈余更少，但偿债负担、多头关系与拒付暴露更高；Fundo 相反。"
                "选择应基于目标客群与风险偏好，而不是单指标排名。"
                if enhanced else
                f"{left_name} 与 {right_name} 各有短板，没有一方全面占优；"
                "差异主要集中在收入结构、支出刚性、借贷依赖与拒付暴露。"
                if deep
                else f"{left_name} 与 {right_name} 各有短板，没有一方全面占优；"
                "差异主要集中在收入结构、借贷依赖与拒付暴露。"
            ),
            takeaways=takeaways,
            source=(
                f"<b>数据源</b>：{pair['source']}。两组共用分析字段、缺失哨兵与清洗规则，月份按各自元数据解析；"
                "以上为描述性对比，不做因果推断。"
            ),
            date=f"生成时间 {generated_at}",
        )
    )

    return ui.deck_document(
        title=f"{left_name} vs {right_name} 人群画像对比 · 演示文稿",
        slides=slides,
        plotly_js=get_plotlyjs(),
        extra_css=profile_extra.EXTRA_CSS+compare_extra.EXTRA_CSS if enhanced else "",
    )


def run_pair(pair_key: str, generated_at: str, out_dir: Path, *, sync_profiles: bool = False) -> None:
    pair = PAIRS[pair_key]
    module = pair["module"]
    left_name, left_mod, left_csv = pair["left"]
    right_name, right_mod, right_csv = pair["right"]
    print(f"[{pair_key}] loading profiles ...")
    pf = module.load_profile(left_name, left_mod, left_csv)
    pw = module.load_profile(right_name, right_mod, right_csv)
    if pair_key == "fundo_wagego":
        compare_extra.export(ROOT,pf,pw)
        if sync_profiles:
            from build_profile_decks import write_profile_deck
            write_profile_deck("fundo",generated_at,out_dir,pf.d,pf.m,pf.seg)
            write_profile_deck("wagego",generated_at,out_dir,pw.d,pw.m,pw.seg)
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
    parser.add_argument("--sync-profiles", action="store_true", help="同时从本次已加载的数据生成 Fundo 和 WageGo 单独画像")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else OUT
    generated_at = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    keys = list(PAIRS) if args.pair == "all" else [args.pair]
    for key in keys:
        run_pair(key, generated_at, out_dir, sync_profiles=args.sync_profiles)


if __name__ == "__main__":
    main()


