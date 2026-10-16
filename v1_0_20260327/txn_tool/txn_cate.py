# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import List, Union, Dict, Set
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# = 0) 核心配置（与原逻辑一致，可按需调整）
# =========================================================
WINDOWS = [7, 14, 28, 56, 84, 168, 182]   # 时间窗
CATEGORIES = [
    "All Other Credits","Automotive","Pet Care","Home Improvement","SACC Loans",
    "Department Stores","Gyms and other memberships","Overdrawn","Information","Retail",
    "Debt Collection","Utilities","Subscription TV","Entertainment","Transport","Donations",
    "Rent","Non SACC Loans","External Transfers","Fees","Debt Consolidation","Gambling",
    "Internal Transfer","Insurance","Travel","Centrelink","Dishonours","Credit Card Repayments",
    "Groceries","Personal Care","Health","Wages","Telecommunications","Dining Out","Education",
]

def _cat_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")

DEBT_CATS = {"SACC Loans", "Non SACC Loans", "Credit Card Repayments", "Debt Consolidation"}

CLUSTERS = {
    # ===== 收入/转账 =====
    "income":                 {"denom": "credit", "cats": {"Wages", "Centrelink"}},
    "income_wages":           {"denom": "credit", "cats": {"Wages"}},
    "income_benefits":        {"denom": "credit", "cats": {"Centrelink"}},
    "other_credits":          {"denom": "credit", "cats": {"All Other Credits"}},
    "transfers":              {"denom": "total",  "cats": {"Internal Transfer", "External Transfers"}},
    "transfer_internal":      {"denom": "total",  "cats": {"Internal Transfer"}},
    "transfer_external":      {"denom": "total",  "cats": {"External Transfers"}},

    # ===== 必需与家庭支出 =====
    "essentials":             {"denom": "debit",  "cats": {"Groceries", "Utilities", "Rent", "Transport", "Health"}},
    "housing":                {"denom": "debit",  "cats": {"Rent", "Home Improvement"}},
    "utilities":              {"denom": "debit",  "cats": {"Utilities", "Subscription TV", "Telecommunications"}},
    "communications":         {"denom": "debit",  "cats": {"Telecommunications", "Information"}},
    "healthcare":             {"denom": "debit",  "cats": {"Health", "Insurance"}},
    "education":              {"denom": "debit",  "cats": {"Education"}},
    "transport":              {"denom": "debit",  "cats": {"Transport", "Automotive"}},

    # 细分必需项
    "household_daily":        {"denom": "debit",  "cats": {"Groceries", "Personal Care", "Pet Care"}},
    "recurring_bills":        {"denom": "debit",  "cats": {"Utilities", "Telecommunications", "Subscription TV", "Insurance", "Rent"}},
    "memberships_subs":       {"denom": "debit",  "cats": {"Gyms and other memberships", "Subscription TV"}},

    # ===== 可选/非必需消费 =====
    "dining":                 {"denom": "debit",  "cats": {"Dining Out"}},
    "entertainment":          {"denom": "debit",  "cats": {"Entertainment"}},
    "retail":                 {"denom": "debit",  "cats": {"Retail", "Department Stores"}},
    "travel":                 {"denom": "debit",  "cats": {"Travel"}},
    "personal_care":          {"denom": "debit",  "cats": {"Personal Care"}},
    "pet":                    {"denom": "debit",  "cats": {"Pet Care"}},
    "donations":              {"denom": "debit",  "cats": {"Donations"}},

    # 汇总“非必需”
    "discretionary":          {"denom": "debit",  "cats": {"Dining Out", "Entertainment", "Travel", "Retail", "Department Stores", "Personal Care", "Pet Care", "Donations"}},

    # ===== 债务/费用/催收/赌博 =====
    "debt":                   {"denom": "debit",  "cats": set(DEBT_CATS)},
    "debt_sacc":              {"denom": "debit",  "cats": {"SACC Loans"}},
    "debt_non_sacc":          {"denom": "debit",  "cats": {"Non SACC Loans"}},
    "debt_cc_repayments":     {"denom": "debit",  "cats": {"Credit Card Repayments"}},
    "debt_consolidation":     {"denom": "debit",  "cats": {"Debt Consolidation"}},

    "fees_overdraft":         {"denom": "debit",  "cats": {"Fees", "Overdrawn", "Dishonours"}},
    "collections":            {"denom": "debit",  "cats": {"Debt Collection"}},
    "gambling":               {"denom": "debit",  "cats": {"Gambling"}},

    # 风险压力打包 + 高波动非必需
    "risk_obligations":       {"denom": "debit",  "cats": {"SACC Loans", "Non SACC Loans", "Centrelink", "Debt Consolidation", "Overdrawn", "Dishonours", "Debt Collection"}},
    "risk_discretionary":     {"denom": "debit",  "cats": {"Gambling", "Entertainment", "Dining Out", "Travel", "Retail", "Department Stores"}},
}

RISK_CLUSTERS_FOR_MOMENTUM = [
    "debt", "fees_overdraft", "collections", "gambling",
    "risk_obligations", "risk_discretionary",
    "dining", "entertainment", "travel", "retail"
]

# =========================================================
# = 1) 依赖 & 分类显式映射（列名全部改为 bank_txn_category_* 前缀）
# =========================================================
def expected_dependent_vars() -> List[str]:
    out = []
    # 无窗全局
    out += [
        "bank_txn_category_global_cnt",
        "bank_txn_category_global_credit_cnt",
        "bank_txn_category_global_debit_cnt",
        "bank_txn_category_global_amt",
        "bank_txn_category_global_credit_amt",
        "bank_txn_category_global_debit_amt",
    ]
    for W in WINDOWS:
        # 全局
        out += [
            f"bank_txn_category_global_cnt_{W}d",
            f"bank_txn_category_global_credit_cnt_{W}d",
            f"bank_txn_category_global_debit_cnt_{W}d",
            f"bank_txn_category_global_amt_{W}d",
            f"bank_txn_category_global_credit_amt_{W}d",
            f"bank_txn_category_global_debit_amt_{W}d",
        ]
        # 各银行分类（total/credit/debit 的 cnt/amt）
        for cat in CATEGORIES:
            ck = _cat_key(cat)
            out += [
                f"bank_txn_category_{ck}_cnt_{W}d",
                f"bank_txn_category_{ck}_credit_cnt_{W}d",
                f"bank_txn_category_{ck}_debit_cnt_{W}d",
                f"bank_txn_category_{ck}_amt_{W}d",
                f"bank_txn_category_{ck}_credit_amt_{W}d",
                f"bank_txn_category_{ck}_debit_amt_{W}d",
            ]
    return out

OPTIONAL_CATEGORY_NAMES = [
    "global_structure", "tickets", "category_share", "category_ticket",
    "cluster_share", "cluster_ticket", "cluster_momentum",
    "activity", "inter_txn_gap", "income_periodicity", "rent_lag", "netflow_streaks",
]

def build_category_map() -> Dict[str, List[str]]:
    M: Dict[str, List[str]] = {k: [] for k in OPTIONAL_CATEGORY_NAMES}
    # 1) global_structure & tickets（基于全局聚合）
    for W in WINDOWS:
        M["global_structure"] += [
            f"bank_txn_category_global_income_share_{W}d",
            f"bank_txn_category_global_debit_share_{W}d",
            f"bank_txn_category_global_burn_rate_{W}d",
            f"bank_txn_category_global_saving_rate_{W}d",
        ]
        M["tickets"] += [
            f"bank_txn_category_global_income_avg_ticket_{W}d",
            f"bank_txn_category_global_expense_avg_ticket_{W}d",
        ]
    # 2) category_*（每个银行分类）
    for W in WINDOWS:
        for cat in CATEGORIES:
            ck = _cat_key(cat)
            M["category_share"] += [f"bank_txn_category_{ck}_share_{W}d"]
            M["category_ticket"] += [f"bank_txn_category_{ck}_avg_ticket_{W}d"]
    # 3) cluster_*（基于 CLUSTERS）
    for W in WINDOWS:
        for cls_key in CLUSTERS.keys():
            M["cluster_share"] += [f"bank_txn_category_cluster_{cls_key}_share_{W}d"]
            M["cluster_ticket"] += [f"bank_txn_category_cluster_{cls_key}_avg_ticket_{W}d"]
    # 4) cluster_momentum（跨窗 delta/accel）
    for cls_key in RISK_CLUSTERS_FOR_MOMENTUM:
        for i in range(len(WINDOWS) - 1):
            w1, w2 = WINDOWS[i], WINDOWS[i + 1]
            M["cluster_momentum"] += [f"bank_txn_category_cluster_{cls_key}_share_delta_{w1}v{w2}"]
        for i in range(len(WINDOWS) - 2):
            w1, w2, w3 = WINDOWS[i], WINDOWS[i + 1], WINDOWS[i + 2]
            M["cluster_momentum"] += [f"bank_txn_category_cluster_{cls_key}_share_accel_{w1}_{w2}_{w3}"]
    # 5) 行为/时间特征（L3/L4 产出）
    for W in WINDOWS:
        M["activity"] += [
            f"bank_txn_category_active_days_{W}d",
            f"bank_txn_category_txn_freq_per_active_day_{W}d",
            f"bank_txn_category_weekend_share_{W}d",
        ]
        M["inter_txn_gap"] += [
            f"bank_txn_category_inter_txn_gap_p50_{W}d",
            f"bank_txn_category_inter_txn_gap_p90_{W}d",
        ]
        M["income_periodicity"] += [
            f"bank_txn_category_income_dom_std_{W}d",
            f"bank_txn_category_income_gap_mean_{W}d",
            f"bank_txn_category_income_gap_std_{W}d",
        ]
        M["rent_lag"] += [
            f"bank_txn_category_rent_after_wage_lag_p50_{W}d",
        ]
        M["netflow_streaks"] += [
            f"bank_txn_category_pos_net_streak_days_{W}d",
            f"bank_txn_category_neg_net_streak_days_{W}d",
        ]
    for k in M:
        M[k] = sorted(list(dict.fromkeys(M[k])))
    return M

DEPENDENT_VARS: List[str] = expected_dependent_vars()
CATEGORY_MAP: Dict[str, List[str]] = build_category_map()
EXPOSE_DEPENDENT = True  # 是否在最终返回中展示依赖变量列

# =========================================================
# = 2) 小工具
# =========================================================
_EPS = 1e-9

def _safe_div(num, den):
    return np.where(np.abs(den) < _EPS, 0.0, num / den)

def _pctl(a: np.ndarray, q: float):
    if a.size == 0:
        return np.nan
    return float(np.nanpercentile(a, q))

def _longest_streak(bool_arr: np.ndarray) -> int:
    if bool_arr.size == 0:
        return 0
    max_run = run = 0
    for v in np.array(bool_arr, dtype=int):
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return int(max_run)

def _weighted_nanmean(values, weights):
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    mask = (~np.isnan(values)) & (~np.isnan(weights)) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))

def list_available_categories() -> List[str]:
    return sorted(list(CATEGORY_MAP.keys()))

def _normalize_categories(categories: Union[str, List[str], None]) -> Set[str]:
    if categories is None:
        return set(list_available_categories())  # 全选
    if isinstance(categories, str):
        c = categories.strip()
        if c.upper() == "ALL":
            return set(list_available_categories())
        chosen = {c}
    else:
        chosen = set([str(x).strip() for x in categories])
    if len(chosen) == 0:
        return set()
    known = set(list_available_categories())
    unknown = chosen - known
    if unknown:
        raise ValueError(
            f"未知分类名: {sorted(list(unknown))}\n"
            f"可选分类为: {sorted(list(known))}"
        )
    return chosen

# =========================================================
# = 3) 数据准备（这里兼容 flowTime & 自动加工 trac_days）
# =========================================================
def _parse_txn_date_series(s: pd.Series) -> pd.Series:
    """
    尝试两轮解析交易日期列（如 '12/23/2024' / '23/12/2024' 均可）。
    返回：按天的日期（去时分秒），失败则 NaT。
    """
    s1 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    mask_fail = s1.isna() & s.notna()
    if mask_fail.any():
        s2 = pd.to_datetime(s[mask_fail], errors="coerce", dayfirst=True)
        s1.loc[mask_fail] = s2
    return s1.dt.floor("D")

def _prepare(df: pd.DataFrame, account_type_mode: str = "with_blank") -> pd.DataFrame:
    """
    必含列（经过兼容与派生后）：user_id, sample_datetime, amount, dr_cr, category, trac_days
    兼容：
      - flowTime -> sample_datetime
      - 若缺少 trac_days，则由 sample_datetime 与 transaction_date 自动计算（按天差）
    说明：
      - 为保持既有窗口筛选语义，无法可靠计算的 trac_days 统一填充 1e9
      - 若交易日期晚于 sample_datetime，视作异常，也置为 1e9（避免被纳入任何窗口）
    """
    ALLOWED_ACCOUNT_TYPES = ["transaction", "investments", "savings", "trading", "Unlabeled"]

    if "account_type" not in df.columns:
        raise ValueError("输入缺少必要列: ['account_type']")

    if account_type_mode == "strict":
        df = df.loc[df["account_type"].isin(ALLOWED_ACCOUNT_TYPES)].copy()
    elif account_type_mode == "with_blank":
        df = df.loc[
            df["account_type"].isin(ALLOWED_ACCOUNT_TYPES)
            | df["account_type"].isna()
            | (df["account_type"] == "")
        ].copy()
    elif account_type_mode == "all":
        df = df.copy()
    else:
        raise ValueError("account_type_mode 只能是: 'strict', 'with_blank', 'all'")

    if "flowTime" in df.columns and "sample_datetime" not in df.columns:
        df = df.rename(columns={"flowTime": "sample_datetime"})

    need_base = ["user_id", "sample_datetime", "amount", "dr_cr", "category"]
    miss_base = [c for c in need_base if c not in df.columns]
    if miss_base:
        raise ValueError(f"输入缺少必要列: {miss_base}")

    g = df.copy()

    # 多卡场景：若缺失 bank_account_id，补默认值，保证老数据兼容
    if "bank_account_id" not in g.columns:
        g["bank_account_id"] = "__single_account__"
    g["bank_account_id"] = g["bank_account_id"].astype(str)
    g.loc[g["bank_account_id"].isna() | (g["bank_account_id"].str.strip() == ""), "bank_account_id"] = "__missing_account__"

    g["amount"] = pd.to_numeric(g["amount"], errors="coerce").fillna(0.0).abs()
    g["dr_cr"] = g["dr_cr"].astype(str).str.lower().str.strip()

    sdt = pd.to_datetime(
        g["sample_datetime"],
        errors="coerce",
        utc=True,
        format=None,
    ).dt.tz_convert(None)
    g["sample_datetime"] = sdt

    txn_date = pd.to_datetime(
        g["transaction_date"],
        errors="coerce",
        format=None,
    )

    sdt_day = g["sample_datetime"].dt.floor("D")
    txn_day = txn_date.dt.floor("D")
    days = (sdt_day - txn_day).dt.days.astype("float64")
    days = days.where((txn_day.notna()) & (days >= 0), np.nan)
    g["trac_days"] = pd.to_numeric(days, errors="coerce").fillna(10 ** 9)

    if not pd.api.types.is_categorical_dtype(g["category"]):
        g["category"] = g["category"].astype("category")
        try:
            g["category"] = g["category"].cat.set_categories(CATEGORIES)
        except Exception:
            pass

    g["_is_credit"] = (g["dr_cr"] == "credit").astype("int8")
    g["_is_debit"] = (g["dr_cr"] == "debit").astype("int8")
    g["_amt_total"] = g["amount"].astype("float64")
    g["_amt_debit_abs"] = g["amount"].astype("float64")
    g["_amt_credit"] = g["_amt_total"] * g["_is_credit"].astype("float64")
    g["_amt_debit"] = g["_amt_debit_abs"] * g["_is_debit"].astype("float64")
    g["_txn_date"] = g["sample_datetime"] - pd.to_timedelta(g["trac_days"], unit="D")
    g["_txn_date"] = pd.to_datetime(g["_txn_date"], errors="coerce").dt.floor("D")
    g["_dow"] = g["_txn_date"].dt.weekday
    g["_dom"] = g["_txn_date"].dt.day
    return g

# =========================================================
# = 4) L0 依赖层（总是全量计算，列名为 bank_txn_category_*）
# =========================================================
def _ensure_cat_columns(df: pd.DataFrame, W: int, kind: str):
    cols_needed = []
    for cat in CATEGORIES:
        ck = _cat_key(cat)
        if kind == "cnt_total":
            cols_needed.append(f"bank_txn_category_{ck}_cnt_{W}d")
        elif kind == "cnt_credit":
            cols_needed.append(f"bank_txn_category_{ck}_credit_cnt_{W}d")
        elif kind == "cnt_debit":
            cols_needed.append(f"bank_txn_category_{ck}_debit_cnt_{W}d")
        elif kind == "amt_total":
            cols_needed.append(f"bank_txn_category_{ck}_amt_{W}d")
        elif kind == "amt_credit":
            cols_needed.append(f"bank_txn_category_{ck}_credit_amt_{W}d")
        elif kind == "amt_debit":
            cols_needed.append(f"bank_txn_category_{ck}_debit_amt_{W}d")
    for c in cols_needed:
        if c not in df.columns:
            df[c] = 0 if ("_cnt" in c) else 0.0
    return df

def _extract_category_metric(
    cat_metrics: pd.DataFrame,
    metric: str,
    index: pd.Index,
    rename_fn,
) -> pd.DataFrame:
    if isinstance(cat_metrics, pd.DataFrame) and metric in cat_metrics.columns.get_level_values(0):
        out = cat_metrics[metric].copy()
        out.columns = [rename_fn(c) for c in out.columns]
        return out
    return pd.DataFrame(index=index)

def _agg_l0_vectorized(g: pd.DataFrame) -> pd.DataFrame:
    key = ["user_id", "sample_datetime"]

    base = g.groupby(key, observed=True).agg(
        **{
            "bank_txn_category_global_cnt": ("amount", "size"),
            "bank_txn_category_global_credit_cnt": ("_is_credit", "sum"),
            "bank_txn_category_global_debit_cnt": ("_is_debit", "sum"),
            "bank_txn_category_global_amt": ("_amt_total", "sum"),
            "bank_txn_category_global_credit_amt": ("_amt_credit", "sum"),
            "bank_txn_category_global_debit_amt": ("_amt_debit", "sum"),
        }
    )
    out = [base]

    for W in WINDOWS:
        vw = g[g["trac_days"] <= W]

        gw = vw.groupby(key, observed=True).agg(
            **{
                f"bank_txn_category_global_cnt_{W}d": ("amount", "size"),
                f"bank_txn_category_global_credit_cnt_{W}d": ("_is_credit", "sum"),
                f"bank_txn_category_global_debit_cnt_{W}d": ("_is_debit", "sum"),
                f"bank_txn_category_global_amt_{W}d": ("_amt_total", "sum"),
                f"bank_txn_category_global_credit_amt_{W}d": ("_amt_credit", "sum"),
                f"bank_txn_category_global_debit_amt_{W}d": ("_amt_debit", "sum"),
            }
        )
        out.append(gw)

        cat_metrics = vw.groupby(key + ["category"], observed=True).agg(
            cnt=("amount", "size"),
            credit_cnt=("_is_credit", "sum"),
            debit_cnt=("_is_debit", "sum"),
            amt=("_amt_total", "sum"),
            credit_amt=("_amt_credit", "sum"),
            debit_amt=("_amt_debit", "sum"),
        ).unstack("category", fill_value=0)
        cat_idx = cat_metrics.index if isinstance(cat_metrics, pd.DataFrame) else gw.index

        cat_cnt = _extract_category_metric(
            cat_metrics,
            "cnt",
            cat_idx,
            lambda c: f"bank_txn_category_{_cat_key(c)}_cnt_{W}d",
        )
        cat_cnt = _ensure_cat_columns(cat_cnt, W, "cnt_total")

        cat_cd_cnt = _extract_category_metric(
            cat_metrics,
            "credit_cnt",
            cat_idx,
            lambda c: f"bank_txn_category_{_cat_key(c)}_credit_cnt_{W}d",
        )
        cat_cd_cnt = _ensure_cat_columns(cat_cd_cnt, W, "cnt_credit")
        debit_cnt = _extract_category_metric(
            cat_metrics,
            "debit_cnt",
            cat_idx,
            lambda c: f"bank_txn_category_{_cat_key(c)}_debit_cnt_{W}d",
        )
        debit_cnt = _ensure_cat_columns(debit_cnt, W, "cnt_debit")
        cat_cd_cnt = cat_cd_cnt.join(debit_cnt, how="outer")

        cat_amt_total = _extract_category_metric(
            cat_metrics,
            "amt",
            cat_idx,
            lambda c: f"bank_txn_category_{_cat_key(c)}_amt_{W}d",
        )
        cat_amt_total = _ensure_cat_columns(cat_amt_total, W, "amt_total")

        cat_amt_credit = _extract_category_metric(
            cat_metrics,
            "credit_amt",
            cat_idx,
            lambda c: f"bank_txn_category_{_cat_key(c)}_credit_amt_{W}d",
        )
        cat_amt_credit = _ensure_cat_columns(cat_amt_credit, W, "amt_credit")

        cat_amt_debit = _extract_category_metric(
            cat_metrics,
            "debit_amt",
            cat_idx,
            lambda c: f"bank_txn_category_{_cat_key(c)}_debit_amt_{W}d",
        )
        cat_amt_debit = _ensure_cat_columns(cat_amt_debit, W, "amt_debit")

        cat_all = cat_cnt.join([cat_cd_cnt, cat_amt_total, cat_amt_credit, cat_amt_debit], how="outer")
        out.append(cat_all)

    l0 = pd.concat(out, axis=1)
    l0 = l0.sort_index().reset_index()
    return l0

# =========================================================
# = 5) L3/L4 行为层（按分类选择性产出，列名为 bank_txn_category_*）
# =========================================================
def _agg_l34(g: pd.DataFrame, selected_cats: Set[str]) -> pd.DataFrame:
    key_cols = ["user_id", "sample_datetime"]
    groups = g.groupby(key_cols, sort=False, dropna=False)
    frames = []

    do_activity = ("activity" in selected_cats)
    do_inter_gap = ("inter_txn_gap" in selected_cats)
    do_income_period = ("income_periodicity" in selected_cats)
    do_rent_lag = ("rent_lag" in selected_cats)
    do_netflow_streaks = ("netflow_streaks" in selected_cats)

    if not any([do_activity, do_inter_gap, do_income_period, do_rent_lag, do_netflow_streaks]):
        return pd.DataFrame(columns=key_cols)

    for (uid, sdt), df in groups:
        out = {"user_id": uid, "sample_datetime": sdt}

        # 先分卡，后聚合：避免多张卡流水直接混在一起导致节奏类变量失真
        acct_dfs = []
        for _, adf in df.groupby("bank_account_id", sort=False, dropna=False):
            acct_dfs.append(adf.sort_values(["_txn_date", "trac_days"], kind="mergesort").copy())

        # 应用级净流量：所有卡先按天汇总，再算连续正/负净流天数
        if do_netflow_streaks:
            signed_amt = np.where(
                df["_is_credit"].eq(1).values,
                df["_amt_total"].values,
                -df["_amt_debit_abs"].values,
            )
            daily_net = (
                pd.DataFrame({"_txn_date": df["_txn_date"], "_net": signed_amt})
                .dropna(subset=["_txn_date"])
                .groupby("_txn_date", as_index=False)["_net"].sum()
                .sort_values("_txn_date")
            )
        else:
            daily_net = None

        for W in WINDOWS:
            dW = df.loc[df["trac_days"] <= W].copy()
            cntW = len(dW)

            # 这三项看总活跃更合理，保留原有“应用级/用户级总行为”口径
            if do_activity:
                active_days = int(dW["_txn_date"].nunique())
                out[f"bank_txn_category_active_days_{W}d"] = active_days
                out[f"bank_txn_category_txn_freq_per_active_day_{W}d"] = float(cntW) / max(1, active_days)
                out[f"bank_txn_category_weekend_share_{W}d"] = (
                    float((dW["_dow"] >= 5).sum()) / float(cntW) if cntW > 0 else 0.0
                )

            # 交易间隔：必须按卡内节奏算，再把各卡 gap 池化后取分位数
            if do_inter_gap:
                pooled_gaps = []
                for adf in acct_dfs:
                    adW = adf.loc[adf["trac_days"] <= W]
                    uniq_days = np.sort(adW["_txn_date"].dropna().unique())
                    if uniq_days.size >= 2:
                        gaps = np.diff(uniq_days).astype("timedelta64[D]").astype(int)
                        if gaps.size:
                            pooled_gaps.extend(gaps.tolist())

                if len(pooled_gaps) > 0:
                    gaps_arr = np.asarray(pooled_gaps, dtype="float64")
                    out[f"bank_txn_category_inter_txn_gap_p50_{W}d"] = _pctl(gaps_arr, 50)
                    out[f"bank_txn_category_inter_txn_gap_p90_{W}d"] = _pctl(gaps_arr, 90)
                else:
                    out[f"bank_txn_category_inter_txn_gap_p50_{W}d"] = np.nan
                    out[f"bank_txn_category_inter_txn_gap_p90_{W}d"] = np.nan

            # 收入周期：按卡内独立识别工资节奏，再按有效样本量加权汇总
            if do_income_period:
                dom_std_list, dom_std_w = [], []
                gap_mean_list, gap_mean_w = [], []
                gap_std_list, gap_std_w = [], []

                for adf in acct_dfs:
                    wagesA = adf.loc[
                        (adf["trac_days"] <= W) & (adf["category"] == "Wages"),
                        ["_txn_date"],
                    ].dropna()

                    wage_days = np.sort(wagesA["_txn_date"].unique())
                    if wage_days.size >= 2:
                        dom_vals = pd.to_datetime(wage_days).day.astype("float64")
                        dom_std = float(np.nanstd(dom_vals, ddof=1)) if len(np.unique(dom_vals)) > 1 else 0.0
                        wgaps = np.diff(wage_days).astype("timedelta64[D]").astype(int)

                        dom_std_list.append(dom_std)
                        dom_std_w.append(len(wage_days))

                        if wgaps.size:
                            gap_mean_list.append(float(np.nanmean(wgaps)))
                            gap_mean_w.append(len(wgaps))
                            gap_std_list.append(float(np.nanstd(wgaps, ddof=1)) if wgaps.size > 1 else 0.0)
                            gap_std_w.append(len(wgaps))

                out[f"bank_txn_category_income_dom_std_{W}d"] = _weighted_nanmean(dom_std_list, dom_std_w)
                out[f"bank_txn_category_income_gap_mean_{W}d"] = _weighted_nanmean(gap_mean_list, gap_mean_w)
                out[f"bank_txn_category_income_gap_std_{W}d"] = _weighted_nanmean(gap_std_list, gap_std_w)

            # 房租跟工资的时滞：只允许同一卡内匹配，避免跨卡错配
            if do_rent_lag:
                pooled_lags = []
                for adf in acct_dfs:
                    adW = adf.loc[adf["trac_days"] <= W]
                    wages_sorted = np.sort(adW.loc[adW["category"] == "Wages", "_txn_date"].dropna().unique())
                    rent_dates = np.sort(adW.loc[adW["category"] == "Rent", "_txn_date"].dropna().unique())

                    if wages_sorted.size > 0 and rent_dates.size > 0:
                        pos = np.searchsorted(wages_sorted, rent_dates, side="right") - 1
                        valid = pos >= 0
                        if valid.any():
                            lags = (rent_dates[valid] - wages_sorted[pos[valid]]).astype("timedelta64[D]").astype(int)
                            pooled_lags.extend(lags.tolist())

                out[f"bank_txn_category_rent_after_wage_lag_p50_{W}d"] = (
                    _pctl(np.asarray(pooled_lags, dtype="float64"), 50) if len(pooled_lags) > 0 else np.nan
                )

            if do_netflow_streaks:
                if daily_net is not None and len(daily_net):
                    start_day = pd.to_datetime(sdt).floor("D") - pd.to_timedelta(W, unit="D")
                    end_day = pd.to_datetime(sdt).floor("D")
                    dnW = daily_net.loc[
                        (daily_net["_txn_date"] >= start_day) &
                        (daily_net["_txn_date"] <= end_day)
                    ]

                    epsilon = 1e-10
                    pos_run = _longest_streak((dnW["_net"].values >= epsilon).astype(bool))
                    neg_run = _longest_streak((dnW["_net"].values < -epsilon).astype(bool))
                else:
                    pos_run = 0
                    neg_run = 0

                out[f"bank_txn_category_pos_net_streak_days_{W}d"] = int(pos_run)
                out[f"bank_txn_category_neg_net_streak_days_{W}d"] = int(neg_run)

        frames.append(pd.DataFrame([out]))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=key_cols)

# =========================================================
# = 6) L1/L2 二次派生（按分类选择性产出，列名为 bank_txn_category_*）
# =========================================================
def post_derive(features: pd.DataFrame, selected_cats: Set[str]) -> pd.DataFrame:
    f = features.copy()

    do_global_structure = ("global_structure" in selected_cats)
    do_tickets = ("tickets" in selected_cats)
    do_cat_share = ("category_share" in selected_cats)
    do_cat_ticket = ("category_ticket" in selected_cats)
    do_cluster_share = ("cluster_share" in selected_cats)
    do_cluster_ticket = ("cluster_ticket" in selected_cats)
    do_cluster_momentum = ("cluster_momentum" in selected_cats)

    for W in WINDOWS:
        tot_amt = f.get(f"bank_txn_category_global_amt_{W}d", 0.0)
        cred_amt = f.get(f"bank_txn_category_global_credit_amt_{W}d", 0.0)
        debit_amt = f.get(f"bank_txn_category_global_debit_amt_{W}d", 0.0)
        cred_cnt = f.get(f"bank_txn_category_global_credit_cnt_{W}d", 0.0)
        debit_cnt = f.get(f"bank_txn_category_global_debit_cnt_{W}d", 0.0)

        if do_global_structure:
            f[f"bank_txn_category_global_income_share_{W}d"] = _safe_div(cred_amt, tot_amt)
            f[f"bank_txn_category_global_debit_share_{W}d"] = _safe_div(debit_amt, tot_amt)
            f[f"bank_txn_category_global_burn_rate_{W}d"] = _safe_div(debit_amt, cred_amt)
            f[f"bank_txn_category_global_saving_rate_{W}d"] = _safe_div(cred_amt - debit_amt, np.maximum(1.0, cred_amt))

        if do_tickets:
            f[f"bank_txn_category_global_income_avg_ticket_{W}d"] = _safe_div(cred_amt, np.maximum(1.0, cred_cnt))
            f[f"bank_txn_category_global_expense_avg_ticket_{W}d"] = _safe_div(debit_amt, np.maximum(1.0, debit_cnt))

        if do_cat_share or do_cat_ticket:
            for cat in CATEGORIES:
                ck = _cat_key(cat)
                cat_amt_total = f.get(f"bank_txn_category_{ck}_amt_{W}d", 0.0)
                cat_cnt_total = f.get(f"bank_txn_category_{ck}_cnt_{W}d", 0.0)
                den = np.maximum(1.0, cred_amt) if cat == "Wages" else np.maximum(1.0, debit_amt)

                if do_cat_share:
                    f[f"bank_txn_category_{ck}_share_{W}d"] = _safe_div(cat_amt_total, den)
                if do_cat_ticket:
                    f[f"bank_txn_category_{ck}_avg_ticket_{W}d"] = _safe_div(cat_amt_total, np.maximum(1.0, cat_cnt_total))

        if do_cluster_share or do_cluster_ticket:
            for cls_key, spec in CLUSTERS.items():
                denom_mode = spec["denom"]
                cats = spec["cats"]
                if denom_mode == "debit":
                    cluster_amt = 0.0
                    cluster_cnt = 0.0
                    denom = np.maximum(1.0, debit_amt)
                    for c in cats:
                        ck = _cat_key(c)
                        cluster_amt += f.get(f"bank_txn_category_{ck}_debit_amt_{W}d", 0.0)
                        cluster_cnt += f.get(f"bank_txn_category_{ck}_debit_cnt_{W}d", 0.0)
                elif denom_mode == "credit":
                    cluster_amt = 0.0
                    cluster_cnt = 0.0
                    denom = np.maximum(1.0, cred_amt)
                    for c in cats:
                        ck = _cat_key(c)
                        cluster_amt += f.get(f"bank_txn_category_{ck}_credit_amt_{W}d", 0.0)
                        cluster_cnt += f.get(f"bank_txn_category_{ck}_credit_cnt_{W}d", 0.0)
                else:
                    cluster_amt = 0.0
                    cluster_cnt = 0.0
                    denom = np.maximum(1.0, tot_amt)
                    for c in cats:
                        ck = _cat_key(c)
                        cluster_amt += f.get(f"bank_txn_category_{ck}_amt_{W}d", 0.0)
                        cluster_cnt += f.get(f"bank_txn_category_{ck}_cnt_{W}d", 0.0)

                if do_cluster_share:
                    f[f"bank_txn_category_cluster_{cls_key}_share_{W}d"] = _safe_div(cluster_amt, denom)
                if do_cluster_ticket:
                    f[f"bank_txn_category_cluster_{cls_key}_avg_ticket_{W}d"] = _safe_div(cluster_amt, np.maximum(1.0, cluster_cnt))

    if do_cluster_momentum:
        for cls_key in RISK_CLUSTERS_FOR_MOMENTUM:
            for i in range(len(WINDOWS) - 1):
                w1, w2 = WINDOWS[i], WINDOWS[i + 1]
                s1 = f.get(f"bank_txn_category_cluster_{cls_key}_share_{w1}d", np.nan)
                s2 = f.get(f"bank_txn_category_cluster_{cls_key}_share_{w2}d", np.nan)
                f[f"bank_txn_category_cluster_{cls_key}_share_delta_{w1}v{w2}"] = (s1 - s2)
            for i in range(len(WINDOWS) - 2):
                w1, w2, w3 = WINDOWS[i], WINDOWS[i + 1], WINDOWS[i + 2]
                s1 = f.get(f"bank_txn_category_cluster_{cls_key}_share_{w1}d", np.nan)
                s2 = f.get(f"bank_txn_category_cluster_{cls_key}_share_{w2}d", np.nan)
                s3 = f.get(f"bank_txn_category_cluster_{cls_key}_share_{w3}d", np.nan)
                f[f"bank_txn_category_cluster_{cls_key}_share_accel_{w1}_{w2}_{w3}"] = (s1 - s2) - (s2 - s3)

    return f

# =========================================================
# = 7) 主装配（总入口）
# =========================================================
def aggregate_transactions(df: pd.DataFrame):
    g = _prepare(df)
    l0 = _agg_l0_vectorized(g)
    return l0, g

def compute_bank_features(
    df: pd.DataFrame,
    categories: Union[str, List[str], None]
) -> pd.DataFrame:
    """
    入参：
      - df：原始明细（支持 flowTime 列；若无 trac_days 将自动用 transaction_date 推算）
      - categories：单选(字符串)、多选(列表)、全选(None 或 "ALL")
    返回：主键 + 依赖（可选暴露）+ 选择分类对应的变量
    """
    selected = _normalize_categories(categories)
    l0, g = aggregate_transactions(df)
    l34 = _agg_l34(g, selected_cats=selected)

    l0["sample_datetime"] = pd.to_datetime(l0["sample_datetime"], errors="coerce")
    l34["sample_datetime"] = pd.to_datetime(l34["sample_datetime"], errors="coerce")

    features_l0_l34 = l0.merge(l34, on=["user_id", "sample_datetime"], how="left")
    features_full = post_derive(features_l0_l34, selected_cats=selected)

    key_cols = ["user_id", "sample_datetime"]
    keep_cols = set(key_cols)

    if EXPOSE_DEPENDENT:
        keep_cols |= set(DEPENDENT_VARS)

    for cat in selected:
        keep_cols |= set(CATEGORY_MAP.get(cat, []))

    final_cols = [c for c in features_full.columns if c in keep_cols]
    return features_full.loc[:, final_cols]

# =========================================================
# = 8) 用法示例（可删）
# =========================================================
if __name__ == "__main__":
    # df_input = pd.read_csv(
    #     "your_file.csv",
    #     dtype={"user_id": str, "bank_account_id": str, "category": str, "dr_cr": str}
    # )
    # features_all = compute_bank_features(df_input, categories=["cluster_share", "category_ticket"])
    # print(features_all.shape)
    pass
