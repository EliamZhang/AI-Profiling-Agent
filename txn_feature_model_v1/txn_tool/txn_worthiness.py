from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=Warning)

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================
# 配置区
# ============================================================
RAW_COLUMNS = [
    "user_id",
    "sample_datetime",
    "application_id",
    "job_id",
    "transaction_id",
    "bank_account_id",
    "account_type",
    "transaction_date",
    "amount",
    "trx_type",
    "dr_cr",
    "category",
    "third_party",
    "illion_trx_uuid",
    "balance",
    "text",
    "tag_level1",
    "tag_level2",
]

DAILY_LIVING_CATEGORIES = {
    "groceries",
    "dining out",
    "department stores",
    "utilities",
    "telecommunications",
}

FIXED_BILL_CATEGORIES = {"utilities", "telecommunications", "insurance", "rent"}

PREMIUM_LIFESTYLE_CATEGORIES = {
    "travel",
    "department stores",
    "retail",
    "entertainment",
    "gyms and other memberships",
    "personal care",
    "subscription tv",
    "pet care",
}

HIGH_COMMITMENT_CATEGORIES = {
    "rent",
    "insurance",
    "education",
    "health",
    "gyms and other memberships",
}

FIXED_COMMITMENT_CATEGORIES = FIXED_BILL_CATEGORIES | HIGH_COMMITMENT_CATEGORIES

# LUXURY_RETAIL_KEYWORDS = [
#     "louis vuitton",
#     "solid gold diamonds",
#     "miscellaneous jewellery stores",
#     "other jewellery stores",
# ]

# PREMIUM_CAR_KEYWORDS = [
#     "tesla",
#     "audi automotive services",
#     "bmw automotive services",
#     "lexus automotive services",
#     "mercedes-benz automotive servi",
#     "mercedes benz automotive servi",
# ]

# PREMIUM_REAL_ESTATE_KEYWORDS = [
#     "marshall white",
#     "jellis craig",
#     "mcgrath estate agents",
#     "colliers international",
#     "strata payments",
#     "strata fees",
# ]

# HIGH_END_HOTEL_KEYWORDS = [
#     "four seasons hotels",
#     "shangri-la hotels and resorts",
#     "shangri la hotels and resorts",
#     "hilton hotels and resorts",
#     "hyatt hotels",
#     "sofitel hotels and resorts",
#     "marriott hotels",
#     "intercontinental hotels group",
# ]

# PREMIUM_AIRLINE_KEYWORDS = [
#     "emirates",
#     "singapore airlines",
#     "cathay pacific",
#     "qatar airways",
# ]

# ELITE_EDUCATION_KEYWORDS = [
#     "edstart",
#     "mlc school",
# ]

# HIGH_VALUE_HEALTHCARE_KEYWORDS = [
#     "periodontist",
#     "orthodontist",
#     "surgery",
# ]

# INVESTMENT_PROPERTY_INSURANCE_KEYWORDS = [
#     "terri scheer",
#     "ebm",
# ]

# LOAN_PROTECTION_INSURANCE_KEYWORDS = [
#     "loan protection insurance",
# ]

# HIGH_VALUE_RETAIL_KEYWORDS = [
#     "david jones",
#     "myer",
# ]

# GAMBLING_KEYWORDS = [
#     "sportsbet",
#     "ladbrokes",
#     "bet365",
#     "pointsbet",
#     "tab",
#     "neds",
#     "bluebet",
#     "unibet",
#     "palmerbet",
# ]

SHOWCASE_SPEND_CATEGORIES = {
    "travel",
    "rent",
    "insurance",
    "education",
    "health",
    "department stores",
    "retail",
    "entertainment",
    "home improvement",
    "automotive",
    "gyms and other memberships",
    "personal care",
    "subscription tv",
    "pet care",
}

BANK_LOAN_KEYWORDS = [
    "anz credit card cash advance", "anz loan redraw", "anz loan repayment", "anz",
    "bankwest credit card cash advance", "bankwest loans", "bankwest",
    "ing loan redraw", "ing loan redraw credit", "ing loan repayment debit", "ing loans", "ing",
    "nab credit card cash advance", "nab loan redraw", "nab loan repayment", "nab",
    "westpac credit card cash advance", "westpac loan redraw", "westpac loan repayment", "westpac",
    "newcastle credit card cash advance", "newcastle loans", "newcastle",
    "peopleschoicecu loans", "people's choice",
]

BANK_LOAN_PATTERN = re.compile(
    "|".join(re.escape(x) for x in sorted(set(BANK_LOAN_KEYWORDS), key=len, reverse=True))
)

LIABILITY_REPAYMENT_TAGS = {"liability_repayment", "liability repayment"}
REFUND_OR_REVERSAL_TAGS = {"refund_or_reversal", "refund or reversal"}
LIABILITY_DRAWDOWN_TAGS = {"liability_drawdown", "liability drawdown"}
DISHONOURS_FEE_TAGS = {"dishonours fee", "dishonours_fee"}


# ============================================================
# 高净值消费识别商户池（严格按本次口径）
# ============================================================
HIGH_WORTHINESS_MERCHANT_POOLS = {
    "insurance": {
        "category": "insurance",
        "premium": [
            "allianz", "chubb", "zurich", "qbe", "aig", "marsh", "iag", "suncorp",
            "bupa", "aia", "tal", "hannover life", "mlc", "cgu", "hollard",
            "resolution life", "metlife",
        ],
    },
    "pet_care": {
        "category": "pet care",
        "unified": [
            "amacron", "veterinary", "pet accommodations", "mad paws", "pet direct",
            "pet supplies", "pet care", "a1 pet meats", "petquarters",
            "greencross vets", "debitsuccess pet care", "animates", "city farmers",
            "your best friends butcher",
        ],
    },
    "automotive": {
        "category": "automotive",
        "premium": [
            "bmw automotive services", "audi automotive services",
            "lexus automotive services", "tesla",
        ],
    },
    "travel": {
        "category": "travel",
        "premium": [
            "celebrity cruises", "qantas", "emirates", "qatar airways",
            "singapore airlines", "etihad", "cathay pacific", "thai airways",
            "four seasons", "hilton", "intercontinental", "marriott", "hyatt",
            "shangri-la", "shangri la", "sofitel", "pullman", "swissotel",
            "raffles", "radisson", "wyndham", "accor", "crown hotels",
            "the star grand", "rydges",
        ],
    },
    "retail": {
        "category": "retail",
        "premium": [
            "louis vuitton", "oakley", "mecca", "seed heritage", "colette",
            "portmans", "williams sonoma", "domayne", "bose", "myhouse",
            "westfield", "kathmandu", "rebel sport", "lush", "crocs", "puma",
            "dell", "pb tech", "snap on tools", "duty free shops",
        ],
    },
    "dining_out": {
        "category": "dining out",
        "premium": [
            "blanca honeysuckle", "siglo", "din tai fung", "opera bar",
            "crown restaurants", "the star restaurants", "hilton restaurants",
            "skycity restaurants", "koko black", "the grand",
            "pacific concepts", "international hotel",
        ],
    },
    "rent": {
        "category": "rent",
        "premium": [
            "marshall white", "jellis craig", "mcgrath", "ray white", "colliers",
            "momentum wealth", "harcourts", "remax", "mgm martin",
            "wentworth partners", "buxton", "dixon commercial",
            "nelson alexander",
        ],
    },
    "department_stores": {
        "category": "department stores",
        "premium": [
            "david jones", "myer", "harris scarfe",
        ],
    },
}

HIGH_WORTHINESS_RIGID_CATEGORIES = {"insurance", "rent"}
HIGH_WORTHINESS_NECESSITY_CATEGORIES = {
    "rent", "groceries", "utilities", "telecommunications", "insurance"
}



@dataclass
class FeatureConfig:
    windows: Sequence[int] = field(default_factory=lambda: [30, 60, 90, 180])

    large_loan_amount_threshold: float = 4000.0
    large_expense_amount_threshold: float = 1000.0
    large_travel_amount_threshold: float = 800.0

    balance_high_threshold_1: float = 1000.0
    balance_high_threshold_2: float = 2000.0
    balance_high_threshold_3: float = 5000.0

    @property
    def max_window(self) -> int:
        return max(int(w) for w in self.windows)


# ============================================================
# 通用工具函数
# ============================================================
def _normalize_str_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _compile_keyword_pattern(keywords: Sequence[str]) -> re.Pattern:
    clean_keywords = [str(x).strip().lower() for x in keywords if str(x).strip()]
    if not clean_keywords:
        return re.compile(r"$^")
    return re.compile("|".join(re.escape(x) for x in sorted(set(clean_keywords), key=len, reverse=True)))


def safe_divide_scalar(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    if pd.isna(numerator):
        return np.nan
    out = numerator / denominator
    if np.isinf(out):
        return np.nan
    return float(out)


def safe_divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan, 0.0: np.nan})
    out = numerator / denominator
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def safe_growth_pct(current_value: float, previous_value: float) -> float:
    if pd.isna(current_value) or pd.isna(previous_value) or previous_value == 0:
        return np.nan
    return float((current_value - previous_value) / abs(previous_value))


def safe_std(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan if len(values) == 0 else 0.0
    return float(values.std())


def safe_cv(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return np.nan
    mean_v = float(values.mean())
    std_v = float(values.std()) if len(values) > 1 else 0.0
    return safe_divide_scalar(std_v, abs(mean_v))


def nonzero_min(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    values = values[values > 0]
    if len(values) == 0:
        return np.nan
    return float(values.min())



# LUXURY_RETAIL_PATTERN = _compile_keyword_pattern(LUXURY_RETAIL_KEYWORDS)
# PREMIUM_CAR_PATTERN = _compile_keyword_pattern(PREMIUM_CAR_KEYWORDS)
# PREMIUM_REAL_ESTATE_PATTERN = _compile_keyword_pattern(PREMIUM_REAL_ESTATE_KEYWORDS)
# HIGH_END_HOTEL_PATTERN = _compile_keyword_pattern(HIGH_END_HOTEL_KEYWORDS)
# PREMIUM_AIRLINE_PATTERN = _compile_keyword_pattern(PREMIUM_AIRLINE_KEYWORDS)
# ELITE_EDUCATION_PATTERN = _compile_keyword_pattern(ELITE_EDUCATION_KEYWORDS)
# HIGH_VALUE_HEALTHCARE_PATTERN = _compile_keyword_pattern(HIGH_VALUE_HEALTHCARE_KEYWORDS)
# INVESTMENT_PROPERTY_INSURANCE_PATTERN = _compile_keyword_pattern(INVESTMENT_PROPERTY_INSURANCE_KEYWORDS)
# LOAN_PROTECTION_INSURANCE_PATTERN = _compile_keyword_pattern(LOAN_PROTECTION_INSURANCE_KEYWORDS)
# HIGH_VALUE_RETAIL_PATTERN = _compile_keyword_pattern(HIGH_VALUE_RETAIL_KEYWORDS)
# GAMBLING_PATTERN = _compile_keyword_pattern(GAMBLING_KEYWORDS)

HIGH_WORTHINESS_PATTERN_MAP = {
    category_key: {
        pool_name: _compile_keyword_pattern(keywords)
        for pool_name, keywords in pool_conf.items()
        if pool_name in {"premium", "unified"}
    }
    for category_key, pool_conf in HIGH_WORTHINESS_MERCHANT_POOLS.items()
}



def max_consecutive_months(periods: Iterable[pd.Period]) -> int:
    vals = sorted({(p.year * 12 + p.month) for p in periods if pd.notna(p)})
    if not vals:
        return 0
    best = 1
    streak = 1
    for i in range(1, len(vals)):
        if vals[i] - vals[i - 1] == 1:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1
    return best


def _read_single_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, low_memory=False)


# ============================================================
# 单申请 worthiness 特征工程
# ============================================================
class SingleApplicationPremiumCustomerFeatureEngineer:
    """
    参考 test_code 的单申请架构：
    - 每次只传入一个 application 的银行流水 DataFrame
    - 初始化时完成 finv 映射、银行卡兜底、时间窗口准备
    - 输出支持按 feature_groups 调用单个/多个类别
    - 最终返回单行宽表 DataFrame

    本版本调整：
    1. 最终输出变量统一加 bank_txn_worthiness_ 前缀
    2. 不输出任何 flag 型变量
    3. category 收入/支出/负债口径通过 finv 映射矩阵补齐 tag_level1 / tag_level2 后计算
    4. 新增多银行卡不失真的余额类资质特征（先卡内算，再申请级聚合）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_windows: Optional[List[int]] = None,
        mapping_file: Optional[str] = None,
        config: Optional[FeatureConfig] = None,
        config_overrides: Optional[Dict] = None,
    ):
        if df is None or len(df) == 0:
            raise ValueError("Input df is empty. This class expects one application's transaction DataFrame.")

        windows = sorted(set(int(w) for w in (time_windows or [30, 60, 90, 180])))
        if config is None:
            cfg_dict = {"windows": windows}
            if config_overrides:
                cfg_dict.update(config_overrides)
            config = FeatureConfig(**cfg_dict)
        else:
            config.windows = windows
            if config_overrides:
                for k, v in config_overrides.items():
                    setattr(config, k, v)

        self.config = config
        self.time_windows = list(self.config.windows)
        # 方案B:所有特征窗口均从 time_windows 配置推导,不硬编码
        # - 长窗口   = 配置中最长的窗口(如 182d)
        # - 环比窗口 = 最接近"一个月"(30天)的配置窗口(如 28d)
        self.long_window = self.config.max_window
        self.mom_window = min(self.time_windows, key=lambda w: abs(w - 30))
        self.mapping_file = mapping_file
        self.features: Dict[str, float] = {}
        self.account_cache: Dict[int, Dict[str, float]] = {}
        self.direct_cache: Dict[int, Dict[str, float]] = {}
        self.account_daily_balance_cache: Dict[str, pd.DataFrame] = {}
        self.application_daily_total_balance_df: pd.DataFrame = pd.DataFrame()

        self.df = self._map(df, mapping_file=self.mapping_file)
        self._prepare_data()

        self.raw_df = self.df.copy()
        self._precompute_window_masks()

    # ========================================================
    # 数据准备 / 映射
    # ========================================================
    def _map(self, df: pd.DataFrame, mapping_file: Optional[str] = None) -> pd.DataFrame:
        """参考 test_code：读取 finv 映射矩阵，并基于 dr_cr/category/account_type 补充 tag_level1/tag_level2。"""
        df = df.copy()

        cols = ["account_type", "category", "dr_cr", "third_party"]
        for c in cols:
            if c not in df.columns:
                df[c] = None

        df[cols] = df[cols].astype(str).apply(lambda x: x.str.strip()).replace({"": None, "nan": None, "None": None})
        df.fillna({"account_type": "Unlabeled", "category": "Unlabeled", "dr_cr": "Unlabeled"}, inplace=True)

        if mapping_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_file = os.path.join(current_dir, "reference", "finv_level", "finv交易类别映射矩阵_0303.csv")

        if not os.path.exists(mapping_file):
            return df

        file_ext = os.path.splitext(mapping_file)[1].lower()
        try:
            mapping_df = (
                pd.read_csv(mapping_file) if file_ext == ".csv"
                else pd.read_excel(mapping_file) if file_ext in [".xlsx", ".xls"]
                else None
            )
        except Exception:
            return df

        if mapping_df is None:
            return df

        need_cols = ["dr_cr", "category", "account_type", "tag_level1", "tag_level2"]
        if any(c not in mapping_df.columns for c in need_cols):
            return df

        left = df.copy()
        for c in ["dr_cr", "category", "account_type"]:
            left[c] = left[c].astype(str).str.strip()
            mapping_df[c] = mapping_df[c].astype(str).str.strip()

        # 当前口径：原始输入不含 tag_level1/tag_level2，统一依赖 finv 映射矩阵补充
        if "tag_level1" in left.columns:
            left.drop(columns=["tag_level1"], inplace=True)
        if "tag_level2" in left.columns:
            left.drop(columns=["tag_level2"], inplace=True)

        left = left.merge(mapping_df[need_cols], on=["dr_cr", "category", "account_type"], how="left")
        return left

    def _prepare_data(self) -> None:
        df = self.df.copy()

        for col in RAW_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        for c in ["account_type", "trx_type", "dr_cr", "category", "third_party", "text", "tag_level1", "tag_level2"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = _normalize_str_series(df[c])

        if "bank_account_id" not in df.columns:
            df["bank_account_id"] = "__missing_account__"
        df["bank_account_id"] = (
            df["bank_account_id"]
            .astype(str)
            .replace({"": "__missing_account__", "nan": "__missing_account__", "None": "__missing_account__"})
            .fillna("__missing_account__")
        )

        df["sample_datetime"] = pd.to_datetime(df["sample_datetime"], errors="coerce")
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["transaction_day"] = df["transaction_date"].dt.floor("D")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs().fillna(0.0)
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
        df["transaction_id"] = df["transaction_id"].astype("string").fillna("").str.strip()
        df["_transaction_id_num"] = pd.to_numeric(df["transaction_id"], errors="coerce")
        df["_transaction_id_str"] = df["transaction_id"].fillna("")

        df = df.dropna(subset=["transaction_date"]).copy()
        if df["sample_datetime"].notna().any():
            self.sample_datetime = df.loc[df["sample_datetime"].notna(), "sample_datetime"].iloc[0]
            self.sample_day = pd.Timestamp(self.sample_datetime).floor("D")
        else:
            self.sample_datetime = pd.NaT
            self.sample_day = pd.NaT

        self.user_id = df["user_id"].dropna().iloc[0] if df["user_id"].notna().any() else None
        self.application_id = df["application_id"].dropna().iloc[0] if df["application_id"].notna().any() else None

        if pd.notna(self.sample_day):
            df["trac_days"] = (self.sample_day - df["transaction_day"]).dt.days
            df = df[(df["trac_days"] >= 0) & (df["trac_days"] <= self.config.max_window)].copy()
        else:
            df["trac_days"] = np.nan
            df = df.iloc[0:0].copy()

        if not df.empty:
            df["month_period"] = df["transaction_date"].dt.to_period("M")
        else:
            df["month_period"] = pd.PeriodIndex([], freq="M")

#         df["income_source_key"] = np.where(df["third_party"].ne(""), df["third_party"], df["text"])
        df["income_source_key"] = df["third_party"]
#         df["counterparty_search_text"] = (
#             df["third_party"].fillna("").astype(str).str.strip()
#             + " "
#             + df["text"].fillna("").astype(str).str.strip()
#         ).str.strip()
#         df["counterparty_search_text"] = _normalize_str_series(df["counterparty_search_text"])
        
        df["counterparty_search_text"] = (
        df["third_party"].fillna("").astype(str).str.strip()
        ).str.strip()

        # ========= 核心标签口径（按 tag_level1/tag_level2 修正） =========
        df["is_income_tag"] = df["tag_level1"].eq("income")
        df["is_expense_tag"] = df["tag_level1"].eq("expense")
        df["is_liability_tag"] = df["tag_level1"].eq("liability")

        df["is_wage"] = df["is_income_tag"] & df["category"].eq("wages")
        df["is_rent"] = df["is_expense_tag"] & df["category"].eq("rent")
        df["is_insurance"] = df["is_expense_tag"] & df["category"].eq("insurance")
        df["is_daily_living"] = df["is_expense_tag"] & df["category"].isin(DAILY_LIVING_CATEGORIES)
        df["is_fixed_bill"] = df["is_expense_tag"] & df["category"].isin(FIXED_BILL_CATEGORIES)

        df["is_liability_repayment"] = df["is_liability_tag"] & df["tag_level2"].isin(LIABILITY_REPAYMENT_TAGS)
        df["is_refund_or_reversal"] = df["is_liability_tag"] & df["tag_level2"].isin(REFUND_OR_REVERSAL_TAGS)
        df["is_liability_drawdown"] = df["is_liability_tag"] & df["tag_level2"].isin(LIABILITY_DRAWDOWN_TAGS)
        df["is_dishonours_fee"] = df["is_liability_tag"] & df["tag_level2"].isin(DISHONOURS_FEE_TAGS)

        df["is_bank_liability_drawdown"] = (
            df["is_liability_drawdown"]
            & df["third_party"].str.contains(BANK_LOAN_PATTERN, regex=True, na=False)
        )
        df["is_bank_liability_repayment"] = (
            df["is_liability_repayment"]
            & df["third_party"].str.contains(BANK_LOAN_PATTERN, regex=True, na=False)
        )
        df["is_large_bank_liability_drawdown"] = (
            df["is_bank_liability_drawdown"] & (df["amount"] >= self.config.large_loan_amount_threshold)
        )

        df["is_travel"] = df["is_expense_tag"] & df["category"].eq("travel")
        df["is_premium_lifestyle"] = df["is_expense_tag"] & df["category"].isin(PREMIUM_LIFESTYLE_CATEGORIES)
        df["is_high_commitment"] = df["is_expense_tag"] & df["category"].isin(HIGH_COMMITMENT_CATEGORIES)
        df["is_fixed_commitment"] = df["is_expense_tag"] & df["category"].isin(FIXED_COMMITMENT_CATEGORIES)
        df["is_showcase_spend"] = df["is_expense_tag"] & df["category"].isin(SHOWCASE_SPEND_CATEGORIES)

#         df["is_luxury_retail_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(LUXURY_RETAIL_PATTERN, regex=True, na=False)
#         df["is_premium_car_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(PREMIUM_CAR_PATTERN, regex=True, na=False)
#         df["is_premium_real_estate_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(PREMIUM_REAL_ESTATE_PATTERN, regex=True, na=False)
#         df["is_high_end_hotel_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(HIGH_END_HOTEL_PATTERN, regex=True, na=False)
#         df["is_premium_airline_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(PREMIUM_AIRLINE_PATTERN, regex=True, na=False)
#         df["is_elite_education_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(ELITE_EDUCATION_PATTERN, regex=True, na=False)
#         df["is_high_value_healthcare_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(HIGH_VALUE_HEALTHCARE_PATTERN, regex=True, na=False)
#         df["is_investment_property_insurance_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(INVESTMENT_PROPERTY_INSURANCE_PATTERN, regex=True, na=False)
#         df["is_loan_protection_insurance_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(LOAN_PROTECTION_INSURANCE_PATTERN, regex=True, na=False)
#         df["is_high_value_retail_merchant"] = df["is_expense_tag"] & df["counterparty_search_text"].str.contains(HIGH_VALUE_RETAIL_PATTERN, regex=True, na=False)
#         df["is_gambling_merchant"] = (
#             df["is_expense_tag"]
#             & (
#                 df["category"].eq("gambling")
#                 | df["counterparty_search_text"].str.contains(GAMBLING_PATTERN, regex=True, na=False)
#             )
#         )

        # ========= 本次新增：8大高净值消费类别识别 =========
        for worthiness_key, pool_conf in HIGH_WORTHINESS_MERCHANT_POOLS.items():
            category_name = pool_conf["category"]
            category_mask = df["is_expense_tag"] & df["category"].eq(category_name)

            if "premium" in pool_conf:
                premium_pattern = HIGH_WORTHINESS_PATTERN_MAP[worthiness_key]["premium"]
                df[f"is_worthiness_{worthiness_key}"] = category_mask
                df[f"is_worthiness_{worthiness_key}_premium"] = (
                    category_mask
                    & df["counterparty_search_text"].str.contains(premium_pattern, regex=True, na=False)
                )
                df[f"is_worthiness_{worthiness_key}_general"] = (
                    category_mask & ~df[f"is_worthiness_{worthiness_key}_premium"]
                )
            else:
                unified_pattern = HIGH_WORTHINESS_PATTERN_MAP[worthiness_key]["unified"]
                df[f"is_worthiness_{worthiness_key}"] = (
                    df["is_expense_tag"]
                    & df["counterparty_search_text"].str.contains(unified_pattern, regex=True, na=False)
                )


        df["is_large_expense"] = df["is_expense_tag"] & (df["amount"] >= self.config.large_expense_amount_threshold)
        df["is_large_travel"] = df["is_travel"] & (df["amount"] >= self.config.large_travel_amount_threshold)
        df["is_large_showcase_spend"] = df["is_showcase_spend"] & (df["amount"] >= self.config.large_expense_amount_threshold)

        df.sort_values(by=["bank_account_id", "transaction_day", "_transaction_id_num", "_transaction_id_str", "transaction_date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.df = df
        self._build_account_daily_balance_cache()
        self._build_application_daily_total_balance_cache()

    def _precompute_window_masks(self) -> None:
        # 掩码只按 time_windows 配置构建;特征窗口全部由配置推导(见 __init__),
        # 因此所有 _window_df 调用必然命中掩码,未命中会由 _window_df 显式报错
        self.window_masks: Dict[int, pd.Series] = {}
        if self.df.empty or "trac_days" not in self.df.columns:
            for w in self.time_windows:
                self.window_masks[w] = pd.Series(dtype=bool)
            return
        for w in self.time_windows:
            self.window_masks[w] = self.df["trac_days"] <= w

    # ========================================================
    # 辅助函数
    # ========================================================
    def _iter_account_dfs(self):
        if self.df.empty:
            return []
        return [(acc_id, g.copy()) for acc_id, g in self.df.groupby("bank_account_id", dropna=False, sort=False)]

    def _window_df(self, window: int) -> pd.DataFrame:
        if window in self.window_masks:
            return self.df.loc[self.window_masks[window]].copy()
        raise KeyError(
            f"window={window} 不在 window_masks({sorted(self.window_masks)}) 中,"
            f"请加入 FIXED_WINDOWS 或 time_windows"
        )

    def _sum_count_max(self, df: pd.DataFrame, mask_col: str, prefix: str, window: int) -> Dict[str, float]:
        if df.empty or mask_col not in df.columns:
            return {
                f"{prefix}_amt_sum_{window}d": 0.0,
                f"{prefix}_cnt_{window}d": 0.0,
                f"{prefix}_amt_max_{window}d": 0.0,
            }

        tmp = df.loc[df[mask_col]].copy()
        if tmp.empty:
            return {
                f"{prefix}_amt_sum_{window}d": 0.0,
                f"{prefix}_cnt_{window}d": 0.0,
                f"{prefix}_amt_max_{window}d": 0.0,
            }

        return {
            f"{prefix}_amt_sum_{window}d": float(tmp["amount"].sum()),
            f"{prefix}_cnt_{window}d": float(tmp["amount"].count()),
            f"{prefix}_amt_max_{window}d": float(tmp["amount"].max()),
        }

    def _sum_only(self, df: pd.DataFrame, mask_col: str, feature_name: str) -> float:
        if df.empty or mask_col not in df.columns:
            return 0.0
        tmp = df.loc[df[mask_col], "amount"]
        return float(tmp.sum()) if len(tmp) > 0 else 0.0

    def _distinct_count(self, df: pd.DataFrame, mask: pd.Series, column: str) -> float:
        if df.empty or column not in df.columns:
            return 0.0
        tmp = df.loc[mask & df[column].ne(""), column]
        return float(tmp.nunique()) if len(tmp) > 0 else 0.0

    def _days_since_last(self, df: pd.DataFrame, mask: pd.Series) -> float:
        if df.empty:
            return np.nan
        tmp = df.loc[mask, "trac_days"]
        return float(tmp.min()) if len(tmp) > 0 else np.nan

    def _monthly_count(self, df: pd.DataFrame, mask: pd.Series) -> float:
        if df.empty:
            return 0.0
        tmp = df.loc[mask, "month_period"]
        return float(tmp.nunique()) if len(tmp) > 0 else 0.0

    def _max_consecutive_month_cnt(self, df: pd.DataFrame, mask: pd.Series) -> float:
        if df.empty:
            return 0.0
        tmp = df.loc[mask, "month_period"]
        if len(tmp) == 0:
            return 0.0
        return float(max_consecutive_months(tmp))

    def _build_account_daily_balance_cache(self) -> None:
        """
        余额口径修正：
        1. 对每张卡、每个自然日，仅保留 transaction_id 最大的那笔作为当日余额；
        2. 若某一天无交易，则沿用上一天最后一笔交易余额（前向填充）；
        3. 所有余额类特征统一基于该“卡-日终余额序列”计算，避免直接用交易明细 balance 失真。
        """
        self.account_daily_balance_cache = {}

        if self.df.empty or pd.isna(getattr(self, "sample_day", pd.NaT)):
            return

        calendar = pd.DataFrame({
            "transaction_day": pd.date_range(
                start=self.sample_day - pd.Timedelta(days=self.config.max_window),
                end=self.sample_day,
                freq="D",
            )
        })

        for acc_id, acc_df in self._iter_account_dfs():
            if acc_df.empty:
                continue

            daily_rows = []
            for _, day_df in acc_df.groupby("transaction_day", sort=True, dropna=False):
                tmp = day_df.copy()
                if tmp["_transaction_id_num"].notna().any():
                    max_tid = tmp["_transaction_id_num"].max()
                    tmp = tmp.loc[tmp["_transaction_id_num"].eq(max_tid)].copy()
                else:
                    max_tid = tmp["_transaction_id_str"].max()
                    tmp = tmp.loc[tmp["_transaction_id_str"].eq(max_tid)].copy()

                tmp.sort_values(by=["transaction_date", "_transaction_id_str"], inplace=True)
                last_row = tmp.iloc[-1]
                balance_value = last_row["balance"]
                if pd.isna(balance_value) and tmp["balance"].notna().any():
                    balance_value = tmp.loc[tmp["balance"].notna(), "balance"].iloc[-1]

                daily_rows.append({
                    "transaction_day": last_row["transaction_day"],
                    "balance": balance_value,
                })

            if daily_rows:
                daily_balance_df = pd.DataFrame(daily_rows).drop_duplicates(subset=["transaction_day"], keep="last")
            else:
                daily_balance_df = pd.DataFrame(columns=["transaction_day", "balance"])

            daily_balance_df = calendar.merge(daily_balance_df, on="transaction_day", how="left")
            daily_balance_df["balance"] = daily_balance_df["balance"].ffill()
            daily_balance_df["trac_days"] = (self.sample_day - daily_balance_df["transaction_day"]).dt.days
            self.account_daily_balance_cache[str(acc_id)] = daily_balance_df


    def _build_application_daily_total_balance_cache(self) -> None:
        """
        申请级总余额口径：
        1. 先得到每张卡的逐日余额序列；
        2. 再按日期对所有银行卡余额求和，得到申请级每日总余额；
        3. 若某日所有卡都缺失余额，则申请级余额保留为空。
        """
        self.application_daily_total_balance_df = pd.DataFrame(columns=["transaction_day", "total_balance", "trac_days"])

        if not self.account_daily_balance_cache or pd.isna(getattr(self, "sample_day", pd.NaT)):
            return

        balance_frames = []
        for acc_id, acc_df in self.account_daily_balance_cache.items():
            if acc_df is None or acc_df.empty:
                continue
            tmp = acc_df[["transaction_day", "balance"]].copy()
            tmp = tmp.rename(columns={"balance": f"balance__{acc_id}"})
            balance_frames.append(tmp)

        if not balance_frames:
            return

        merged = balance_frames[0].copy()
        for tmp in balance_frames[1:]:
            merged = merged.merge(tmp, on="transaction_day", how="outer")

        balance_cols = [c for c in merged.columns if c.startswith("balance__")]
        merged.sort_values("transaction_day", inplace=True)
        merged["total_balance"] = merged[balance_cols].sum(axis=1, min_count=1)
        merged["trac_days"] = (self.sample_day - merged["transaction_day"]).dt.days
        self.application_daily_total_balance_df = merged[["transaction_day", "total_balance", "trac_days"]].copy()

    def _get_application_daily_total_balance(self, window: int) -> pd.DataFrame:
        if self.application_daily_total_balance_df is None or self.application_daily_total_balance_df.empty:
            return pd.DataFrame(columns=["transaction_day", "total_balance", "trac_days"])
        out = self.application_daily_total_balance_df.loc[
            self.application_daily_total_balance_df["trac_days"] <= window,
            ["transaction_day", "total_balance", "trac_days"],
        ].copy()
        out.sort_values("transaction_day", inplace=True)
        return out

    def _sum_amount(self, df: pd.DataFrame, mask_col: str) -> float:
        if df.empty or mask_col not in df.columns:
            return 0.0
        s = df.loc[df[mask_col], "amount"]
        return float(s.sum()) if len(s) > 0 else 0.0

    def _count_amount(self, df: pd.DataFrame, mask_col: str) -> float:
        if df.empty or mask_col not in df.columns:
            return 0.0
        s = df.loc[df[mask_col], "amount"]
        return float(s.count()) if len(s) > 0 else 0.0

    def _max_amount(self, df: pd.DataFrame, mask_col: str) -> float:
        if df.empty or mask_col not in df.columns:
            return 0.0
        s = df.loc[df[mask_col], "amount"]
        return float(s.max()) if len(s) > 0 else 0.0

    def _avg_amount(self, df: pd.DataFrame, mask_col: str) -> float:
        if df.empty or mask_col not in df.columns:
            return np.nan
        s = df.loc[df[mask_col], "amount"]
        return float(s.mean()) if len(s) > 0 else np.nan

    def _balance_account_metrics(self, bank_account_id: str, window: int) -> Dict[str, float]:
        """
        基于“每卡每日最后一笔余额 + 无交易日前向填充”的日级余额序列计算余额特征。
        """
        daily_balance_df = self.account_daily_balance_cache.get(str(bank_account_id))
        if daily_balance_df is None or daily_balance_df.empty:
            return {
                f"balance_obs_cnt_{window}d": 0.0,
                f"balance_mean_{window}d": np.nan,
                f"balance_median_{window}d": np.nan,
                f"balance_max_{window}d": np.nan,
                f"balance_min_{window}d": np.nan,
                f"balance_std_{window}d": np.nan,
                f"balance_cv_{window}d": np.nan,
                f"balance_iqr_{window}d": np.nan,
                f"balance_non_negative_share_{window}d": np.nan,
                f"balance_negative_share_{window}d": np.nan,
                f"balance_ge1000_share_{window}d": np.nan,
                f"balance_ge2000_share_{window}d": np.nan,
                f"balance_ge5000_share_{window}d": np.nan,
            }

        bal = daily_balance_df.loc[daily_balance_df["trac_days"] <= window, "balance"].dropna()
        if bal.empty:
            return {
                f"balance_obs_cnt_{window}d": 0.0,
                f"balance_mean_{window}d": np.nan,
                f"balance_median_{window}d": np.nan,
                f"balance_max_{window}d": np.nan,
                f"balance_min_{window}d": np.nan,
                f"balance_std_{window}d": np.nan,
                f"balance_cv_{window}d": np.nan,
                f"balance_iqr_{window}d": np.nan,
                f"balance_non_negative_share_{window}d": np.nan,
                f"balance_negative_share_{window}d": np.nan,
                f"balance_ge1000_share_{window}d": np.nan,
                f"balance_ge2000_share_{window}d": np.nan,
                f"balance_ge5000_share_{window}d": np.nan,
            }

        mean_v = float(bal.mean())
        std_v = float(bal.std()) if len(bal) > 1 else 0.0
        q25 = float(bal.quantile(0.25))
        q75 = float(bal.quantile(0.75))

        return {
            f"balance_obs_cnt_{window}d": float(bal.count()),
            f"balance_mean_{window}d": mean_v,
            f"balance_median_{window}d": float(bal.median()),
            f"balance_max_{window}d": float(bal.max()),
            f"balance_min_{window}d": float(bal.min()),
            f"balance_std_{window}d": std_v,
            f"balance_cv_{window}d": safe_divide_scalar(std_v, abs(mean_v)),
            f"balance_iqr_{window}d": float(q75 - q25),
            f"balance_non_negative_share_{window}d": float((bal >= 0).mean()),
            f"balance_negative_share_{window}d": float((bal < 0).mean()),
            f"balance_ge1000_share_{window}d": float((bal >= self.config.balance_high_threshold_1).mean()),
            f"balance_ge2000_share_{window}d": float((bal >= self.config.balance_high_threshold_2).mean()),
            f"balance_ge5000_share_{window}d": float((bal >= self.config.balance_high_threshold_3).mean()),
        }

    # ========================================================
    # 账户层原子特征：先按卡算，再在 application 汇总
    # ========================================================
    def cal_account_features(self) -> Dict:
        features: Dict[str, float] = {}
        account_dfs = self._iter_account_dfs()

        features["bank_account_cnt"] = float(len(account_dfs))

        for window in self.time_windows:
            account_metric_list: List[Dict[str, float]] = []
            balance_metric_list: List[Dict[str, float]] = []
#
            for acc_id, acc_df in account_dfs:
                acc_win = acc_df.loc[acc_df["trac_days"] <= window].copy()
                metric = {}

                # ===== 收入 / 支出 / 负债 =====
                metric.update(self._sum_count_max(acc_win, "is_wage", "wage_income", window))
                metric.update(self._sum_count_max(acc_win, "is_income_tag", "income_tag", window))
                metric.update(self._sum_count_max(acc_win, "is_expense_tag", "expense_tag", window))
                metric.update(self._sum_count_max(acc_win, "is_rent", "rent", window))
                metric.update(self._sum_count_max(acc_win, "is_insurance", "insurance", window))
                metric.update(self._sum_count_max(acc_win, "is_daily_living", "daily_living", window))
                metric.update(self._sum_count_max(acc_win, "is_fixed_bill", "fixed_bill", window))

                metric.update(self._sum_count_max(acc_win, "is_liability_repayment", "liability_repayment", window))
                metric.update(self._sum_count_max(acc_win, "is_refund_or_reversal", "refund_or_reversal", window))
                metric.update(self._sum_count_max(acc_win, "is_liability_drawdown", "liability_drawdown", window))
                metric.update(self._sum_count_max(acc_win, "is_dishonours_fee", "dishonours_fee", window))
                metric.update(self._sum_count_max(acc_win, "is_bank_liability_drawdown", "bank_liability_drawdown", window))
                metric.update(self._sum_count_max(acc_win, "is_bank_liability_repayment", "bank_liability_repayment", window))
                metric.update(self._sum_count_max(acc_win, "is_large_bank_liability_drawdown", "large_bank_liability_drawdown", window))

                metric.update(self._sum_count_max(acc_win, "is_travel", "travel", window))
                metric.update(self._sum_count_max(acc_win, "is_premium_lifestyle", "premium_lifestyle", window))
                metric.update(self._sum_count_max(acc_win, "is_high_commitment", "high_commitment", window))
                metric.update(self._sum_count_max(acc_win, "is_showcase_spend", "showcase_spend", window))
                metric.update(self._sum_count_max(acc_win, "is_large_expense", "large_expense", window))
                metric.update(self._sum_count_max(acc_win, "is_large_travel", "large_travel", window))
                metric.update(self._sum_count_max(acc_win, "is_large_showcase_spend", "large_showcase_spend", window))

                metric[f"active_txn_day_cnt_{window}d"] = float(acc_win["transaction_date"].dt.floor("D").nunique()) if not acc_win.empty else 0.0
                account_metric_list.append(metric)
                balance_metric_list.append(self._balance_account_metrics(acc_id, window))

            agg: Dict[str, float] = {}
            if account_metric_list:
                all_keys = sorted({k for m in account_metric_list for k in m.keys()})
                for k in all_keys:
                    vals = [m.get(k, np.nan) for m in account_metric_list]
                    if "_amt_max_" in k:
                        arr = [v for v in vals if not pd.isna(v)]
                        agg[k] = float(max(arr)) if arr else 0.0
                    else:
                        agg[k] = float(np.nansum(vals)) if vals else 0.0
            else:
                default_prefixes = [
                    "wage_income", "income_tag", "expense_tag", "rent", "insurance", "daily_living", "fixed_bill",
                    "liability_repayment", "refund_or_reversal", "liability_drawdown", "dishonours_fee",
                    "bank_liability_drawdown", "bank_liability_repayment", "large_bank_liability_drawdown",
                    "travel", "premium_lifestyle", "high_commitment", "showcase_spend",
                    "large_expense", "large_travel", "large_showcase_spend",
                ]
                for prefix in default_prefixes:
                    agg[f"{prefix}_amt_sum_{window}d"] = 0.0
                    agg[f"{prefix}_cnt_{window}d"] = 0.0
                    agg[f"{prefix}_amt_max_{window}d"] = 0.0
                agg[f"active_txn_day_cnt_{window}d"] = 0.0

            # ===== 多卡余额聚合：先卡内统计，再申请级做 sum / avg / max / min =====
            balance_obs_cnt_vals = [m.get(f"balance_obs_cnt_{window}d", 0.0) for m in balance_metric_list]
            balance_mean_vals = [m.get(f"balance_mean_{window}d", np.nan) for m in balance_metric_list]
            balance_median_vals = [m.get(f"balance_median_{window}d", np.nan) for m in balance_metric_list]
            balance_max_vals = [m.get(f"balance_max_{window}d", np.nan) for m in balance_metric_list]
            balance_min_vals = [m.get(f"balance_min_{window}d", np.nan) for m in balance_metric_list]
            balance_std_vals = [m.get(f"balance_std_{window}d", np.nan) for m in balance_metric_list]
            balance_cv_vals = [m.get(f"balance_cv_{window}d", np.nan) for m in balance_metric_list]
            balance_iqr_vals = [m.get(f"balance_iqr_{window}d", np.nan) for m in balance_metric_list]
            balance_non_negative_share_vals = [m.get(f"balance_non_negative_share_{window}d", np.nan) for m in balance_metric_list]
            balance_negative_share_vals = [m.get(f"balance_negative_share_{window}d", np.nan) for m in balance_metric_list]
            balance_ge1000_share_vals = [m.get(f"balance_ge1000_share_{window}d", np.nan) for m in balance_metric_list]
            balance_ge2000_share_vals = [m.get(f"balance_ge2000_share_{window}d", np.nan) for m in balance_metric_list]
            balance_ge5000_share_vals = [m.get(f"balance_ge5000_share_{window}d", np.nan) for m in balance_metric_list]

            balance_mean_valid = [v for v in balance_mean_vals if not pd.isna(v)]
            balance_median_valid = [v for v in balance_median_vals if not pd.isna(v)]
            balance_max_valid = [v for v in balance_max_vals if not pd.isna(v)]
            balance_min_valid = [v for v in balance_min_vals if not pd.isna(v)]
            balance_std_valid = [v for v in balance_std_vals if not pd.isna(v)]
            balance_cv_valid = [v for v in balance_cv_vals if not pd.isna(v)]
            balance_iqr_valid = [v for v in balance_iqr_vals if not pd.isna(v)]
            balance_non_negative_share_valid = [v for v in balance_non_negative_share_vals if not pd.isna(v)]
            balance_negative_share_valid = [v for v in balance_negative_share_vals if not pd.isna(v)]
            balance_ge1000_share_valid = [v for v in balance_ge1000_share_vals if not pd.isna(v)]
            balance_ge2000_share_valid = [v for v in balance_ge2000_share_vals if not pd.isna(v)]
            balance_ge5000_share_valid = [v for v in balance_ge5000_share_vals if not pd.isna(v)]

            agg[f"balance_obs_cnt_{window}d"] = float(np.nansum(balance_obs_cnt_vals)) if balance_obs_cnt_vals else 0.0
            agg[f"balance_account_covered_cnt_{window}d"] = float(len(balance_mean_valid))
            agg[f"balance_mean_sum_acct_{window}d"] = float(np.nansum(balance_mean_valid)) if balance_mean_valid else np.nan
            agg[f"balance_mean_avg_acct_{window}d"] = float(np.nanmean(balance_mean_valid)) if balance_mean_valid else np.nan
            agg[f"balance_median_avg_acct_{window}d"] = float(np.nanmean(balance_median_valid)) if balance_median_valid else np.nan
            agg[f"balance_max_max_acct_{window}d"] = float(np.nanmax(balance_max_valid)) if balance_max_valid else np.nan
            agg[f"balance_min_min_acct_{window}d"] = float(np.nanmin(balance_min_valid)) if balance_min_valid else np.nan
            agg[f"balance_std_avg_acct_{window}d"] = float(np.nanmean(balance_std_valid)) if balance_std_valid else np.nan
            agg[f"balance_cv_avg_acct_{window}d"] = float(np.nanmean(balance_cv_valid)) if balance_cv_valid else np.nan
            agg[f"balance_iqr_avg_acct_{window}d"] = float(np.nanmean(balance_iqr_valid)) if balance_iqr_valid else np.nan
            agg[f"balance_non_negative_share_avg_acct_{window}d"] = float(np.nanmean(balance_non_negative_share_valid)) if balance_non_negative_share_valid else np.nan
            agg[f"balance_negative_share_avg_acct_{window}d"] = float(np.nanmean(balance_negative_share_valid)) if balance_negative_share_valid else np.nan
            agg[f"balance_ge1000_share_avg_acct_{window}d"] = float(np.nanmean(balance_ge1000_share_valid)) if balance_ge1000_share_valid else np.nan
            agg[f"balance_ge2000_share_avg_acct_{window}d"] = float(np.nanmean(balance_ge2000_share_valid)) if balance_ge2000_share_valid else np.nan
            agg[f"balance_ge5000_share_avg_acct_{window}d"] = float(np.nanmean(balance_ge5000_share_valid)) if balance_ge5000_share_valid else np.nan

            # ===== 均值类派生 =====
            agg[f"wage_income_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"wage_income_amt_sum_{window}d", 0.0),
                agg.get(f"wage_income_cnt_{window}d", 0.0),
            )
            agg[f"income_tag_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"income_tag_amt_sum_{window}d", 0.0),
                agg.get(f"income_tag_cnt_{window}d", 0.0),
            )
            agg[f"expense_tag_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"expense_tag_amt_sum_{window}d", 0.0),
                agg.get(f"expense_tag_cnt_{window}d", 0.0),
            )
            agg[f"rent_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"rent_amt_sum_{window}d", 0.0),
                agg.get(f"rent_cnt_{window}d", 0.0),
            )
            agg[f"insurance_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"insurance_amt_sum_{window}d", 0.0),
                agg.get(f"insurance_cnt_{window}d", 0.0),
            )
            agg[f"daily_living_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"daily_living_amt_sum_{window}d", 0.0),
                agg.get(f"daily_living_cnt_{window}d", 0.0),
            )
            agg[f"liability_repayment_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"liability_repayment_amt_sum_{window}d", 0.0),
                agg.get(f"liability_repayment_cnt_{window}d", 0.0),
            )
            agg[f"refund_or_reversal_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"refund_or_reversal_amt_sum_{window}d", 0.0),
                agg.get(f"refund_or_reversal_cnt_{window}d", 0.0),
            )
            agg[f"liability_drawdown_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"liability_drawdown_amt_sum_{window}d", 0.0),
                agg.get(f"liability_drawdown_cnt_{window}d", 0.0),
            )
            agg[f"dishonours_fee_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"dishonours_fee_amt_sum_{window}d", 0.0),
                agg.get(f"dishonours_fee_cnt_{window}d", 0.0),
            )
            agg[f"bank_liability_drawdown_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"bank_liability_drawdown_amt_sum_{window}d", 0.0),
                agg.get(f"bank_liability_drawdown_cnt_{window}d", 0.0),
            )
            agg[f"bank_liability_repayment_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"bank_liability_repayment_amt_sum_{window}d", 0.0),
                agg.get(f"bank_liability_repayment_cnt_{window}d", 0.0),
            )
            agg[f"travel_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"travel_amt_sum_{window}d", 0.0),
                agg.get(f"travel_cnt_{window}d", 0.0),
            )
            agg[f"premium_lifestyle_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"premium_lifestyle_amt_sum_{window}d", 0.0),
                agg.get(f"premium_lifestyle_cnt_{window}d", 0.0),
            )
            agg[f"high_commitment_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"high_commitment_amt_sum_{window}d", 0.0),
                agg.get(f"high_commitment_cnt_{window}d", 0.0),
            )
            agg[f"showcase_spend_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"showcase_spend_amt_sum_{window}d", 0.0),
                agg.get(f"showcase_spend_cnt_{window}d", 0.0),
            )
            agg[f"large_expense_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"large_expense_amt_sum_{window}d", 0.0),
                agg.get(f"large_expense_cnt_{window}d", 0.0),
            )
            agg[f"large_travel_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"large_travel_amt_sum_{window}d", 0.0),
                agg.get(f"large_travel_cnt_{window}d", 0.0),
            )
            agg[f"large_bank_liability_drawdown_amt_avg_{window}d"] = safe_divide_scalar(
                agg.get(f"large_bank_liability_drawdown_amt_sum_{window}d", 0.0),
                agg.get(f"large_bank_liability_drawdown_cnt_{window}d", 0.0),
            )

            features.update(agg)

        self.features.update(features)
        return features

    # ========================================================
    # application 直接特征：去重 / 最近性
    # ========================================================
    def cal_direct_features(self) -> Dict:
        features: Dict[str, float] = {}

        for window in self.time_windows:
            d = self._window_df(window)
            features[f"income_source_cnt_{window}d"] = self._distinct_count(d, d["is_income_tag"], "third_party") if not d.empty else 0.0
            features[f"wage_income_source_cnt_{window}d"] = self._distinct_count(d, d["is_wage"], "third_party") if not d.empty else 0.0
            features[f"bank_liability_drawdown_lender_cnt_{window}d"] = self._distinct_count(d, d["is_bank_liability_drawdown"], "third_party") if not d.empty else 0.0
            features[f"bank_liability_repayment_lender_cnt_{window}d"] = self._distinct_count(d, d["is_bank_liability_repayment"], "third_party") if not d.empty else 0.0
            features[f"large_bank_liability_drawdown_lender_cnt_{window}d"] = self._distinct_count(d, d["is_large_bank_liability_drawdown"], "third_party") if not d.empty else 0.0
            features[f"fixed_bill_category_cnt_{window}d"] = self._distinct_count(d, d["is_fixed_bill"], "category") if not d.empty else 0.0
            features[f"showcase_spend_category_cnt_{window}d"] = self._distinct_count(d, d["is_showcase_spend"], "category") if not d.empty else 0.0
            features[f"premium_lifestyle_merchant_cnt_{window}d"] = self._distinct_count(d, d["is_premium_lifestyle"], "third_party") if not d.empty else 0.0

            features[f"days_since_last_wage_income_{window}d"] = self._days_since_last(d, d["is_wage"]) if not d.empty else np.nan
            features[f"days_since_last_bank_liability_drawdown_{window}d"] = self._days_since_last(d, d["is_bank_liability_drawdown"]) if not d.empty else np.nan
            features[f"days_since_last_bank_liability_repayment_{window}d"] = self._days_since_last(d, d["is_bank_liability_repayment"]) if not d.empty else np.nan
            features[f"days_since_last_liability_drawdown_{window}d"] = self._days_since_last(d, d["is_liability_drawdown"]) if not d.empty else np.nan
            features[f"days_since_last_liability_repayment_{window}d"] = self._days_since_last(d, d["is_liability_repayment"]) if not d.empty else np.nan
            features[f"days_since_last_travel_spend_{window}d"] = self._days_since_last(d, d["is_travel"]) if not d.empty else np.nan
            features[f"days_since_last_large_expense_{window}d"] = self._days_since_last(d, d["is_large_expense"]) if not d.empty else np.nan
            features[f"days_since_last_showcase_spend_{window}d"] = self._days_since_last(d, d["is_showcase_spend"]) if not d.empty else np.nan

        self.features.update(features)
        return features

    # ========================================================
    # 连续性 / 活跃月 / 波动性
    # ========================================================
    def cal_continuity_features(self) -> Dict:
        features: Dict[str, float] = {}
        long_wd = self.long_window
        long_df = self._window_df(long_wd)

        if long_df.empty:
            feature_names = [
                f"wage_income_active_month_cnt_{long_wd}d",
                f"wage_income_consecutive_month_cnt_{long_wd}d",
                f"wage_income_amt_std_{long_wd}d",
            ]
            label_names = [
                "rent", "insurance", "daily_living", "travel", "premium_lifestyle", "high_commitment",
                "showcase_spend", "liability_repayment", "liability_drawdown", "bank_liability_drawdown",
            ]
            for name in feature_names:
                features[name] = 0.0 if "_cnt_" in name else np.nan
            for label in label_names:
                features[f"{label}_consecutive_month_cnt_{long_wd}d"] = 0.0
                features[f"{label}_active_month_cnt_{long_wd}d"] = 0.0
        else:
            features[f"wage_income_active_month_cnt_{long_wd}d"] = self._monthly_count(long_df, long_df["is_wage"])
            features[f"wage_income_consecutive_month_cnt_{long_wd}d"] = self._max_consecutive_month_cnt(long_df, long_df["is_wage"])
            wage_amounts = long_df.loc[long_df["is_wage"], "amount"]
            features[f"wage_income_amt_std_{long_wd}d"] = float(wage_amounts.std()) if len(wage_amounts) > 1 else np.nan

            label_mask_map = {
                "rent": long_df["is_rent"],
                "insurance": long_df["is_insurance"],
                "daily_living": long_df["is_daily_living"],
                "travel": long_df["is_travel"],
                "premium_lifestyle": long_df["is_premium_lifestyle"],
                "high_commitment": long_df["is_high_commitment"],
                "showcase_spend": long_df["is_showcase_spend"],
                "liability_repayment": long_df["is_liability_repayment"],
                "liability_drawdown": long_df["is_liability_drawdown"],
                "bank_liability_drawdown": long_df["is_bank_liability_drawdown"],
            }
            for label, mask in label_mask_map.items():
                features[f"{label}_consecutive_month_cnt_{long_wd}d"] = self._max_consecutive_month_cnt(long_df, mask)
                features[f"{label}_active_month_cnt_{long_wd}d"] = self._monthly_count(long_df, mask)

        self.features.update(features)
        return features

    # ========================================================
    # 最终衍生特征：依赖 account + direct + continuity
    # ========================================================
    def cal_final_features(self) -> Dict:
        if "bank_account_cnt" not in self.features:
            self.cal_account_features()
        long_wd = self.long_window
        if f"wage_income_active_month_cnt_{long_wd}d" not in self.features:
            self.cal_continuity_features()
        if f"income_source_cnt_{self.time_windows[0]}d" not in self.features:
            self.cal_direct_features()

        features: Dict[str, float] = {}

        if f"wage_income_amt_std_{long_wd}d" in self.features and f"wage_income_amt_avg_{long_wd}d" in self.features:
            features[f"wage_income_amt_cv_{long_wd}d"] = safe_divide_scalar(
                self.features.get(f"wage_income_amt_std_{long_wd}d", np.nan),
                self.features.get(f"wage_income_amt_avg_{long_wd}d", np.nan),
            )

        for window in self.time_windows:
            wage_sum = self.features.get(f"wage_income_amt_sum_{window}d", 0.0)
            wage_cnt = self.features.get(f"wage_income_cnt_{window}d", 0.0)
            income_sum = self.features.get(f"income_tag_amt_sum_{window}d", 0.0)
            income_cnt = self.features.get(f"income_tag_cnt_{window}d", 0.0)
            expense_sum = self.features.get(f"expense_tag_amt_sum_{window}d", 0.0)
            rent_sum = self.features.get(f"rent_amt_sum_{window}d", 0.0)
            insurance_sum = self.features.get(f"insurance_amt_sum_{window}d", 0.0)
            daily_living_sum = self.features.get(f"daily_living_amt_sum_{window}d", 0.0)
            fixed_bill_sum = self.features.get(f"fixed_bill_amt_sum_{window}d", 0.0)
            liability_repayment_sum = self.features.get(f"liability_repayment_amt_sum_{window}d", 0.0)
            refund_or_reversal_sum = self.features.get(f"refund_or_reversal_amt_sum_{window}d", 0.0)
            liability_drawdown_sum = self.features.get(f"liability_drawdown_amt_sum_{window}d", 0.0)
            dishonours_fee_sum = self.features.get(f"dishonours_fee_amt_sum_{window}d", 0.0)
            bank_liability_drawdown_sum = self.features.get(f"bank_liability_drawdown_amt_sum_{window}d", 0.0)
            bank_liability_repayment_sum = self.features.get(f"bank_liability_repayment_amt_sum_{window}d", 0.0)
            large_bank_liability_drawdown_sum = self.features.get(f"large_bank_liability_drawdown_amt_sum_{window}d", 0.0)
            travel_sum = self.features.get(f"travel_amt_sum_{window}d", 0.0)
            premium_sum = self.features.get(f"premium_lifestyle_amt_sum_{window}d", 0.0)
            showcase_sum = self.features.get(f"showcase_spend_amt_sum_{window}d", 0.0)
            commitment_sum = self.features.get(f"high_commitment_amt_sum_{window}d", 0.0)
            active_txn_day_cnt = self.features.get(f"active_txn_day_cnt_{window}d", 0.0)
            balance_mean_sum_acct = self.features.get(f"balance_mean_sum_acct_{window}d", np.nan)
            balance_mean_avg_acct = self.features.get(f"balance_mean_avg_acct_{window}d", np.nan)
            balance_max_max_acct = self.features.get(f"balance_max_max_acct_{window}d", np.nan)

            features[f"wage_income_amt_share_in_income_tag_{window}d"] = safe_divide_scalar(wage_sum, income_sum)
            features[f"wage_income_cnt_share_in_income_tag_{window}d"] = safe_divide_scalar(wage_cnt, income_cnt)
            features[f"income_to_expense_ratio_{window}d"] = safe_divide_scalar(income_sum, expense_sum)
            features[f"net_income_after_expense_amt_{window}d"] = float(income_sum - expense_sum)
            features[f"net_income_after_expense_ratio_{window}d"] = safe_divide_scalar(income_sum - expense_sum, income_sum)

            features[f"rent_to_wage_ratio_{window}d"] = safe_divide_scalar(rent_sum, wage_sum)
            features[f"insurance_to_wage_ratio_{window}d"] = safe_divide_scalar(insurance_sum, wage_sum)
            features[f"daily_living_to_wage_ratio_{window}d"] = safe_divide_scalar(daily_living_sum, wage_sum)
            features[f"fixed_bill_to_wage_ratio_{window}d"] = safe_divide_scalar(fixed_bill_sum, wage_sum)
            features[f"travel_to_wage_ratio_{window}d"] = safe_divide_scalar(travel_sum, wage_sum)
            features[f"premium_lifestyle_to_wage_ratio_{window}d"] = safe_divide_scalar(premium_sum, wage_sum)
            features[f"showcase_spend_to_wage_ratio_{window}d"] = safe_divide_scalar(showcase_sum, wage_sum)
            features[f"high_commitment_to_wage_ratio_{window}d"] = safe_divide_scalar(commitment_sum, wage_sum)

            features[f"liability_repayment_to_wage_ratio_{window}d"] = safe_divide_scalar(liability_repayment_sum, wage_sum)
            features[f"liability_drawdown_to_wage_ratio_{window}d"] = safe_divide_scalar(liability_drawdown_sum, wage_sum)
            features[f"bank_liability_drawdown_to_wage_ratio_{window}d"] = safe_divide_scalar(bank_liability_drawdown_sum, wage_sum)
            features[f"bank_liability_repayment_to_wage_ratio_{window}d"] = safe_divide_scalar(bank_liability_repayment_sum, wage_sum)
            features[f"refund_or_reversal_to_drawdown_ratio_{window}d"] = safe_divide_scalar(refund_or_reversal_sum, liability_drawdown_sum)
            features[f"liability_repayment_to_drawdown_ratio_{window}d"] = safe_divide_scalar(liability_repayment_sum, liability_drawdown_sum)
            features[f"bank_liability_drawdown_share_in_liability_drawdown_{window}d"] = safe_divide_scalar(bank_liability_drawdown_sum, liability_drawdown_sum)
            features[f"large_bank_liability_drawdown_share_in_bank_liability_drawdown_{window}d"] = safe_divide_scalar(
                large_bank_liability_drawdown_sum, bank_liability_drawdown_sum
            )
            features[f"dishonours_fee_to_liability_repayment_ratio_{window}d"] = safe_divide_scalar(dishonours_fee_sum, liability_repayment_sum)
            features[f"bank_liability_repayment_to_bank_liability_drawdown_ratio_{window}d"] = safe_divide_scalar(
                bank_liability_repayment_sum, bank_liability_drawdown_sum
            )

            features[f"balance_mean_sum_acct_to_income_ratio_{window}d"] = safe_divide_scalar(balance_mean_sum_acct, income_sum)
            features[f"balance_mean_avg_acct_to_wage_avg_ratio_{window}d"] = safe_divide_scalar(
                balance_mean_avg_acct, self.features.get(f"wage_income_amt_avg_{window}d", np.nan)
            )
            features[f"balance_max_max_acct_to_wage_avg_ratio_{window}d"] = safe_divide_scalar(
                balance_max_max_acct, self.features.get(f"wage_income_amt_avg_{window}d", np.nan)
            )
            features[f"active_txn_day_density_{window}d"] = safe_divide_scalar(active_txn_day_cnt, float(window))

        self.features.update(features)
        return features


    # ========================================================
    # 新增资质特征：资产厚度 / 收入质量 / 高消费倾向
    # ========================================================
    def cal_enhanced_worthiness_features(self) -> Dict:
        features: Dict[str, float] = {}
        # 长窗口 = 配置中最长的窗口(如 182d),特征名随配置生成
        window = self.long_window

        long_df = self._window_df(window)
        balance_df = self._get_application_daily_total_balance(window)

        # ===== 一、资产厚度类特征 =====
        if balance_df.empty or balance_df["total_balance"].dropna().empty:
            features[f"avg_month_end_balance_{window}d"] = np.nan
            features[f"min_month_end_balance_{window}d"] = np.nan
            features[f"post_salary_7d_min_balance_ratio_{window}d"] = np.nan
            features[f"balance_above_1x_monthly_fixed_spend_days_share_{window}d"] = np.nan
            features[f"max_balance_drawdown_{window}d"] = np.nan
            features[f"drawdown_recovery_days_{window}d"] = np.nan
            features[f"reserve_months_{window}d"] = np.nan
        else:
            bal = balance_df.dropna(subset=["total_balance"]).copy()
            bal["month_period"] = bal["transaction_day"].dt.to_period("M")
            month_end_balances = bal.groupby("month_period", as_index=False).tail(1)

            features[f"avg_month_end_balance_{window}d"] = float(month_end_balances["total_balance"].mean()) if not month_end_balances.empty else np.nan
            features[f"min_month_end_balance_{window}d"] = float(month_end_balances["total_balance"].min()) if not month_end_balances.empty else np.nan

            wage_events = long_df.loc[long_df["is_wage"] & long_df["amount"].gt(0), ["transaction_day", "amount"]].copy()
            salary_ratios = []
            if not wage_events.empty:
                for _, wage_row in wage_events.iterrows():
                    obs = bal.loc[
                        (bal["transaction_day"] >= wage_row["transaction_day"])
                        & (bal["transaction_day"] <= wage_row["transaction_day"] + pd.Timedelta(days=7)),
                        "total_balance",
                    ].dropna()
                    if not obs.empty:
                        salary_ratios.append(safe_divide_scalar(float(obs.min()), float(wage_row["amount"])))
            features[f"post_salary_7d_min_balance_ratio_{window}d"] = float(np.nanmean(salary_ratios)) if len(salary_ratios) > 0 else np.nan

            monthly_fixed_spend = safe_divide_scalar(
                self._sum_amount(long_df, "is_fixed_bill"),
                6.0,
            )
            if pd.isna(monthly_fixed_spend) or monthly_fixed_spend <= 0:
                features[f"balance_above_1x_monthly_fixed_spend_days_share_{window}d"] = np.nan
            else:
                features[f"balance_above_1x_monthly_fixed_spend_days_share_{window}d"] = float((bal["total_balance"] > monthly_fixed_spend).mean())

            balance_series = bal["total_balance"].astype(float).reset_index(drop=True)
            date_series = bal["transaction_day"].reset_index(drop=True)
            running_peak = balance_series.cummax()
            peak_day_series = []
            current_peak = np.nan
            current_peak_day = pd.NaT
            for cur_day, cur_balance in zip(date_series, balance_series):
                if pd.isna(cur_balance):
                    peak_day_series.append(current_peak_day)
                    continue
                if pd.isna(current_peak) or cur_balance >= current_peak:
                    current_peak = cur_balance
                    current_peak_day = cur_day
                peak_day_series.append(current_peak_day)
            peak_day_series = pd.Series(peak_day_series)

            valid_peak = running_peak.where(running_peak > 0)
            drawdown = (running_peak - balance_series) / valid_peak
            if drawdown.notna().any():
                trough_idx = int(drawdown.fillna(-np.inf).idxmax())
                features[f"max_balance_drawdown_{window}d"] = float(drawdown.iloc[trough_idx])

                trough_day = date_series.iloc[trough_idx]
                peak_day = peak_day_series.iloc[trough_idx]
                peak_value = running_peak.iloc[trough_idx]
                recovery_target = peak_value * 0.8 if pd.notna(peak_value) else np.nan

                if pd.notna(trough_day) and pd.notna(recovery_target):
                    recovery_slice = bal.loc[bal["transaction_day"] > trough_day].copy()
                    recovery_hit = recovery_slice.loc[recovery_slice["total_balance"] >= recovery_target, "transaction_day"]
                    if len(recovery_hit) > 0:
                        features[f"drawdown_recovery_days_{window}d"] = float((recovery_hit.iloc[0] - trough_day).days)
                    else:
                        features[f"drawdown_recovery_days_{window}d"] = np.nan
                else:
                    features[f"drawdown_recovery_days_{window}d"] = np.nan
            else:
                features[f"max_balance_drawdown_{window}d"] = np.nan
                features[f"drawdown_recovery_days_{window}d"] = np.nan

            avg_monthly_fixed_commitment = safe_divide_scalar(
                self._sum_amount(long_df, "is_fixed_commitment"),
                6.0,
            )
            features[f"reserve_months_{window}d"] = safe_divide_scalar(
                float(balance_series.mean()),
                avg_monthly_fixed_commitment,
            )

        # ===== 二、收入质量类特征 =====
        wage_df = long_df.loc[long_df["is_wage"]].copy()
        income_df = long_df.loc[long_df["is_income_tag"]].copy()

        wage_days = (
            wage_df[["transaction_day"]]
            .dropna()
            .drop_duplicates()
            .sort_values("transaction_day")
        )
        if len(wage_days) >= 2:
            intervals = wage_days["transaction_day"].diff().dt.days.dropna()
            features[f"salary_interval_std_days_{window}d"] = float(intervals.std()) if len(intervals) > 1 else 0.0
        else:
            features[f"salary_interval_std_days_{window}d"] = np.nan

        if not wage_df.empty:
            wage_df["income_source_key"] = wage_df["income_source_key"].replace("", np.nan)
            wage_by_source = (
                wage_df.dropna(subset=["income_source_key"])
                .groupby("income_source_key", dropna=False)["amount"]
                .sum()
                .sort_values(ascending=False)
            )
            if len(wage_by_source) > 0:
                primary_employer = wage_by_source.index[0]
                primary_wage_df = wage_df.loc[wage_df["income_source_key"].eq(primary_employer)].copy()
                features[f"primary_employer_streak_months_{window}d"] = float(
                    max_consecutive_months(primary_wage_df["month_period"])
                )
            else:
                features[f"primary_employer_streak_months_{window}d"] = 0.0

            wage_mean = float(wage_df["amount"].mean()) if len(wage_df) > 0 else np.nan
            wage_std = float(wage_df["amount"].std()) if len(wage_df) > 1 else 0.0
            features[f"salary_amount_cv_{window}d"] = safe_divide_scalar(wage_std, wage_mean)
            features[f"salary_active_months_{window}d"] = float(wage_df["month_period"].nunique())
        else:
            features[f"primary_employer_streak_months_{window}d"] = 0.0
            features[f"salary_amount_cv_{window}d"] = np.nan
            features[f"salary_active_months_{window}d"] = 0.0

        if not income_df.empty:
            income_df["income_source_key"] = income_df["income_source_key"].replace("", np.nan)
            income_by_source = (
                income_df.dropna(subset=["income_source_key"])
                .groupby("income_source_key", dropna=False)["amount"]
                .sum()
                .sort_values(ascending=False)
            )
            total_income_amt = float(income_df["amount"].sum())
            primary_income_amt = float(income_by_source.iloc[0]) if len(income_by_source) > 0 else 0.0
            features[f"primary_income_concentration_{window}d"] = safe_divide_scalar(primary_income_amt, total_income_amt)
            features[f"non_salary_income_share_{window}d"] = safe_divide_scalar(
                total_income_amt - float(wage_df["amount"].sum()) if not wage_df.empty else total_income_amt,
                total_income_amt,
            )
            features[f"income_source_count_{window}d"] = float(
                income_df.loc[income_df["income_source_key"].notna() & income_df["income_source_key"].ne(""), "income_source_key"].nunique()
            )
        else:
            features[f"primary_income_concentration_{window}d"] = np.nan
            features[f"non_salary_income_share_{window}d"] = np.nan
            features[f"income_source_count_{window}d"] = 0.0

        # ===== 三、高消费倾向类特征 =====
#         features["txn_worthiness_cnt_luxury_retail_180d"] = self._count_amount(long_df, "is_luxury_retail_merchant")
#         features["txn_worthiness_amt_luxury_retail_total_180d"] = self._sum_amount(long_df, "is_luxury_retail_merchant")
#         features["txn_worthiness_amt_luxury_retail_max_180d"] = self._max_amount(long_df, "is_luxury_retail_merchant")

#         features["txn_worthiness_cnt_premium_car_180d"] = self._count_amount(long_df, "is_premium_car_merchant")
#         features["txn_worthiness_amt_premium_car_total_180d"] = self._sum_amount(long_df, "is_premium_car_merchant")

#         features["txn_worthiness_cnt_premium_real_estate_180d"] = self._count_amount(long_df, "is_premium_real_estate_merchant")
#         features["txn_worthiness_amt_premium_real_estate_total_180d"] = self._sum_amount(long_df, "is_premium_real_estate_merchant")

#         features["txn_worthiness_cnt_high_end_hotel_180d"] = self._count_amount(long_df, "is_high_end_hotel_merchant")
#         features["txn_worthiness_amt_high_end_hotel_total_180d"] = self._sum_amount(long_df, "is_high_end_hotel_merchant")

#         features["txn_worthiness_cnt_premium_airline_180d"] = self._count_amount(long_df, "is_premium_airline_merchant")
#         features["txn_worthiness_amt_premium_airline_avg_180d"] = self._avg_amount(long_df, "is_premium_airline_merchant")

#         features["txn_worthiness_cnt_elite_education_180d"] = self._count_amount(long_df, "is_elite_education_merchant")
#         features["txn_worthiness_amt_elite_education_total_180d"] = self._sum_amount(long_df, "is_elite_education_merchant")

#         features["txn_worthiness_cnt_high_value_healthcare_180d"] = self._count_amount(long_df, "is_high_value_healthcare_merchant")
#         features["txn_worthiness_amt_high_value_healthcare_max_180d"] = self._max_amount(long_df, "is_high_value_healthcare_merchant")

#         features["txn_worthiness_cnt_investment_property_insurance_180d"] = self._count_amount(long_df, "is_investment_property_insurance_merchant")
#         features["txn_worthiness_amt_investment_property_insurance_total_180d"] = self._sum_amount(long_df, "is_investment_property_insurance_merchant")

#         features["txn_worthiness_cnt_loan_protection_insurance_180d"] = self._count_amount(long_df, "is_loan_protection_insurance_merchant")

#         features["txn_worthiness_amt_high_value_retail_total_180d"] = self._sum_amount(long_df, "is_high_value_retail_merchant")

#         luxury_travel_amt = (
#             self._sum_amount(long_df, "is_premium_airline_merchant")
#             + self._sum_amount(long_df, "is_high_end_hotel_merchant")
#         )
#         total_expense_amt = self._sum_amount(long_df, "is_expense_tag")
#         fixed_commitment_amt = self._sum_amount(long_df, "is_fixed_commitment")
#         discretionary_expense_amt = total_expense_amt - fixed_commitment_amt
#         if discretionary_expense_amt <= 0:
#             features["txn_worthiness_pct_luxury_travel_of_discretionary_longd"] = 0.0
#         else:
#             features["txn_worthiness_pct_luxury_travel_of_discretionary_longd"] = float(luxury_travel_amt / discretionary_expense_amt)

#         gambling_amt = self._sum_amount(long_df, "is_gambling_merchant")
#         features["txn_worthiness_pct_gambling_of_total_180d"] = (
#             float(gambling_amt / total_expense_amt) if total_expense_amt > 0 else 0.0
#         )

        self.features.update(features)
        return features


    def _window_df_between(self, min_days: int, max_days: int) -> pd.DataFrame:
        if self.df.empty:
            return self.df.iloc[0:0].copy()
        return self.df.loc[(self.df["trac_days"] >= min_days) & (self.df["trac_days"] <= max_days)].copy()

    def _get_discretionary_total(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        necessity_mask = df["is_expense_tag"] & df["category"].isin(HIGH_WORTHINESS_NECESSITY_CATEGORIES)
        total_expense = float(df.loc[df["is_expense_tag"], "amount"].sum())
        necessity_expense = float(df.loc[necessity_mask, "amount"].sum())
        discretionary_expense = total_expense - necessity_expense
        return float(discretionary_expense) if discretionary_expense > 0 else 0.0

    def _build_monthly_agg(self, df: pd.DataFrame, mask: pd.Series, window: int) -> pd.DataFrame:
        if pd.isna(getattr(self, "sample_day", pd.NaT)):
            return pd.DataFrame(columns=["month_period", "amt", "cnt"])
        sample_month = self.sample_day.to_period("M")
        month_cnt = int(max(1, np.ceil(window / 30.0)))
        periods = [sample_month - i for i in range(month_cnt - 1, -1, -1)]
        out = pd.DataFrame({"month_period": pd.PeriodIndex(periods, freq="M")})
        if df.empty:
            out["amt"] = 0.0
            out["cnt"] = 0.0
            return out
        tmp = (
            df.loc[mask]
            .groupby("month_period", as_index=False)
            .agg(amt=("amount", "sum"), cnt=("amount", "count"))
        )
        out = out.merge(tmp, on="month_period", how="left")
        out["amt"] = pd.to_numeric(out["amt"], errors="coerce").fillna(0.0)
        out["cnt"] = pd.to_numeric(out["cnt"], errors="coerce").fillna(0.0)
        return out

    def _calc_interval_features(self, df: pd.DataFrame, mask: pd.Series, prefix: str, window: int) -> Dict[str, float]:
        out = {}
        if df.empty:
            out[f"{prefix}_interval_avg_days_{window}d"] = np.nan
            out[f"{prefix}_interval_std_days_{window}d"] = np.nan
            out[f"{prefix}_span_days_{window}d"] = np.nan
            return out

        day_values = (
            df.loc[mask, "transaction_day"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        if len(day_values) <= 1:
            out[f"{prefix}_interval_avg_days_{window}d"] = np.nan
            out[f"{prefix}_interval_std_days_{window}d"] = np.nan
            out[f"{prefix}_span_days_{window}d"] = 0.0 if len(day_values) == 1 else np.nan
            return out

        intervals = day_values.diff().dt.days.dropna()
        out[f"{prefix}_interval_avg_days_{window}d"] = float(intervals.mean()) if len(intervals) > 0 else np.nan
        out[f"{prefix}_interval_std_days_{window}d"] = float(intervals.std()) if len(intervals) > 1 else 0.0
        out[f"{prefix}_span_days_{window}d"] = float((day_values.iloc[-1] - day_values.iloc[0]).days)
        return out

    def _calc_amount_features(self, df: pd.DataFrame, mask: pd.Series, prefix: str, window: int) -> Dict[str, float]:
        out = {}
        amount_s = pd.to_numeric(df.loc[mask, "amount"], errors="coerce").dropna()
        out[f"{prefix}_amt_total_{window}d"] = float(amount_s.sum()) if len(amount_s) > 0 else 0.0
        out[f"{prefix}_cnt_total_{window}d"] = float(amount_s.count()) if len(amount_s) > 0 else 0.0
        out[f"{prefix}_amt_max_{window}d"] = float(amount_s.max()) if len(amount_s) > 0 else np.nan
        out[f"{prefix}_amt_min_{window}d"] = nonzero_min(amount_s)
        out[f"{prefix}_amt_std_{window}d"] = safe_std(amount_s)
        out[f"{prefix}_amt_cv_{window}d"] = safe_cv(amount_s)

        monthly_df = self._build_monthly_agg(df, mask, window)
        month_cnt = float(len(monthly_df)) if len(monthly_df) > 0 else np.nan
        out[f"{prefix}_amt_monthly_avg_{window}d"] = safe_divide_scalar(out[f"{prefix}_amt_total_{window}d"], month_cnt)
        out[f"{prefix}_cnt_monthly_avg_{window}d"] = safe_divide_scalar(out[f"{prefix}_cnt_total_{window}d"], month_cnt)
        out[f"{prefix}_amt_monthly_std_{window}d"] = safe_std(monthly_df["amt"]) if len(monthly_df) > 0 else np.nan
        out[f"{prefix}_amt_monthly_cv_{window}d"] = safe_cv(monthly_df["amt"]) if len(monthly_df) > 0 else np.nan
        out[f"{prefix}_amt_monthly_max_{window}d"] = float(monthly_df["amt"].max()) if len(monthly_df) > 0 else np.nan
        out[f"{prefix}_amt_monthly_min_nonzero_{window}d"] = nonzero_min(monthly_df["amt"]) if len(monthly_df) > 0 else np.nan
        return out

    def _calc_share_features(
        self,
        df_long: pd.DataFrame,
        df_cur: pd.DataFrame,
        df_prev: pd.DataFrame,
        mask_long: pd.Series,
        mask_cur: pd.Series,
        mask_prev: pd.Series,
        prefix: str,
        include_discretionary: bool,
        window: int,
        mom_window: int,
    ) -> Dict[str, float]:
        out = {}
        amt_long = float(df_long.loc[mask_long, "amount"].sum()) if not df_long.empty else 0.0
        amt_cur = float(df_cur.loc[mask_cur, "amount"].sum()) if not df_cur.empty else 0.0
        amt_prev = float(df_prev.loc[mask_prev, "amount"].sum()) if not df_prev.empty else 0.0

        total_expense_long = float(df_long.loc[df_long["is_expense_tag"], "amount"].sum()) if not df_long.empty else 0.0
        total_expense_cur = float(df_cur.loc[df_cur["is_expense_tag"], "amount"].sum()) if not df_cur.empty else 0.0
        total_expense_prev = float(df_prev.loc[df_prev["is_expense_tag"], "amount"].sum()) if not df_prev.empty else 0.0

        out[f"{prefix}_share_in_expense_{window}d"] = safe_divide_scalar(amt_long, total_expense_long)
        cur_share_expense = safe_divide_scalar(amt_cur, total_expense_cur)
        prev_share_expense = safe_divide_scalar(amt_prev, total_expense_prev)
        out[f"{prefix}_amt_mom_growth_pct_{mom_window}d"] = safe_growth_pct(amt_cur, amt_prev)
        out[f"{prefix}_share_in_expense_mom_diff_{mom_window}d"] = (
            float(cur_share_expense - prev_share_expense)
            if pd.notna(cur_share_expense) and pd.notna(prev_share_expense)
            else np.nan
        )

        if include_discretionary:
            discretionary_long = self._get_discretionary_total(df_long)
            discretionary_cur = self._get_discretionary_total(df_cur)
            discretionary_prev = self._get_discretionary_total(df_prev)
            out[f"{prefix}_share_in_discretionary_{window}d"] = safe_divide_scalar(amt_long, discretionary_long)
            cur_share_discretionary = safe_divide_scalar(amt_cur, discretionary_cur)
            prev_share_discretionary = safe_divide_scalar(amt_prev, discretionary_prev)
            out[f"{prefix}_share_in_discretionary_mom_diff_{mom_window}d"] = (
                float(cur_share_discretionary - prev_share_discretionary)
                if pd.notna(cur_share_discretionary) and pd.notna(prev_share_discretionary)
                else np.nan
            )
        else:
            out[f"{prefix}_share_in_discretionary_{window}d"] = np.nan
            out[f"{prefix}_share_in_discretionary_mom_diff_{mom_window}d"] = np.nan

        return out

    def _calc_merchant_diversity_features(
        self,
        df: pd.DataFrame,
        total_mask: pd.Series,
        prefix: str,
        window: int,
        premium_mask: Optional[pd.Series] = None,
        general_mask: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        out = {}
        out[f"{prefix}_merchant_distinct_total_{window}d"] = self._distinct_count(df, total_mask, "third_party")
        if premium_mask is not None:
            out[f"{prefix}_merchant_distinct_premium_{window}d"] = self._distinct_count(df, premium_mask, "third_party")
            out[f"{prefix}_merchant_distinct_general_{window}d"] = self._distinct_count(df, general_mask, "third_party")
            premium_hit_cnt = float(df.loc[premium_mask, "amount"].count()) if not df.empty else 0.0
            general_hit_cnt = float(df.loc[general_mask, "amount"].count()) if not df.empty else 0.0
            out[f"{prefix}_distinct_tier_cnt_{window}d"] = float((premium_hit_cnt > 0) + (general_hit_cnt > 0))
        return out

    def cal_high_net_worth_consumption_features(self) -> Dict:
        """
        新增：
        1. 严格按 8 大类别 + 指定商户池识别高净值消费；
        2. 输出金额 / 周期 / 多样性 / 30d 环比 / 可支配消费占比；
        3. 保留原有字段不变，仅新增字段。
        """
        features: Dict[str, float] = {}
        window = self.long_window  # 长窗口(如 182d)
        mom_window = self.mom_window  # 环比窗口(如 28d)

        df_long = self._window_df(window)
        df_cur = self._window_df(self.mom_window)  # 环比窗口(如 28d)
        df_prev = self._window_df_between(self.mom_window + 1, 2 * self.mom_window)

        total_expense_long = float(df_long.loc[df_long["is_expense_tag"], "amount"].sum()) if not df_long.empty else 0.0
        discretionary_long = self._get_discretionary_total(df_long)

        category_hit_total_cnt = 0
        category_hit_premium_cnt = 0
        total_high_worth_amt = 0.0
        total_high_worth_cnt = 0.0
        total_high_worth_cur_amt = 0.0
        total_high_worth_prev_amt = 0.0

        combo_core_lifestyle = 0
        combo_shopping_lifestyle = 0
        combo_asset_commitment = 0

        category_result_map: Dict[str, Dict[str, float]] = {}

        for worthiness_key, pool_conf in HIGH_WORTHINESS_MERCHANT_POOLS.items():
            prefix = f"txn_worthiness_{worthiness_key}"
            total_mask_long = df_long[f"is_worthiness_{worthiness_key}"] if not df_long.empty else pd.Series(dtype=bool)
            total_mask_cur = df_cur[f"is_worthiness_{worthiness_key}"] if not df_cur.empty else pd.Series(dtype=bool)
            total_mask_prev = df_prev[f"is_worthiness_{worthiness_key}"] if not df_prev.empty else pd.Series(dtype=bool)

            features.update(self._calc_amount_features(df_long, total_mask_long, prefix, window))
            features.update(self._calc_interval_features(df_long, total_mask_long, prefix, window))
            features.update(
                self._calc_share_features(
                    df_long=df_long,
                    df_cur=df_cur,
                    df_prev=df_prev,
                    mask_long=total_mask_long,
                    mask_cur=total_mask_cur,
                    mask_prev=total_mask_prev,
                    prefix=prefix,
                    include_discretionary=(pool_conf["category"] not in HIGH_WORTHINESS_RIGID_CATEGORIES),
                    mom_window=self.mom_window,
                    window=window,
                )
            )

            if "premium" in pool_conf:
                premium_mask_long = df_long[f"is_worthiness_{worthiness_key}_premium"] if not df_long.empty else pd.Series(dtype=bool)
                general_mask_long = df_long[f"is_worthiness_{worthiness_key}_general"] if not df_long.empty else pd.Series(dtype=bool)
                premium_mask_cur = df_cur[f"is_worthiness_{worthiness_key}_premium"] if not df_cur.empty else pd.Series(dtype=bool)
                premium_mask_prev = df_prev[f"is_worthiness_{worthiness_key}_premium"] if not df_prev.empty else pd.Series(dtype=bool)
                general_mask_cur = df_cur[f"is_worthiness_{worthiness_key}_general"] if not df_cur.empty else pd.Series(dtype=bool)
                general_mask_prev = df_prev[f"is_worthiness_{worthiness_key}_general"] if not df_prev.empty else pd.Series(dtype=bool)

                features.update(self._calc_amount_features(df_long, premium_mask_long, f"{prefix}_premium", window))
                features.update(self._calc_amount_features(df_long, general_mask_long, f"{prefix}_general", window))
                features.update(
                    self._calc_share_features(
                        df_long=df_long,
                        df_cur=df_cur,
                        df_prev=df_prev,
                        mask_long=premium_mask_long,
                        mask_cur=premium_mask_cur,
                        mask_prev=premium_mask_prev,
                        prefix=f"{prefix}_premium",
                        include_discretionary=(pool_conf["category"] not in HIGH_WORTHINESS_RIGID_CATEGORIES),
                        mom_window=self.mom_window,
                        window=window,
                    )
                )
                features.update(
                    self._calc_share_features(
                        df_long=df_long,
                        df_cur=df_cur,
                        df_prev=df_prev,
                        mask_long=general_mask_long,
                        mask_cur=general_mask_cur,
                        mask_prev=general_mask_prev,
                        prefix=f"{prefix}_general",
                        include_discretionary=(pool_conf["category"] not in HIGH_WORTHINESS_RIGID_CATEGORIES),
                        mom_window=self.mom_window,
                        window=window,
                    )
                )

                features.update(
                    self._calc_merchant_diversity_features(
                        df=df_long,
                        total_mask=total_mask_long,
                        premium_mask=premium_mask_long,
                        general_mask=general_mask_long,
                        prefix=prefix,
                        window=window,
                    )
                )
                premium_amt = float(df_long.loc[premium_mask_long, "amount"].sum()) if not df_long.empty else 0.0
                premium_cnt = float(df_long.loc[premium_mask_long, "amount"].count()) if not df_long.empty else 0.0
                total_amt = float(df_long.loc[total_mask_long, "amount"].sum()) if not df_long.empty else 0.0
                total_cnt = float(df_long.loc[total_mask_long, "amount"].count()) if not df_long.empty else 0.0
                features[f"{prefix}_premium_amt_share_in_category_{window}d"] = safe_divide_scalar(premium_amt, total_amt)
                features[f"{prefix}_premium_cnt_share_in_category_{window}d"] = safe_divide_scalar(premium_cnt, total_cnt)
                premium_hit = premium_cnt > 0
            else:
                features.update(
                    self._calc_merchant_diversity_features(
                        df=df_long,
                        total_mask=total_mask_long,
                        prefix=prefix,
                        window=window,
                    )
                )
                premium_hit = float(df_long.loc[total_mask_long, "amount"].count()) > 0

            total_amt = float(df_long.loc[total_mask_long, "amount"].sum()) if not df_long.empty else 0.0
            total_cnt = float(df_long.loc[total_mask_long, "amount"].count()) if not df_long.empty else 0.0
            cur_amt = float(df_cur.loc[total_mask_cur, "amount"].sum()) if not df_cur.empty else 0.0
            prev_amt = float(df_prev.loc[total_mask_prev, "amount"].sum()) if not df_prev.empty else 0.0

            category_result_map[worthiness_key] = {
                "amt": total_amt,
                "cnt": total_cnt,
                "premium_hit": float(premium_hit),
            }

            if total_cnt > 0:
                category_hit_total_cnt += 1
            if premium_hit:
                category_hit_premium_cnt += 1

            total_high_worth_amt += total_amt
            total_high_worth_cnt += total_cnt
            total_high_worth_cur_amt += cur_amt
            total_high_worth_prev_amt += prev_amt

        combo_core_lifestyle = int(category_result_map.get("automotive", {}).get("premium_hit", 0) > 0) \
            + int(category_result_map.get("travel", {}).get("premium_hit", 0) > 0) \
            + int(category_result_map.get("insurance", {}).get("premium_hit", 0) > 0)

        combo_shopping_lifestyle = int(category_result_map.get("retail", {}).get("premium_hit", 0) > 0) \
            + int(category_result_map.get("dining_out", {}).get("premium_hit", 0) > 0) \
            + int(category_result_map.get("department_stores", {}).get("premium_hit", 0) > 0)

        combo_asset_commitment = int(category_result_map.get("rent", {}).get("premium_hit", 0) > 0) \
            + int(category_result_map.get("insurance", {}).get("premium_hit", 0) > 0) \
            + int(category_result_map.get("automotive", {}).get("premium_hit", 0) > 0)

        features[f"txn_worthiness_category_distinct_total_{window}d"] = float(category_hit_total_cnt)
        features[f"txn_worthiness_category_distinct_premium_{window}d"] = float(category_hit_premium_cnt)
        features[f"txn_worthiness_amt_total_{window}d"] = float(total_high_worth_amt)
        features[f"txn_worthiness_cnt_total_{window}d"] = float(total_high_worth_cnt)
        features[f"txn_worthiness_share_in_expense_{window}d"] = safe_divide_scalar(total_high_worth_amt, total_expense_long)
        features[f"txn_worthiness_share_in_discretionary_{window}d"] = safe_divide_scalar(total_high_worth_amt, discretionary_long)
        features[f"txn_worthiness_amt_mom_growth_pct_{mom_window}d"] = safe_growth_pct(
            total_high_worth_cur_amt, total_high_worth_prev_amt
        )

        cur_total_expense = float(df_cur.loc[df_cur["is_expense_tag"], "amount"].sum()) if not df_cur.empty else 0.0
        prev_total_expense = float(df_prev.loc[df_prev["is_expense_tag"], "amount"].sum()) if not df_prev.empty else 0.0
        cur_discretionary = self._get_discretionary_total(df_cur)
        prev_discretionary = self._get_discretionary_total(df_prev)

        cur_share_expense = safe_divide_scalar(total_high_worth_cur_amt, cur_total_expense)
        prev_share_expense = safe_divide_scalar(total_high_worth_prev_amt, prev_total_expense)
        features[f"txn_worthiness_share_in_expense_mom_diff_{mom_window}d"] = (
            float(cur_share_expense - prev_share_expense)
            if pd.notna(cur_share_expense) and pd.notna(prev_share_expense)
            else np.nan
        )

        cur_share_discretionary = safe_divide_scalar(total_high_worth_cur_amt, cur_discretionary)
        prev_share_discretionary = safe_divide_scalar(total_high_worth_prev_amt, prev_discretionary)
        features[f"txn_worthiness_share_in_discretionary_mom_diff_{mom_window}d"] = (
            float(cur_share_discretionary - prev_share_discretionary)
            if pd.notna(cur_share_discretionary) and pd.notna(prev_share_discretionary)
            else np.nan
        )

        features[f"txn_worthiness_discretionary_share_of_expense_{window}d"] = safe_divide_scalar(
            discretionary_long, total_expense_long
        )
        features[f"txn_worthiness_combo_core_lifestyle_category_cnt_{window}d"] = float(combo_core_lifestyle)
        features[f"txn_worthiness_combo_shopping_lifestyle_category_cnt_{window}d"] = float(combo_shopping_lifestyle)
        features[f"txn_worthiness_combo_asset_commitment_category_cnt_{window}d"] = float(combo_asset_commitment)

        self.features.update(features)
        return features


    # ========================================================
    # 输出
    # ========================================================
    def generate_all_features(self, feature_groups: Optional[List[str]] = None, include_metadata: bool = True) -> pd.DataFrame:
        if feature_groups is None:
            feature_groups = [
                "cal_account_features",
                "cal_direct_features",
                "cal_continuity_features",
                "cal_enhanced_worthiness_features",
                "cal_high_net_worth_consumption_features",
                "cal_final_features",
            ]

        self.features = {}
        for feature in feature_groups:
            if not hasattr(self, feature):
                raise ValueError(f"Unsupported feature group: {feature}")
            method = getattr(self, feature)
            method()

        result = dict(self.features)
        if include_metadata:
            result["user_id"] = self.user_id
            #result["application_id"] = self.application_id
            result["sample_datetime"] = self.sample_datetime

#         df = pd.DataFrame([result])
#         #base_columns = ["user_id", "application_id", "sample_datetime"]
#         base_columns = ["user_id", "sample_datetime"]
#         existing_base = [col for col in base_columns if col in df.columns]
#         feature_cols = [col for col in df.columns if col not in existing_base]

#         # 统一加前缀；metadata 不加
#         prefix = "bank_txn_worthiness_"
#         renamed_feature_cols = []
#         rename_map = {}
#         for col in feature_cols:
#             if col.startswith(prefix):
#                 new_col = col
#             elif col.startswith("txn_worthiness_"):
#                 new_col = f"bank_{col}"
#             else:
#                 new_col = f"{prefix}{col}"
#             rename_map[col] = new_col
#             renamed_feature_cols.append(new_col)
#         df = df.rename(columns=rename_map)

#         df = df[existing_base + renamed_feature_cols]
#         df.columns = df.columns.str.strip().str.replace(" ", "", regex=False).str.lower()

        df = pd.DataFrame([result])
        base_columns = ["user_id", "sample_datetime"]
        existing_base = [col for col in base_columns if col in df.columns]
        feature_cols = [col for col in df.columns if col not in existing_base]

        prefix = "bank_txn_worthiness_"
        rename_map = {}
        for col in feature_cols:
            if col.startswith(prefix):
                new_col = col
            elif col.startswith("txn_worthiness_"):
                new_col = f"bank_{col}"
            else:
                new_col = f"{prefix}{col}"
            rename_map[col] = new_col

        df = df.rename(columns=rename_map)

        # ✅ 基于 rename 后的实际列名构建最终列顺序
        current_columns = df.columns.tolist()
        new_feature_cols = [rename_map.get(col, col) for col in feature_cols]
        final_cols = existing_base + [col for col in new_feature_cols if col in current_columns]
        
        
        df = df[final_cols]
        df.columns = df.columns.str.strip().str.replace(" ", "", regex=False).str.lower()

        # ✅ 新增：移除重复列名（保留第一个出现的列）
        df = df.loc[:, ~df.columns.duplicated()]

        df.columns = df.columns.str.strip().str.replace(" ", "", regex=False).str.lower()
        return df



# ============================================================
# 对外调用函数：与 test_code 类似
# ============================================================
def generate_premium_customer_features(
    df: pd.DataFrame = None,
    feature_groups: Optional[List[str]] = None,
    include_metadata: bool = True,
    time_windows: Optional[List[int]] = None,
    mapping_file: Optional[str] = None,
    config_overrides: Optional[Dict] = None,
) -> pd.DataFrame:
    engineer = SingleApplicationPremiumCustomerFeatureEngineer(
        df=df,
        time_windows=time_windows or [30, 60, 90, 180],
        mapping_file=mapping_file,
        config_overrides=config_overrides,
    )
    return engineer.generate_all_features(feature_groups=feature_groups, include_metadata=include_metadata)


# ============================================================
# 命令行：单申请 CSV -> 单行宽表 CSV
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build worthiness features for a single application.")
    parser.add_argument("--input_csv", required=True, help="One CSV file containing one application's bank transactions")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--windows", nargs="+", type=int, default=[30, 60, 90, 180], help="Lookback windows")
    parser.add_argument(
        "--feature_groups",
        nargs="*",
        default=None,
        help="Optional feature groups, e.g. cal_account_features cal_direct_features cal_continuity_features cal_final_features",
    )
    parser.add_argument("--mapping_file", default=None, help="Optional finv mapping file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _read_single_csv(args.input_csv)
    result = generate_premium_customer_features(
        df=df,
        feature_groups=args.feature_groups,
        include_metadata=True,
        time_windows=args.windows,
        mapping_file=args.mapping_file,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    result.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
