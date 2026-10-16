import glob
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


current_dir = os.path.dirname(os.path.abspath(__file__))
_MAPPING_FILE_PATH = os.path.join(current_dir, "reference/industry_config/industry_config.csv")
_INDUSTRY_CODE_MAP = {
    "Accommodation and Food Services": 1,
    "Bank-issued Payroll Distribution": 2,
    "Cleaning/Security/Property Management": 3,
    "Construction": 4,
    "Education and Training": 5,
    "Financial and Insurance Services": 6,
    "Government/Public Administration": 7,
    "Health Care and Social Assistance": 8,
    "Labor Dispatch/Recruitment": 9,
    "Manufacturing": 10,
    "Mining": 11,
    "Retail Trade": 12,
    "Social Services/Non-Profit Organizations": 13,
    "Sports/Entertainment/Gambling": 14,
    "Technical Services/Information Media": 15,
    "Transport, Postal and Warehousing": 16,
    "Unknown": 17,
}


def _load_mappings(mapping_file):
    df = pd.read_csv(mapping_file).fillna("")
    trx_exact_rules = {}
    trx_fuzzy_rules = []
    all_trx_fuzzy_keywords = []
    category_fuzzy_rules = OrderedDict()

    industries = set(df["industry"].dropna().unique())
    if "Unknown" not in industries:
        industries.add("Unknown")
    industries = sorted(industries)

    for _, row in df.iterrows():
        rule_type = str(row.get("rule_type", "")).strip().lower()
        match_type = str(row.get("match_type", "")).strip().lower()
        match_values = str(row.get("match_values", "")).strip()
        ind = str(row.get("industry", "")).strip()
        occ = str(row.get("occupation", "")).strip()
        if not match_values:
            continue
        values_list = [v.strip() for v in match_values.split(",") if v.strip()]

        if rule_type == "trx_type":
            if match_type == "exact":
                for v in values_list:
                    trx_exact_rules[v.lower()] = {"industry": ind, "occupation": occ}
            elif match_type == "fuzzy":
                trx_fuzzy_rules.append(
                    {
                        "keywords": [v.lower() for v in values_list],
                        "industry": ind,
                        "occupation": occ,
                    }
                )
                all_trx_fuzzy_keywords.extend([v.lower() for v in values_list])
        elif rule_type == "category":
            if ind not in category_fuzzy_rules:
                category_fuzzy_rules[ind] = []
            norm_values = [_normalize_text(v) for v in values_list]
            category_fuzzy_rules[ind].append(
                {
                    "occupation": occ,
                    "keywords": norm_values,
                }
            )

    if "Unknown" not in category_fuzzy_rules:
        category_fuzzy_rules["Unknown"] = []
    category_fuzzy_rules.move_to_end("Unknown")

    return trx_exact_rules, trx_fuzzy_rules, all_trx_fuzzy_keywords, category_fuzzy_rules, industries


def _normalize_text(x: str, keep_chars: str = "&'-.()/") -> str:
    x = str(x).upper().strip()
    allowed = r"[^A-Z0-9\s" + re.escape(keep_chars) + r"]"
    x = re.sub(allowed, " ", x)
    x = re.sub(r"\s+", " ", x)
    return x


if os.path.exists(_MAPPING_FILE_PATH):
    (
        _TRX_EXACT_RULES,
        _TRX_FUZZY_RULES,
        _ALL_TRX_FUZZY_KEYWORDS,
        _CATEGORY_FUZZY_RULES,
        _ALL_INDUSTRIES,
    ) = _load_mappings(_MAPPING_FILE_PATH)
else:
    raise FileNotFoundError(f"mapping file not found: {_MAPPING_FILE_PATH}")


def _infer_industry_and_occupation(row):
    trx_type = row["trx_type_std"]
    category = row["category_std"]
    match_text = _normalize_text(row["match_text"])

    if trx_type in _TRX_EXACT_RULES:
        rule = _TRX_EXACT_RULES[trx_type]
        return [rule["industry"]], [rule["occupation"]]

    if category == "":
        for rule in _TRX_FUZZY_RULES:
            if any(kw in trx_type for kw in rule["keywords"]):
                return [rule["industry"]], [rule["occupation"]]

    matched_industry = None
    matched_occupation = None

    for ind, rules in _CATEGORY_FUZZY_RULES.items():
        if ind == "Unknown":
            continue
        for rule in rules:
            for kw in rule["keywords"]:
                if kw in match_text:
                    matched_industry = ind
                    matched_occupation = rule["occupation"] if rule["occupation"] else None
                    break
            if matched_industry is not None:
                break
        if matched_industry is not None:
            break

    industry_list = [matched_industry] if matched_industry else ["Unknown"]
    occupation_list = [matched_occupation] if matched_occupation else ["Unknown"]
    return industry_list, occupation_list


def _analyze_pay_cycle(dates):
    if len(dates) < 2:
        return np.nan, np.nan, "insufficient_data"
    dates = pd.Series(dates).dropna().sort_values().reset_index(drop=True)
    intervals = dates.diff().dt.days.dropna().astype(int)
    median_interval = intervals.median()
    std_interval = intervals.std()
    if median_interval <= 0:
        pattern = "irregular"
    elif 5 <= median_interval <= 9:
        pattern = "weekly"
    elif 12 <= median_interval <= 16:
        pattern = "fortnightly"
    elif 27 <= median_interval <= 33:
        pattern = "monthly"
    else:
        pattern = "irregular"
    return median_interval, std_interval, pattern


def _unique_flatten(series_of_lists):
    result, seen = [], set()
    for item in series_of_lists:
        if isinstance(item, list):
            for x in item:
                if x not in seen:
                    seen.add(x)
                    result.append(x)
    return result


def _get_employer_amt_dict(group):
    amt_by_employer = group.groupby("third_party")["amount"].sum().to_dict()
    if not amt_by_employer:
        return {}, None, np.nan
    total_amt = sum(amt_by_employer.values())
    primary_employer = max(amt_by_employer, key=amt_by_employer.get)
    primary_ratio = amt_by_employer[primary_employer] / total_amt if total_amt > 0 else np.nan
    return amt_by_employer, primary_employer, primary_ratio


def _aggregate_user_wages(wages_df):
    prefix = "bank_txn_industry_"
    if wages_df.empty:
        base_series = {
            prefix + "wages_txn_cnt": np.nan,
            prefix + "wages_total_amt": np.nan,
            prefix + "employer_cnt": np.nan,
            prefix + "primary_employer_ratio": np.nan,
            prefix + "pay_interval_median": np.nan,
            prefix + "pay_interval_std": np.nan,
            prefix + "primary_industry_amt": np.nan,
            prefix + "primary_industry_code": np.nan,
        }
        for pattern in ["irregular", "weekly", "fortnightly", "monthly"]:
            base_series[f"{prefix}pattern_{pattern}"] = 0
        return pd.Series(base_series)

    wages_txn_cnt = len(wages_df)
    wages_total_amt = wages_df["amount"].sum()
    employer_cnt = wages_df["third_party"].nunique()
    _, _, primary_employer_ratio = _get_employer_amt_dict(wages_df)

    dates = wages_df["transaction_date"].dropna().sort_values().reset_index(drop=True)
    median_interval, std_interval, pay_cycle_pattern = _analyze_pay_cycle(dates)

    exploded = wages_df[["amount", "industry_list"]].explode("industry_list")
    if not exploded.empty:
        primary_industry_series = (
            exploded.groupby("industry_list")["amount"].sum().sort_values(ascending=False)
        )
        primary_industry = primary_industry_series.index[0] if len(primary_industry_series) > 0 else None
        primary_industry_amt = primary_industry_series.iloc[0] if len(primary_industry_series) > 0 else 0.0
    else:
        primary_industry = None
        primary_industry_amt = 0.0

    primary_industry_code = _INDUSTRY_CODE_MAP.get(
        primary_industry or "Unknown",
        _INDUSTRY_CODE_MAP["Unknown"],
    )

    pattern_irregular = 1 if pay_cycle_pattern == "irregular" else 0
    pattern_weekly = 1 if pay_cycle_pattern == "weekly" else 0
    pattern_fortnightly = 1 if pay_cycle_pattern == "fortnightly" else 0
    pattern_monthly = 1 if pay_cycle_pattern == "monthly" else 0

    result = {
        prefix + "wages_txn_cnt": wages_txn_cnt,
        prefix + "wages_total_amt": wages_total_amt,
        prefix + "employer_cnt": employer_cnt,
        prefix + "primary_employer_ratio": primary_employer_ratio,
        prefix + "pay_interval_median": median_interval,
        prefix + "pay_interval_std": std_interval,
        prefix + "primary_industry_amt": primary_industry_amt,
        prefix + "primary_industry_code": primary_industry_code,
        prefix + "pattern_irregular": pattern_irregular,
        prefix + "pattern_weekly": pattern_weekly,
        prefix + "pattern_fortnightly": pattern_fortnightly,
        prefix + "pattern_monthly": pattern_monthly,
    }
    return pd.Series(result)


def analyze_user_industry(transactions_df):
    df = transactions_df.copy()

    user_id = df["user_id"].iloc[0] if "user_id" in df.columns else None
    sample_dt = df["sample_datetime"].iloc[0] if "sample_datetime" in df.columns else None

    required_cols = ["transaction_date", "amount", "dr_cr", "category", "third_party", "text", "trx_type"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    for col in ["category", "dr_cr", "third_party", "text", "trx_type"]:
        df[col] = df[col].fillna("").astype(str)

    df["category_std"] = df["category"].str.strip().str.lower()
    df["dr_cr_std"] = df["dr_cr"].str.strip().str.lower()
    df["trx_type_std"] = df["trx_type"].str.strip().str.lower()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    wages_cond = (df["category_std"] == "wages") & (df["dr_cr_std"] == "credit")
    if _ALL_TRX_FUZZY_KEYWORDS:
        gig_cond = (
            (df["dr_cr_std"] == "credit")
            & (df["category_std"] == "")
            & (df["trx_type_std"].str.contains("|".join(_ALL_TRX_FUZZY_KEYWORDS), na=False))
        )
    else:
        gig_cond = pd.Series(False, index=df.index)

    income_df = df[wages_cond | gig_cond].copy()

    if income_df.empty:
        empty_summary = _aggregate_user_wages(income_df)
        data = {
            "user_id": user_id,
            "sample_datetime": sample_dt,
        }
        data.update(empty_summary.to_dict())
        result_df = pd.DataFrame([data])
    else:
        income_df["third_party_std"] = income_df["third_party"].apply(_normalize_text)
        income_df["trx_type_std_norm"] = income_df["trx_type"].apply(_normalize_text)
        income_df["match_text"] = income_df["third_party_std"].fillna("").str.strip()

        inferred = income_df.apply(_infer_industry_and_occupation, axis=1)
        income_df["industry_list"] = inferred.apply(lambda x: x[0])
        income_df["occupation_list"] = inferred.apply(lambda x: x[1])

        user_summary = _aggregate_user_wages(income_df)

        data = {
            "user_id": user_id,
            "sample_datetime": sample_dt,
        }
        data.update(user_summary.to_dict())
        result_df = pd.DataFrame([data])

    result_df.columns = result_df.columns.str.strip().str.replace(" ", "", regex=False).str.lower()
    return result_df


def process_all_folders(base_dir, output_path):
    all_results = []
    csv_files = glob.glob(f"{base_dir}/**/*.csv", recursive=True)

    for file_path in tqdm(csv_files, desc="processing csv files"):
        try:
            df = pd.read_csv(file_path)
            group_col = "application_id"
            for _, group_df in df.groupby(group_col):
                result_row = analyze_user_industry(group_df)
                if "application_id" not in result_row.columns and "application_id" in group_df.columns:
                    result_row.insert(0, "application_id", group_df["application_id"].iloc[0])
                all_results.append(result_row)
        except Exception as e:
            print(f"failed to process {file_path}: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"saved output to {output_path}")
