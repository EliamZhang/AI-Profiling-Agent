# coding: utf-8
import os
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from . import txn_tool
from .input_helpers import (
    balance_from_slice,
    enrich_transactions_with_account_type,
    filter_future_records,
    raw_txn_from_slice,
)


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_DICT_CSV_PATH = os.path.join(PROJECT_ROOT, "dict.csv")
LENDER_DAY_LIST = [None, 1, 3, 7, 14, 28, 56, 84, 168, 182]
LENDER_CATEGORY_LIST = ["personal loan", "BNPL", "cash/wage advance", "bank"]
NUMERIC_FILL = -999999.99999
LENDER_FILL = -1
ALL_FEATURE_MODULES = (
    "balance",
    "category",
    "income",
    "expense",
    "surplus",
    "worthiness",
    "dishonour",
    "industry",
    "lender",
)
MODULE_PREFIXES = {
    "balance": ("bank_txn_balance_",),
    "category": ("bank_txn_category_",),
    "income": ("bank_txn_income_",),
    "expense": ("bank_txn_expense_",),
    "surplus": ("bank_txn_surplus_",),
    "worthiness": ("bank_txn_worthiness_",),
    "dishonour": ("bank_txn_dishonour_",),
    "industry": ("bank_txn_industry_",),
    "lender": ("bank_txn_lender_",),
}


def attach_sample_keys(df, user_id, send_time):
    if df is None or df.shape[0] == 0:
        return None

    df = df.copy()
    df["user_id"] = user_id
    df["sample_datetime"] = pd.to_datetime(send_time)
    return df


def normalize_numeric_feature_frame_generic(df, fill_value):
    if df is None or df.shape[0] == 0:
        return None

    df = df.copy()
    df = df.replace(["", " ", "None", "nan"], np.nan)
    cols = df.columns.difference(["user_id", "sample_datetime"])
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[cols] = df[cols].fillna(fill_value).clip(lower=fill_value)
    return df


def normalize_numeric_feature_frame(df, fill_value):
    if df is None or df.shape[0] == 0:
        return None

    cols = df.columns.difference(["user_id", "sample_datetime"])
    if len(cols) == 0:
        return df.copy()

    dtypes = df.dtypes.to_dict()
    # to_numeric leaves integer/bool columns untouched; dtypes it would treat
    # differently (datetime, timedelta, ...) fall back to the column-wise version.
    if any(
        dtypes[col] != object
        and not pd.api.types.is_float_dtype(dtypes[col])
        and not pd.api.types.is_integer_dtype(dtypes[col])
        and not pd.api.types.is_bool_dtype(dtypes[col])
        for col in cols
    ):
        return normalize_numeric_feature_frame_generic(df, fill_value)

    convert_cols = [
        col
        for col in cols
        if not (
            pd.api.types.is_integer_dtype(dtypes[col])
            or pd.api.types.is_bool_dtype(dtypes[col])
        )
    ]
    if not convert_cols:
        return df.copy()

    # Per-column pandas ops dominate on frames with thousands of feature columns,
    # so parse/fill/clip the whole block as one numpy array and rebuild by concat.
    block = df[convert_cols].replace(["", " ", "None", "nan"], np.nan)
    values = (
        pd.to_numeric(
            pd.Series(block.to_numpy(dtype="object").ravel()),
            errors="coerce",
        )
        .to_numpy(dtype="float64")
        .reshape(df.shape[0], len(convert_cols))
    )
    values = np.where(np.isnan(values), fill_value, values)
    np.maximum(values, fill_value, out=values)

    convert_set = set(convert_cols)
    keep_cols = [col for col in df.columns if col not in convert_set]
    converted = pd.DataFrame(values, columns=convert_cols, index=df.index)
    return pd.concat([df[keep_cols], converted], axis=1)


def build_lender_features(raw_data, user_id, send_time):
    alv = txn_tool.TxnListSampleVar(
        user_id=user_id,
        send_time=send_time,
        raw_data=raw_data,
    )

    alv.clean_txn_step1()
    alv.create_lender_cate_list(
        lender_cate_mapping_df=txn_tool.load_refdata.load_third_category("third_party_cate")
    )
    alv.grouped_app_cate_list()

    alv.calc_vars_summary_disburse_stat(day_list=LENDER_DAY_LIST)
    alv.calc_vars_summary_disburse_jgs_stat(day_list=LENDER_DAY_LIST)
    alv.calc_vars_disburse_stat(
        cate_list=LENDER_CATEGORY_LIST,
        day_list=LENDER_DAY_LIST,
    )
    alv.calc_vars_disburse_single_stat(cate_list=LENDER_CATEGORY_LIST)
    alv.calc_vars_disburse_jgs_stat(
        cate_list=LENDER_CATEGORY_LIST,
        day_list=LENDER_DAY_LIST,
    )

    alv.calc_vars_summary_repay_stat(day_list=LENDER_DAY_LIST)
    alv.calc_vars_summary_repay_jgs_stat(day_list=LENDER_DAY_LIST)
    alv.calc_vars_repay_stat(
        cate_list=LENDER_CATEGORY_LIST,
        day_list=LENDER_DAY_LIST,
    )
    alv.calc_vars_repay_single_stat(cate_list=LENDER_CATEGORY_LIST)
    alv.calc_vars_repay_jgs_stat(
        cate_list=LENDER_CATEGORY_LIST,
        day_list=LENDER_DAY_LIST,
    )

    competitor_info = txn_tool.load_refdata.load_third_category("competitor_list")
    competitor_list = competitor_info.loc[
        competitor_info["installed_sample_prop"] >= 0.05, "third_party"
    ].tolist()
    alv.calc_vars_repay_competitor_stat(
        competitor_list=competitor_list,
        day_list=LENDER_DAY_LIST,
    )
    alv.calc_vars_disburse_competitor_stat(
        competitor_list=competitor_list,
        day_list=LENDER_DAY_LIST,
    )

    return pd.json_normalize(alv.variables)


def normalize_feature_modules(feature_modules):
    if feature_modules is None:
        return ALL_FEATURE_MODULES

    if isinstance(feature_modules, str):
        normalized = feature_modules.strip().lower()
        if normalized == "all":
            return ALL_FEATURE_MODULES
        modules = [item.strip().lower() for item in feature_modules.split(",") if item.strip()]
    else:
        modules = []
        for item in feature_modules:
            if item is None:
                continue
            normalized = str(item).strip().lower()
            if normalized == "all":
                return ALL_FEATURE_MODULES
            if normalized:
                modules.append(normalized)

    invalid_modules = sorted(set(modules) - set(ALL_FEATURE_MODULES))
    if invalid_modules:
        raise ValueError(
            f"Unsupported feature_modules: {invalid_modules}. "
            f"Expected one or more of {list(ALL_FEATURE_MODULES)} or 'all'."
        )

    deduped_modules = []
    seen = set()
    for module_name in modules:
        if module_name not in seen:
            deduped_modules.append(module_name)
            seen.add(module_name)
    return tuple(deduped_modules)


def is_feature_in_modules(feature_name, selected_modules):
    if feature_name in {"user_id", "application_id", "sample_datetime"}:
        return True

    prefixes = []
    for module_name in selected_modules:
        prefixes.extend(MODULE_PREFIXES[module_name])
    return feature_name.startswith(tuple(prefixes))


def build_feature_frames(raw_data, eod_date, user_id, send_time, selected_modules):
    feature_frames = []

    if raw_data is not None and raw_data.shape[0] > 0:
        raw_module_builders = {
            "category": lambda: normalize_numeric_feature_frame(
                attach_sample_keys(
                    txn_tool.txn_cate.compute_bank_features(
                        df=raw_data,
                        categories=None,
                    ),
                    user_id,
                    send_time,
                ),
                NUMERIC_FILL,
            ),
            "income": lambda: normalize_numeric_feature_frame(
                attach_sample_keys(
                    txn_tool.txn_income_v1_1.generate_income_feature(
                        df=raw_data,
                        feature_groups=None,
                    ),
                    user_id,
                    send_time,
                ),
                NUMERIC_FILL,
            ),
            "expense": lambda: normalize_numeric_feature_frame(
                attach_sample_keys(
                    txn_tool.txn_expense.generate_expense_feature(
                        df=raw_data,
                        feature_groups=None,
                    ),
                    user_id,
                    send_time,
                ),
                NUMERIC_FILL,
            ),
            "surplus": lambda: normalize_numeric_feature_frame(
                attach_sample_keys(
                    txn_tool.txn_surplus.generate_surplus_features(
                        df=raw_data,
                        feature_groups=None,
                    ),
                    user_id,
                    send_time,
                ),
                NUMERIC_FILL,
            ),
            "worthiness": lambda: normalize_numeric_feature_frame(
                attach_sample_keys(
                    txn_tool.txn_worthiness.generate_premium_customer_features(
                        df=raw_data,
                        feature_groups=None,
                        time_windows=[7, 14, 28, 56, 84, 168, 182],
                    ),
                    user_id,
                    send_time,
                ),
                NUMERIC_FILL,
            ),
            "dishonour": lambda: normalize_numeric_feature_frame(
                attach_sample_keys(
                    txn_tool.txn_dishonour.compute_dishonour_features(
                        df=raw_data,
                        lender_cate_mapping_df=txn_tool.load_refdata.load_third_category("third_party_cate"),
                        send_time=send_time,
                    ),
                    user_id,
                    send_time,
                ),
                NUMERIC_FILL,
            ),
            "industry": lambda: normalize_numeric_feature_frame(
                attach_sample_keys(
                    txn_tool.txn_industry.analyze_user_industry(raw_data),
                    user_id,
                    send_time,
                ),
                NUMERIC_FILL,
            ),
            "lender": lambda: normalize_numeric_feature_frame(
                attach_sample_keys(
                    build_lender_features(raw_data, user_id, send_time),
                    user_id,
                    send_time,
                ),
                LENDER_FILL,
            ),
        }

        for module_name in selected_modules:
            builder = raw_module_builders.get(module_name)
            if builder is not None:
                feature_frames.append(builder())

    if (
        "balance" in selected_modules
        and eod_date is not None
        and eod_date.shape[0] > 0
    ):
        feature_frames.append(
            normalize_numeric_feature_frame(
                attach_sample_keys(
                    txn_tool.txn_eod.compute_balance_timeseries_features(
                        df_input=eod_date,
                        categories=None,
                    ),
                    user_id,
                    send_time,
                ),
                NUMERIC_FILL,
            )
        )

    return feature_frames


def data_prepare(user_id, raw_data, eod_date, send_time, applicationid, selected_modules):
    df_master = pd.DataFrame(
        [
            {
                "user_id": user_id,
                "application_id": applicationid,
                "sample_datetime": pd.to_datetime(send_time),
            }
        ]
    )

    feature_frames = build_feature_frames(
        raw_data=raw_data,
        eod_date=eod_date,
        user_id=user_id,
        send_time=send_time,
        selected_modules=selected_modules,
    )

    for df in feature_frames:
        if df is not None and df.shape[0] > 0:
            df_master = df_master.merge(
                df,
                on=["user_id", "sample_datetime"],
                how="left",
            )

    float_cols = df_master.columns[df_master.dtypes == "float64"]
    if len(float_cols):
        rounded = np.round(df_master[float_cols].to_numpy(dtype="float64"), 5)
        float_set = set(float_cols)
        keep_cols = [col for col in df_master.columns if col not in float_set]
        float_frame = pd.DataFrame(rounded, columns=float_cols, index=df_master.index)
        df_master = pd.concat([df_master[keep_cols], float_frame], axis=1)

    return df_master


@lru_cache()
def load_fixed_feature_config(dict_csv_path):
    dict_csv_path = os.path.abspath(dict_csv_path)
    df = pd.read_csv(dict_csv_path)
    df["feature"] = df["feature"].astype(str)
    fixed_feature_cols = df["feature"].tolist()
    default_map = dict(zip(df["feature"], df["default"]))
    return fixed_feature_cols, default_map


def filter_feature_config(fixed_feature_cols, default_map, selected_modules):
    filtered_cols = [
        col for col in fixed_feature_cols if is_feature_in_modules(col, selected_modules)
    ]
    filtered_default_map = {col: default_map[col] for col in filtered_cols}
    return filtered_cols, filtered_default_map


def run_model_feature(input_vars, dict_csv_path=None, feature_modules="all"):
    user_id = input_vars["userId"]
    application_id = input_vars["applicationId"]
    send_time = str(pd.to_datetime(input_vars["flowTime"][:19]))
    selected_modules = normalize_feature_modules(feature_modules)

    dict_csv_path = dict_csv_path or DEFAULT_DICT_CSV_PATH
    fixed_feature_cols, default_map = load_fixed_feature_config(dict_csv_path)
    fixed_feature_cols, default_map = filter_feature_config(
        fixed_feature_cols,
        default_map,
        selected_modules,
    )

    txn_raw = raw_txn_from_slice(
        input_vars.get("illion_raw_transactions", []),
        user_id,
        send_time,
    )
    day_balance = balance_from_slice(
        input_vars.get("illion_day_end_balances", []),
        user_id,
        send_time,
    )
    txn_raw = enrich_transactions_with_account_type(
        txn_raw,
        input_vars.get("bank_accounts", []),
    )
    txn_raw, day_balance = filter_future_records(
        txn_raw,
        day_balance,
        send_time,
    )

    if txn_raw.shape[0] == 0 and day_balance.shape[0] == 0:
        empty_row = {
            "user_id": user_id,
            "application_id": application_id,
            "sample_datetime": send_time,
        }
        for col in fixed_feature_cols:
            empty_row[col] = default_map.get(col)
        empty_row.update(
            {
                "txn_raw_input_cnt": 0,
                "txn_balance_input_cnt": 0,
                "send_time": send_time,
                "transaction_date_max": None,
                "balance_date_max": None,
            }
        )
        return empty_row

    df_master = data_prepare(
        user_id=user_id,
        raw_data=txn_raw,
        eod_date=day_balance,
        send_time=send_time,
        applicationid=application_id,
        selected_modules=selected_modules,
    )
    feature_row = df_master.to_dict(orient="records")[0]

    aligned_row = {
        "user_id": user_id,
        "application_id": application_id,
        "sample_datetime": send_time,
    }
    for col in fixed_feature_cols:
        aligned_row[col] = feature_row.get(col, default_map.get(col))

    aligned_row.update(
        {
            "txn_raw_input_cnt": txn_raw.shape[0],
            "txn_balance_input_cnt": day_balance.shape[0],
            "send_time": send_time,
            "transaction_date_max": (
                None
                if txn_raw.shape[0] == 0
                else txn_raw["transaction_date"].dt.strftime("%Y-%m-%d").max()
            ),
            "balance_date_max": (
                None
                if day_balance.shape[0] == 0
                else day_balance["balance_date"].dt.strftime("%Y-%m-%d").max()
            ),
        }
    )
    return aligned_row
