# -*- coding: utf-8 -*-
import os
import sys

import joblib
import numpy as np
import pandas as pd

from . import txn_tool
from .input_helpers import (
    balance_from_slice,
    enrich_transactions_with_account_type,
    filter_future_records,
    raw_txn_from_slice,
)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "git_pickle")

NUMERIC_FILL = -999999.99999
LENDER_FILL = -1

MODEL_VARS_BALANCE = {
    "bank_txn_balance_std_168d": [NUMERIC_FILL],
    "bank_txn_balance_kurtosis_182d": [NUMERIC_FILL],
    "bank_txn_balance_debit_gap_max_182d": [NUMERIC_FILL],
    "bank_txn_balance_skewness_182d": [NUMERIC_FILL],
}

MODEL_VARS_CATEGORY = {
    "bank_txn_category_cluster_debt_share_182d": [NUMERIC_FILL],
    "bank_txn_category_dishonours_credit_cnt_56d": [NUMERIC_FILL],
    "bank_txn_category_internal_transfer_credit_amt_182d": [NUMERIC_FILL],
    "bank_txn_category_non_sacc_loans_debit_cnt_28d": [NUMERIC_FILL],
    "bank_txn_category_external_transfers_share_28d": [NUMERIC_FILL],
}

MODEL_VARS_LENDER = {
    "bank_txn_lender_disburse_advance_amount_min": [LENDER_FILL],
    "bank_txn_lender_repay_jgs": [LENDER_FILL],
    "bank_txn_lender_repay_bnpl_amount_min_l28d": [LENDER_FILL],
    "bank_txn_lender_disburse_competitor_MyPayNow_count_l84d": [LENDER_FILL],
}

MODEL_VARS_INCOME = {
    "bank_txn_income_wages_cv_56d": [NUMERIC_FILL],
    "bank_txn_income_wages_mean_consumption_rate_14d": [NUMERIC_FILL],
}

MODEL_VARS_EXPENSE = {
    "bank_txn_expense_insurance_ratio_14d": [NUMERIC_FILL],
    "bank_txn_expense_internaltransfer_daily_max_14d": [NUMERIC_FILL],
    "bank_txn_expense_internaltransfer_max_consecutive_days": [NUMERIC_FILL],
}

MODEL_FILES = {
    "bank_model": "aus_old_risk_bid_submodel_20260323_v1_2_txn_lgb.pkl",
}

MODEL_SCORE_NAME = "aus_old_risk_bid_submodel_20260323_v1_2_txn_lgb_score"
MODEL_FEATURES_NAME = f"{MODEL_SCORE_NAME}_features"

MODEL_INPUT_VARS = [
    "bank_txn_balance_std_168d",
    "bank_txn_balance_kurtosis_182d",
    "bank_txn_balance_debit_gap_max_182d",
    "bank_txn_balance_skewness_182d",
    "bank_txn_category_cluster_debt_share_182d",
    "bank_txn_category_dishonours_credit_cnt_56d",
    "bank_txn_category_internal_transfer_credit_amt_182d",
    "bank_txn_category_non_sacc_loans_debit_cnt_28d",
    "bank_txn_category_external_transfers_share_28d",
    "bank_txn_lender_disburse_advance_amount_min",
    "bank_txn_lender_repay_jgs",
    "bank_txn_lender_repay_bnpl_amount_min_l28d",
    "bank_txn_lender_disburse_competitor_MyPayNow_count_l84d",
    "bank_txn_expense_insurance_ratio_14d",
    "bank_txn_expense_internaltransfer_daily_max_14d",
    "bank_txn_expense_internaltransfer_max_consecutive_days",
    "bank_txn_income_wages_cv_56d",
    "bank_txn_income_wages_mean_consumption_rate_14d",
]


def build_default_df(var_dict, user_id, send_time):
    row = {
        "user_id": user_id,
        "sample_datetime": pd.to_datetime(send_time),
    }
    for key, value in var_dict.items():
        row[key] = value[0] if isinstance(value, list) else value
    return pd.DataFrame([row])


def fillna_clip_numeric(df, fill_value):
    if df is None or df.shape[0] == 0:
        return df

    df = df.copy()
    df = df.replace(["", " ", "None", "nan"], np.nan)
    cols = df.columns.difference(["user_id", "sample_datetime"])
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[cols] = df[cols].fillna(fill_value).clip(lower=fill_value)
    return df


def data_prepare(user_id, raw_data, eod_date, send_time, applicationid):
    df_user_info = pd.DataFrame(
        [
            {
                "user_id": user_id,
                "application_id": applicationid,
                "sample_datetime": pd.to_datetime(send_time),
            }
        ]
    )

    if raw_data is None or raw_data.shape[0] == 0:
        df_txn_cate = build_default_df(MODEL_VARS_CATEGORY, user_id, send_time)
        df_txn_income = build_default_df(MODEL_VARS_INCOME, user_id, send_time)
        df_txn_expense = build_default_df(MODEL_VARS_EXPENSE, user_id, send_time)
        df_txn_lender = build_default_df(MODEL_VARS_LENDER, user_id, send_time)
    else:
        df_txn_cate = txn_tool.txn_cate.compute_bank_features(
            df=raw_data,
            categories=["cluster_share", "category_share"],
        )
        df_txn_cate["user_id"] = user_id
        df_txn_cate["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_cate = fillna_clip_numeric(df_txn_cate, NUMERIC_FILL)

        df_txn_income = txn_tool.txn_income_v1_1.generate_income_feature(
            df=raw_data,
            feature_groups=["amount_by_type", "consumption_rate"],
        )
        df_txn_income["user_id"] = user_id
        df_txn_income["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_income = fillna_clip_numeric(df_txn_income, NUMERIC_FILL)

        df_txn_expense = txn_tool.txn_expense.generate_expense_feature(
            df=raw_data,
            feature_groups=["ratio", "amount_daily", "max_consecutive_days"],
        )
        df_txn_expense["user_id"] = user_id
        df_txn_expense["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_expense = fillna_clip_numeric(df_txn_expense, NUMERIC_FILL)

        lender_vars = txn_tool.txn_lender.TxnListSampleVar(
            user_id=user_id,
            send_time=send_time,
            raw_data=raw_data,
        )
        lender_vars.clean_txn_step1()
        lender_vars.create_lender_cate_list(
            txn_tool.load_refdata.load_third_category("third_party_cate")
        )
        lender_vars.grouped_app_cate_list()
        lender_vars.calc_vars_disburse_stat(
            cate_list=["cash/wage advance"],
            day_list=[None],
        )
        lender_vars.calc_vars_summary_repay_jgs_stat(day_list=[None])
        lender_vars.calc_vars_repay_stat(cate_list=["BNPL"], day_list=[28])
        lender_vars.calc_vars_disburse_competitor_stat(
            competitor_list=["MyPayNow"],
            day_list=[84],
        )

        df_txn_lender = pd.json_normalize(lender_vars.variables)
        df_txn_lender["user_id"] = user_id
        df_txn_lender["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_lender = fillna_clip_numeric(df_txn_lender, LENDER_FILL)

    if eod_date is None or eod_date.shape[0] == 0:
        df_txn_eod = build_default_df(MODEL_VARS_BALANCE, user_id, send_time)
    else:
        df_txn_eod = txn_tool.txn_eod.compute_balance_timeseries_features(
            df_input=eod_date,
            categories=["basic", "timeseries", "gaps"],
        )
        df_txn_eod["user_id"] = user_id
        df_txn_eod["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_eod = fillna_clip_numeric(df_txn_eod, NUMERIC_FILL)

    df_master = df_user_info
    for df in [df_txn_cate, df_txn_eod, df_txn_income, df_txn_expense, df_txn_lender]:
        if df is not None and df.shape[0] > 0:
            df_master = df_master.merge(
                df,
                on=["user_id", "sample_datetime"],
                how="left",
            )

    for col in df_master.columns:
        if df_master[col].dtype == "float64":
            df_master[col] = df_master[col].round(5)

    return df_master


def load_model(model_files):
    clfs = {}
    for model_name, model_file in model_files.items():
        model_path = os.path.join(PKL_DIR, model_file)
        try:
            clfs[model_name] = joblib.load(model_path)
        except ModuleNotFoundError as exc:
            if exc.name == "lightgbm":
                raise RuntimeError(
                    "Model scoring requires 'lightgbm' to load the packaged pickle. "
                    "This project README expects Python 3.7 with lightgbm==2.3.1, "
                    f"but the current interpreter is Python {sys.version_info.major}.{sys.version_info.minor}. "
                    f"Missing dependency while loading: {model_path}"
                ) from exc
            raise
    return clfs


def model_scoring(df, clfs):
    result = {}
    for model_name in MODEL_FILES.keys():
        result[MODEL_SCORE_NAME] = clfs[model_name].predict(
            df[clfs[model_name].feature_name()].values,
            num_iteration=clfs[model_name].best_iteration,
        )[0]
    return result


def generate_score(input_vars):
    user_id = input_vars["userId"]
    application_id = input_vars["applicationId"]
    send_time = str(pd.to_datetime(input_vars["flowTime"][:19]))

    txn_raw = raw_txn_from_slice(
        input_vars["illion_raw_transactions"],
        user_id,
        send_time,
    )
    day_balance = balance_from_slice(
        input_vars["illion_day_end_balances"],
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
        return {
            MODEL_SCORE_NAME: None,
            MODEL_FEATURES_NAME: None,
        }

    df_master = data_prepare(
        user_id=user_id,
        raw_data=txn_raw,
        eod_date=day_balance,
        send_time=send_time,
        applicationid=application_id,
    )[MODEL_INPUT_VARS]

    clfs = load_model(model_files=MODEL_FILES)
    result = model_scoring(df=df_master, clfs=clfs)

    feature_dict = df_master.loc[:, MODEL_INPUT_VARS].to_dict(orient="records")[0]
    feature_dict["txn_raw_input_cnt"] = txn_raw.shape[0]
    feature_dict["txn_balance_input_cnt"] = day_balance.shape[0]
    feature_dict["send_time"] = send_time
    feature_dict["transaction_date_max"] = (
        None
        if txn_raw.shape[0] == 0
        else txn_raw["transaction_date"].dt.strftime("%Y-%m-%d").max()
    )
    feature_dict["balance_date_max"] = (
        None
        if day_balance.shape[0] == 0
        else day_balance["balance_date"].dt.strftime("%Y-%m-%d").max()
    )

    result[MODEL_FEATURES_NAME] = feature_dict
    return result
