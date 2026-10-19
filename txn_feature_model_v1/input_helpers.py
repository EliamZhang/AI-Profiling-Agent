# coding: utf-8
import pandas as pd


RAW_TXN_COLUMNS = [
    "user_id",
    "sample_datetime",
    "transaction_id",
    "bank_account_id",
    "illion_trx_uuid",
    "transaction_date",
    "text",
    "amount",
    "balance",
    "dr_cr",
    "trx_type",
    "category",
    "third_party",
]

BALANCE_COLUMNS = [
    "user_id",
    "sample_datetime",
    "balance_id",
    "bank_account_id",
    "balance_date",
    "balance",
]


def raw_txn_from_slice(data, user_id, send_time):
    if len(data) == 0:
        return pd.DataFrame(columns=RAW_TXN_COLUMNS)

    df = pd.DataFrame(data)
    df["user_id"] = user_id
    df["sample_datetime"] = pd.to_datetime(send_time)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["amount"] = df["amount"].astype("float")
    df["balance"] = df["balance"].astype("float")
    return df


def balance_from_slice(data, user_id, send_time):
    if len(data) == 0:
        return pd.DataFrame(columns=BALANCE_COLUMNS)

    df = pd.DataFrame(data)
    df["user_id"] = user_id
    df["sample_datetime"] = pd.to_datetime(send_time)
    df["balance_date"] = pd.to_datetime(df["balance_date"])
    df["balance"] = df["balance"].astype("float")
    return df


def enrich_transactions_with_account_type(txn_raw, bank_accounts):
    if txn_raw.shape[0] == 0:
        return txn_raw

    bank_accounts_df = pd.DataFrame(bank_accounts)
    if bank_accounts_df.shape[0] > 0 and "bank_account_id" in bank_accounts_df.columns:
        txn_raw = txn_raw.copy()
        txn_raw["bank_account_id"] = txn_raw["bank_account_id"].astype(str)
        bank_accounts_df["bank_account_id"] = bank_accounts_df["bank_account_id"].astype(str)
        if "account_type" not in bank_accounts_df.columns:
            bank_accounts_df["account_type"] = ""
        return txn_raw.merge(
            bank_accounts_df[["bank_account_id", "account_type"]].drop_duplicates("bank_account_id"),
            on="bank_account_id",
            how="left",
        )

    txn_raw = txn_raw.copy()
    txn_raw["account_type"] = ""
    return txn_raw


def filter_future_records(txn_raw, day_balance, send_time):
    sample_day = pd.to_datetime(send_time).floor("D")

    if txn_raw is not None and txn_raw.shape[0] > 0:
        txn_raw = txn_raw.copy()
        txn_day = pd.to_datetime(txn_raw["transaction_date"], errors="coerce").dt.floor("D")
        txn_raw = txn_raw.loc[txn_day.notna() & (txn_day <= sample_day)].copy()

    if day_balance is not None and day_balance.shape[0] > 0:
        day_balance = day_balance.copy()
        balance_day = pd.to_datetime(day_balance["balance_date"], errors="coerce").dt.floor("D")
        day_balance = day_balance.loc[
            balance_day.notna() & (balance_day <= sample_day)
        ].copy()

    return txn_raw, day_balance
