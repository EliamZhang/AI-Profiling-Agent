# -*- coding: utf-8 -*-
"""Lender-attributed dishonour (bounced direct debit) detection.

Classifies each dishonour event as *resolved* (a same-lender direct-debit
retry for the modal repayment amount, or an integer multiple of it, landed
within a few days afterward) or *unresolved* (no such recovery found). A
bounce that gets retried and paid within days is a materially different
signal from one that never gets made good, and treating them the same (as a
flat dishonour count would) understates how noisy that signal is.

Deliberately does NOT reuse txn_lender.TxnListSampleVar's clean_txn_step1(),
which pre-filters transactions to rows where `category` contains "loan" --
dishonour events are typically categorized "Dishonours" (a distinct category,
see txn_cate.CATEGORIES), so that filter would silently discard every
dishonour row before this module ever saw them. This module runs its own pass
over the full, unfiltered transaction set instead, reusing only the
third_party -> competitor/product_type reference mapping and the merge
pattern `create_lender_cate_list` already uses.
"""
import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (7, 14, 28, 56, 84, 168, 182)
DEFAULT_RECOVERY_WINDOW_DAYS = 3
DEFAULT_AMOUNT_TOL_ABS_CAP = 1.0
DEFAULT_AMOUNT_TOL_PCT = 0.01
# How many extra instalment-multiples beyond the bounce's own implied k to
# search, to allow for the customer falling further behind between the
# bounce and the eventual retry.
MAX_CATCHUP_LOOKAHEAD = 5

REQUIRED_COLUMNS = ("category", "text", "trx_type", "third_party", "dr_cr", "amount", "transaction_date")

# lender_cate_mapping_df's `product_type` vocabulary -> the slug used in
# output column names. Anything not listed here (mapping-table drift) falls
# back to "other" rather than raising, since this must never crash a live
# feature-production run -- but "other" showing up is worth investigating.
PRODUCT_TYPE_TO_SLUG = {
    "personal loan": "personal_loan",
    "BNPL": "bnpl",
    "cash/wage advance": "cash_wage_advance",
    "bank": "bank",
    "debt collection": "debt_collection",
}
LENDER_TYPE_SLUGS = tuple(PRODUCT_TYPE_TO_SLUG.values()) + ("other",)

# Real debt-collection activity from a lender that isn't in the mapping table
# at all (e.g. Panthera Finance, Pioneer Credit, ARL Collect, Baycorp, Prushka
# -- known AU debt-recovery agencies, not present in either source list this
# table was merged from, see the mapping-table comment below) would otherwise
# be silently dropped as an unattributable orphan (_resolve_orphan_lenders
# only tries an amount/window match, and debt-recovery activity rarely has
# a matching prior "loan" repayment to match against anyway). Caught instead
# via the transaction's own text -- broader coverage than a named mapping-
# table row, at the cost of not knowing which specific collector it is (every
# such row shares one synthetic identity, since there's no name to group by).
_UNMAPPED_DEBT_COLLECTION_TEXT_KEYWORDS = ("RECOVERY", "COLLECTION", "COLLECTIONS", "ARREARS")
_UNKNOWN_DEBT_COLLECTOR_COMPETITOR = "Unknown Debt Collector"

_METRICS_PER_LENDER_TYPE = (
    "resolved_cnt", "unresolved_cnt",
    "resolved_amt", "unresolved_amt",
    "unresolved_institution_cnt", "resolved_catchup_cnt",
    "ratio", "resolved_ratio", "unresolved_ratio",
)

# `ratio` (total bounce rate, resolved+unresolved combined) is computed before
# the resolved/unresolved split is ever applied, so its value is genuinely
# invariant to `recovery_window_days` -- unlike every other metric here,
# including the two *_ratio siblings added alongside it (2026-07-24, Henry:
# "you can have resolved and unresolved, and total ratio, but you need to
# calibrate the calculation logic, and the naming" -- the earlier version
# baked the `{g}gp` grace-period token into every metric's name uniformly,
# including `ratio`, which was misleading since that one metric's value never
# actually changes with grace). Metrics in this set get NO `{g}gp` token;
# every other metric does.
_GP_INDEPENDENT_METRICS = ("ratio",)


def _lender_type_slug(product_type_series):
    return product_type_series.map(PRODUCT_TYPE_TO_SLUG).fillna("other")


def _metric_col_name(prefix, metric, g, window=None):
    """`prefix` like `bank_txn_dishonour_{slug}` or `bank_txn_dishonour_lender`.
    Appends `_{g}gp` unless `metric` is grace-independent (see
    _GP_INDEPENDENT_METRICS), then the window suffix if given."""
    base = f"{prefix}_{metric}" if metric in _GP_INDEPENDENT_METRICS else f"{prefix}_{metric}_{g}gp"
    return f"{base}_{window}d" if window is not None else base


def feature_names(windows, recovery_window_days=DEFAULT_RECOVERY_WINDOW_DAYS):
    """Four blocks, in order: (1) the per-lender_type, per-window grid: `bank_txn_
    dishonour_{slug}_{metric}_{g}gp_{w}d`; (2) the same metrics with the lender_type
    dimension collapsed (`..._lender_{metric}_{g}gp_{w}d`) -- all sub-categories
    combined, still bounded to a window; (3) the same metrics with the window
    dimension collapsed instead (`..._{slug}_{metric}_{g}gp`, no window suffix) --
    all-time totals per lender_type; (4) two grace-independent recency columns per
    lender_type (`..._{slug}_days_since_closest`/`..._days_since_furthest`), a day
    count relative to send_time, not a calendar date. The `{g}gp` token bakes in
    whatever `recovery_window_days` was actually used to classify resolved/
    unresolved -- every metric gets it EXCEPT `ratio` (see
    _GP_INDEPENDENT_METRICS), since that's the one metric whose value doesn't
    actually depend on grace; `resolved_ratio`/`unresolved_ratio` do and keep the
    token."""
    names = []
    g = recovery_window_days
    for w in windows:
        for slug in LENDER_TYPE_SLUGS:
            for metric in _METRICS_PER_LENDER_TYPE:
                names.append(_metric_col_name(f"bank_txn_dishonour_{slug}", metric, g, window=w))
        for metric in _METRICS_PER_LENDER_TYPE:
            names.append(_metric_col_name("bank_txn_dishonour_lender", metric, g, window=w))
    for slug in LENDER_TYPE_SLUGS:
        for metric in _METRICS_PER_LENDER_TYPE:
            names.append(_metric_col_name(f"bank_txn_dishonour_{slug}", metric, g))
        names.append(f"bank_txn_dishonour_{slug}_days_since_closest")
        names.append(f"bank_txn_dishonour_{slug}_days_since_furthest")
    return names


def _empty_feature_row(windows, recovery_window_days=DEFAULT_RECOVERY_WINDOW_DAYS):
    """No dishonour activity is a confirmed-clean fact from the data, not missing
    data -- every count/amount/ratio column defaults to 0, not a sentinel. The
    recency columns are the one exception: "days since a dishonour that never
    happened" is undefined, not zero (0 would read as "dishonoured today"), so those
    default to NaN -- same distinction txn_worthiness.py's _days_since_last makes."""
    row = {name: 0 for name in feature_names(windows, recovery_window_days)}
    for slug in LENDER_TYPE_SLUGS:
        row[f"bank_txn_dishonour_{slug}_days_since_closest"] = np.nan
        row[f"bank_txn_dishonour_{slug}_days_since_furthest"] = np.nan
    return pd.DataFrame([row])


def _prep(df):
    """Own copy; guarantee required columns exist; coerce dtypes. Never
    mutates the caller's DataFrame."""
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    for col in ("category", "text", "trx_type", "third_party", "dr_cr"):
        out[col] = out[col].fillna("").astype(str)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").abs()
    out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")
    return out


def _flag_dishonours(prepped):
    """category == 'Dishonours' OR text/trx_type carries a dishonour/reversal
    keyword -- mirrors liability_detection's detection logic, adapted to this
    codebase's real column names."""
    text_upper = prepped["text"].str.upper()
    trx_type_upper = prepped["trx_type"].str.upper()
    return (
        (prepped["category"] == "Dishonours")
        | text_upper.str.contains("DISHONOUR", na=False)
        | text_upper.str.contains("DISHON", na=False)
        | trx_type_upper.str.contains("REVERSAL", na=False)
        | trx_type_upper.str.contains("RETURN", na=False)
    )


def _attribute_direct(prepped, lender_cate_mapping_df):
    """Merge on third_party -> competitor/product_type, same join
    `create_lender_cate_list` uses. Rows whose third_party isn't in the
    mapping (or is blank) come back with competitor/product_type == NaN."""
    mapping = (
        lender_cate_mapping_df[["third_party", "competitor", "product_type"]]
        .drop_duplicates("third_party")
    )
    return prepped.merge(mapping, on="third_party", how="left")


def _detect_unmapped_debt_collection(attributed):
    """For rows the mapping table couldn't attribute at all (competitor is
    still NaN after the merge above), scan text for debt-collection language
    and, if found, give them a synthetic debt_collection identity instead of
    leaving them to fall through as an unattributable orphan (see
    _UNMAPPED_DEBT_COLLECTION_TEXT_KEYWORDS above). Only ever fills in a
    missing attribution -- never overrides a real mapping-table match."""
    unmapped = attributed["competitor"].isna()
    if not unmapped.any():
        return attributed

    text_upper = attributed["text"].str.upper()
    has_keyword = pd.Series(False, index=attributed.index)
    for kw in _UNMAPPED_DEBT_COLLECTION_TEXT_KEYWORDS:
        has_keyword = has_keyword | text_upper.str.contains(kw, na=False)

    hit = unmapped & has_keyword
    if not hit.any():
        return attributed

    out = attributed.copy()
    out.loc[hit, "competitor"] = _UNKNOWN_DEBT_COLLECTOR_COMPETITOR
    out.loc[hit, "product_type"] = "debt collection"
    return out


def _loan_debit_pool(attributed):
    """Successful (non-dishonour) debit repayments on loan-category rows,
    lender-resolved -- the candidate pool for both modal-amount computation
    and recovery matching."""
    is_loan_cat = attributed["category"].str.lower().str.contains("loan", na=False)
    is_debit = attributed["dr_cr"].str.lower() == "debit"
    pool = attributed[
        is_loan_cat
        & is_debit
        & (~attributed["is_dishonour"])
        & attributed["competitor"].notna()
    ]
    return pool.copy()


def _modal_repayment_amounts(loan_debit_pool):
    """Per-lender mode of repayment amount -- the recurring instalment size.
    Nothing else in txn_tool computes this today."""
    modal = {}
    if loan_debit_pool.empty:
        return modal
    for lender, grp in loan_debit_pool.groupby("competitor"):
        amounts = grp["amount"].dropna().round(2)
        if amounts.empty:
            continue
        mode_vals = amounts.mode()
        if not mode_vals.empty:
            modal[lender] = float(mode_vals.iloc[0])
    return modal


def _resolve_orphan_lenders(dishonour_rows, loan_debit_pool, recovery_window_days):
    """Attribute dishonours with no direct competitor match (missing or
    unmapped third_party) to a lender via exact-amount + 0..recovery_window
    -day-earlier proximity against the loan-debit pool -- same idiom as
    liability_detection's orphan-dishonour pass. Dishonours that still can't
    be attributed (e.g. a bill/rent/utility dishonour) are dropped -- this
    module only reports on lender dishonours."""
    resolved = dishonour_rows[dishonour_rows["competitor"].notna()].copy()
    orphans = dishonour_rows[dishonour_rows["competitor"].isna()].copy()
    if orphans.empty or loan_debit_pool.empty:
        return resolved

    newly_resolved_rows = []
    for _, dish in orphans.iterrows():
        if pd.isna(dish["amount"]) or pd.isna(dish["transaction_date"]):
            continue
        same_amt = loan_debit_pool[(loan_debit_pool["amount"] - dish["amount"]).abs() < 0.01]
        if same_amt.empty:
            continue
        gap_days = (dish["transaction_date"] - same_amt["transaction_date"]).dt.days
        near = same_amt[(gap_days >= 0) & (gap_days <= recovery_window_days)]
        if near.empty:
            continue
        best = near.loc[(dish["transaction_date"] - near["transaction_date"]).abs().idxmin()]
        dish = dish.copy()
        dish["competitor"] = best["competitor"]
        dish["product_type"] = best["product_type"]
        newly_resolved_rows.append(dish)

    if newly_resolved_rows:
        resolved = pd.concat([resolved, pd.DataFrame(newly_resolved_rows)], ignore_index=True)
    return resolved


def _classify_resolution(
    dishonours, loan_debit_pool, recovery_window_days, amount_tol_abs_cap, amount_tol_pct
):
    """For each lender-attributed dishonour, search forward for a same-lender
    recovery for the modal repayment amount (or an integer multiple of it,
    for multi-period catch-up), within a short window. Adds `resolved`
    (bool) and `catchup` (bool, only meaningful when resolved) columns.

    Direct-debit draws are all-or-nothing -- a retry always pulls the full
    currently-owed amount, an exact multiple of the modal instalment, never a
    partial/fee-adjusted figure -- so the match target is `modal * k` for an
    integer k, with only a small tolerance for rounding noise, not a loose
    percentage band.
    """
    modal_amounts = _modal_repayment_amounts(loan_debit_pool)
    available = loan_debit_pool.copy()
    available["_consumed"] = False

    dishonours = dishonours.sort_values("transaction_date").reset_index(drop=True)
    resolved_flags = []
    catchup_flags = []

    for _, dish in dishonours.iterrows():
        lender = dish["competitor"]
        d_date = dish["transaction_date"]
        d_amt = dish["amount"]

        if pd.isna(lender) or pd.isna(d_date) or pd.isna(d_amt):
            resolved_flags.append(False)
            catchup_flags.append(False)
            continue

        modal = modal_amounts.get(lender)
        if modal and modal > 0:
            k_bounce = max(1, round(d_amt / modal))
        else:
            k_bounce = 1

        window_start = d_date
        window_end = d_date + pd.Timedelta(days=recovery_window_days)
        candidates = available[
            (~available["_consumed"])
            & (available["competitor"] == lender)
            & (available["transaction_date"] > window_start)
            & (available["transaction_date"] <= window_end)
        ]

        match_idx = None
        match_k = None
        best_gap = None
        if not candidates.empty:
            for k in range(k_bounce, k_bounce + MAX_CATCHUP_LOOKAHEAD + 1):
                if modal and modal > 0:
                    target = modal * k
                elif k == 1:
                    # No modal amount known yet (e.g. first-ever transaction
                    # with this lender) -- fall back to matching the bounce's
                    # own amount directly, k=1 only (no basis for a multiple).
                    target = d_amt
                else:
                    continue
                tol = min(amount_tol_abs_cap, amount_tol_pct * target)
                close = candidates[(candidates["amount"] - target).abs() <= tol]
                if close.empty:
                    continue
                gaps = close["transaction_date"] - d_date
                local_best_idx = gaps.idxmin()
                local_gap = gaps.loc[local_best_idx]
                if best_gap is None or local_gap < best_gap:
                    best_gap = local_gap
                    match_idx = local_best_idx
                    match_k = k

        if match_idx is not None:
            available.loc[match_idx, "_consumed"] = True
            resolved_flags.append(True)
            catchup_flags.append(bool(match_k and match_k > 1))
        else:
            resolved_flags.append(False)
            catchup_flags.append(False)

    dishonours = dishonours.copy()
    dishonours["resolved"] = resolved_flags
    dishonours["catchup"] = catchup_flags
    return dishonours


def _metrics_for(dish_subset, pool_subset):
    """The 9-metric dict for one (dishonour rows, repayment-pool rows) slice --
    shared by every cut _aggregate produces (per-slug-per-window, lender-collapsed-
    per-window, per-slug-all-time), so the definitions can't drift between cuts.

    Three ratio variants, same denominator throughout (dishonour_cnt + repay_cnt,
    the same successful-repayment universe used everywhere else here):
    `ratio` = total bounce rate (resolved+unresolved together, grace-independent
    -- see _GP_INDEPENDENT_METRICS), `resolved_ratio` + `unresolved_ratio` = the
    same total split by resolution status, so resolved_ratio + unresolved_ratio
    == ratio exactly for every slice."""
    unresolved = dish_subset[~dish_subset["resolved"]]
    resolved = dish_subset[dish_subset["resolved"]]
    dishonour_cnt = int(len(dish_subset))
    repay_cnt = int(len(pool_subset))
    denom = dishonour_cnt + repay_cnt
    return {
        "unresolved_cnt": int(len(unresolved)),
        "resolved_cnt": int(len(resolved)),
        "unresolved_amt": float(unresolved["amount"].sum()),
        "resolved_amt": float(resolved["amount"].sum()),
        "unresolved_institution_cnt": int(unresolved["competitor"].nunique()),
        "resolved_catchup_cnt": int(resolved["catchup"].sum()),
        "resolved_ratio": len(resolved) / denom if denom > 0 else 0.0,
        "unresolved_ratio": len(unresolved) / denom if denom > 0 else 0.0,
        "ratio": dishonour_cnt / denom if denom > 0 else 0.0,
    }


def _aggregate(classified, loan_debit_pool, windows, recovery_window_days):
    """Roll up per lender_type (personal_loan/bnpl/cash_wage_advance/bank/
    debt_collection/other), per window, using _metrics_for -- resolved/unresolved
    cnt (the subcut Henry asked to keep) and amt, unresolved_institution_cnt,
    resolved_catchup_cnt, and a bounce-rate ratio = dishonour_cnt /
    (dishonour_cnt + successful_repayment_cnt) for that lender_type -- the
    successful-repayment universe comes from loan_debit_pool, the same pool
    used upstream for modal-amount/recovery matching.

    Also produces two additional cuts of the same metrics: a lender_type-
    collapsed cut (all sub-categories combined, still bounded to each window)
    and a window-collapsed cut (all-time totals per lender_type), plus two
    grace-independent recency columns per lender_type (days from the closest/
    furthest dishonour to send_time).
    """
    if classified.empty:
        return _empty_feature_row(windows, recovery_window_days)

    g = recovery_window_days
    classified = classified.copy()
    classified["_lender_type"] = _lender_type_slug(classified["product_type"])
    pool = loan_debit_pool.copy()
    pool["_lender_type"] = _lender_type_slug(pool["product_type"])

    row = {}
    for w in windows:
        in_window = classified[classified["trac_diffdays"] <= w]
        pool_in_window = pool[pool["trac_diffdays"] <= w]
        for slug in LENDER_TYPE_SLUGS:
            dish = in_window[in_window["_lender_type"] == slug]
            pool_slug = pool_in_window[pool_in_window["_lender_type"] == slug]
            for metric, val in _metrics_for(dish, pool_slug).items():
                row[_metric_col_name(f"bank_txn_dishonour_{slug}", metric, g, window=w)] = val
        # all sub-categories combined, still bounded to this window
        for metric, val in _metrics_for(in_window, pool_in_window).items():
            row[_metric_col_name("bank_txn_dishonour_lender", metric, g, window=w)] = val

    for slug in LENDER_TYPE_SLUGS:
        dish_all = classified[classified["_lender_type"] == slug]
        pool_all = pool[pool["_lender_type"] == slug]
        for metric, val in _metrics_for(dish_all, pool_all).items():
            row[_metric_col_name(f"bank_txn_dishonour_{slug}", metric, g)] = val
        if not dish_all.empty:
            row[f"bank_txn_dishonour_{slug}_days_since_closest"] = float(dish_all["trac_diffdays"].min())
            row[f"bank_txn_dishonour_{slug}_days_since_furthest"] = float(dish_all["trac_diffdays"].max())
        else:
            row[f"bank_txn_dishonour_{slug}_days_since_closest"] = np.nan
            row[f"bank_txn_dishonour_{slug}_days_since_furthest"] = np.nan
    return pd.DataFrame([row])


def compute_dishonour_features(
    df,
    lender_cate_mapping_df,
    send_time,
    windows=DEFAULT_WINDOWS,
    recovery_window_days=DEFAULT_RECOVERY_WINDOW_DAYS,
    amount_tol_abs_cap=DEFAULT_AMOUNT_TOL_ABS_CAP,
    amount_tol_pct=DEFAULT_AMOUNT_TOL_PCT,
):
    """Compute lender-attributed resolved/unresolved dishonour counts for one
    application's raw transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Full raw transactions, unfiltered by category (must NOT be
        pre-filtered to loan-category rows -- see module docstring).
    lender_cate_mapping_df : pd.DataFrame
        third_party -> competitor/product_type mapping, e.g.
        `load_refdata.load_third_category('third_party_cate')`.
    send_time : str or pd.Timestamp
        As-of time; dishonour recency windows are measured from here.

    Returns
    -------
    pd.DataFrame
        One row of `bank_txn_dishonour_{lender_type}_*` feature columns, one
        set per lender_type slug (personal_loan/bnpl/cash_wage_advance/bank/
        debt_collection/other) per window -- see `feature_names()`. Does not
        include `user_id`/`sample_datetime` -- the caller stamps those on,
        matching how the lender feature block in `data_prepare` works.
    """
    if df is None or df.shape[0] == 0:
        return _empty_feature_row(windows, recovery_window_days)

    prepped = _prep(df)
    prepped["is_dishonour"] = _flag_dishonours(prepped)

    send_time = pd.to_datetime(send_time)
    prepped["trac_diffdays"] = (send_time.normalize() - prepped["transaction_date"]).dt.days

    attributed = _attribute_direct(prepped, lender_cate_mapping_df)
    attributed = _detect_unmapped_debt_collection(attributed)
    loan_debit_pool = _loan_debit_pool(attributed)

    dishonour_rows = attributed[attributed["is_dishonour"]].copy()
    if dishonour_rows.empty:
        return _empty_feature_row(windows, recovery_window_days)

    lender_dishonours = _resolve_orphan_lenders(dishonour_rows, loan_debit_pool, recovery_window_days)
    if lender_dishonours.empty:
        return _empty_feature_row(windows, recovery_window_days)

    classified = _classify_resolution(
        lender_dishonours, loan_debit_pool, recovery_window_days,
        amount_tol_abs_cap, amount_tol_pct,
    )
    return _aggregate(classified, loan_debit_pool, windows, recovery_window_days)
