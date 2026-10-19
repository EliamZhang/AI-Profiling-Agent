import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# --- tqdm（可选）---
try:
    from tqdm import tqdm  # 终端/Notebook 都能显示
except Exception:
    def tqdm(it, *args, **kwargs):
        return it


# ===================== 配置与校验 =====================
# 这里允许 sample_datetime 或 flowTime 至少一个存在；若只有 flowTime 则会自动映射为 sample_datetime
# REQUIRED_COLS_BASE = [
#     "user_id", "job_id", "bank_account_id",
#     "balance", "balance_date",  # "sample_datetime" 会在预处理里从 flowTime 映射
# ]
REQUIRED_COLS_BASE = [
    "user_id", "bank_account_id",
    "balance", "balance_date",  # "sample_datetime" 会在预处理里从 flowTime 映射
]
OPTIONAL_COLS = ["application_id", "flowTime", "sample_datetime"]

WINDOWS = [7, 14, 28, 56, 84, 168, 182]

# 可选类别（留档）
VALID_CATEGORIES = {
    "basic","debit","credit","ratios",
    "timeseries","regression","ar_ewma","kmeans","pca","kgram",
    "direction","pct_change","autocorr","snapshot",
    "inequality","gaps","tails","global"
}

def _validate_columns(df: pd.DataFrame, need_cols):
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必需列: {missing}")

def _strip_utc_and_quotes(x: pd.Series) -> pd.Series:
    """去掉双引号、尾部的 ' UTC'、首尾空格"""
    s = x.astype("string")
    s = s.str.strip().str.strip('"').str.replace(" UTC", "", regex=False)
    return s

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    """
    兼容诸如:
      - 2023-07-14 00:00:00.000000 UTC
      - "2024-01-10 06:59:11.0"
      - 2024/01/10 06:59:11
    解析失败统一为 NaT，不抛异常
    """
    if s.dtype != "datetime64[ns]":
        s = _strip_utc_and_quotes(s)
    # 先用 pandas 解析；若仍 NaT，则尝试再替换掉可能的微秒格式
    out = pd.to_datetime(s, errors="coerce")
    return out

def _ensure_sample_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    若没有 sample_datetime 但有 flowTime，则创建 sample_datetime = parsed(flowTime)。
    两列都会保留（flowTime 不参与后续计算，只作为原样记录）。
    """
    if "sample_datetime" not in df.columns:
        if "flowTime" not in df.columns:
            raise KeyError("既没有 'sample_datetime' 也没有 'flowTime'，无法确定样本时间。")
        tmp = _safe_to_datetime(df["flowTime"])
        if tmp.isna().all():
            raise ValueError("flowTime 全部解析失败，请检查时间格式。")
        df = df.copy()
        df["sample_datetime"] = tmp
    else:
        # 若已有 sample_datetime，但也提供了 flowTime，二者以 sample_datetime 为准
        df = df.copy()
        df["sample_datetime"] = _safe_to_datetime(df["sample_datetime"])
    return df

def _streak_for(series: pd.Series, val: int) -> pd.Series:
    """计算某个值的连续天数，例如 val=1 连涨，val=-1 连跌"""
    m = series.eq(val)
    out = m.groupby((~m).cumsum()).cumsum()
    return out.where(m, 0)


def _last_valid(x: pd.Series):
    """返回最后一个非空值；若全为空则返回 NaN。"""
    s = pd.Series(x)
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan

def _build_balance_series_features(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    """
    对已经是“单条余额序列”的数据做派生。
    这里的 df 可以是：
    1) 单卡序列
    2) 多卡汇总后的用户级日余额序列

    只要每个 group_cols 内部是一条按日变化的余额序列，就能复用同一套逻辑。
    """
    df = df.copy()
    df = df.sort_values(group_cols + ["balance_date"], kind="stable").reset_index(drop=True)

    gb_bal = df.groupby(group_cols, dropna=False)["balance"]

    sample_d = pd.to_datetime(df["sample_datetime"], errors="coerce").dt.normalize()
    balance_d = pd.to_datetime(df["balance_date"], errors="coerce").dt.normalize()
    df["date_diff"] = (sample_d - balance_d).dt.days.clip(lower=0)

    df["balance_lag"] = gb_bal.shift(1)
    df["balance_change"] = df["balance"] - df["balance_lag"]

    bal = df["balance"]
    lag = df["balance_lag"]
    ok = bal.notna() & lag.notna()

    trend = pd.Series(pd.NA, index=df.index, dtype="Int8")
    trend.loc[ok & bal.eq(lag)] = 0
    trend.loc[ok & bal.gt(lag)] = 1
    trend.loc[ok & bal.lt(lag)] = -1
    df["trend"] = trend

    base = df["balance_lag"].notna() & df["balance_lag"].ne(0)
    df["increase_rate"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    df.loc[base & df["balance"].gt(df["balance_lag"]), "increase_rate"] = (
        (df["balance"] - df["balance_lag"]) / df["balance_lag"]
    )

    df["decrease_rate"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    df.loc[base & df["balance"].lt(df["balance_lag"]), "decrease_rate"] = (
        (df["balance_lag"] - df["balance"]) / df["balance_lag"]
    )

    cnt = df.groupby(group_cols, dropna=False).cumcount() + 1
    cs = gb_bal.cumsum()
    df["avg_balance"] = (cs / cnt).where(df["balance"].notna())

    std = gb_bal.expanding().std()
    df["stddev_balance"] = std.reset_index(level=group_cols, drop=True)

    df["balance_deviation"] = (df["balance"] - df["avg_balance"]).where(df["avg_balance"].notna())
    df["is_balance_anomaly"] = (
        df["stddev_balance"].notna()
        & df["stddev_balance"].gt(0)
        & df["balance_deviation"].abs().gt(3 * df["stddev_balance"])
    ).astype(int)

    df["is_fluctuation_anomaly"] = (
        df["balance_lag"].notna()
        & (df["balance"].sub(df["balance_lag"]).abs() > 0.2 * df["balance"].abs())
    ).astype(float).where(df["balance_lag"].notna())

    df["balance_unchanged"] = (
        df["balance_lag"].notna() & df["balance"].eq(df["balance_lag"])
    ).astype(float).where(df["balance_lag"].notna())

    # 对于余额日序列，收入/支出仍沿用“余额变动拆分”的原口径，
    # 但这里是在“用户汇总余额序列”上计算，避免多卡直接混行。
    df["income_amount"] = df["balance_change"].clip(lower=0)
    df["expense_amount"] = (-df["balance_change"].clip(upper=0))
    df["net_inflow"] = df["income_amount"] - df["expense_amount"]

    BIG_EXPENSE_THRESHOLD = 2000.0
    BIG_INCOME_THRESHOLD = 2000.0
    df["is_big_expense"] = (df["expense_amount"] > BIG_EXPENSE_THRESHOLD).astype(int)
    df["is_big_income"] = (df["income_amount"] > BIG_INCOME_THRESHOLD).astype(int)

    df["next_balance"] = gb_bal.shift(-1)
    df["drop_after_big_expense"] = np.where(
        df["is_big_expense"].eq(1),
        df["balance"] - df["next_balance"],
        np.nan,
    )

    mask_na = df["trend"].isna()
    df["trend"] = df["trend"].fillna(0)
    df["increasing_streak"] = df.groupby(group_cols, dropna=False)["trend"].transform(
        lambda s: _streak_for(s, 1)
    )
    df["decreasing_streak"] = df.groupby(group_cols, dropna=False)["trend"].transform(
        lambda s: _streak_for(s, -1)
    )
    df.loc[mask_na, ["increasing_streak", "decreasing_streak"]] = pd.NA
    df["increasing_streak"] = df["increasing_streak"].astype("Int64")
    df["decreasing_streak"] = df["decreasing_streak"].astype("Int64")

    need_cols = [
        "date_diff", "balance_lag", "balance_change", "trend",
        "increase_rate", "decrease_rate", "avg_balance", "stddev_balance",
        "balance_deviation", "is_balance_anomaly", "is_fluctuation_anomaly",
        "balance_unchanged", "income_amount", "expense_amount", "net_inflow",
        "next_balance", "drop_after_big_expense", "increasing_streak",
        "decreasing_streak", "sample_datetime"
    ]
    missing = [c for c in need_cols if c not in df.columns]
    assert not missing, f"缺失列: {missing}"

    return df

# ===================== 明细派生 =====================
def transform_s_l_a_d(s_l_a_d: pd.DataFrame) -> pd.DataFrame:
    """
    多卡版本：
    1) 先在单卡单日层面去重，避免同一卡同一天重复记录扰动序列；
    2) 再把同一 user_id + sample_datetime 下的多张卡按 balance_date 汇总成“用户级日余额”；
    3) 最后在这条用户级余额序列上计算全部派生字段。

    这样可以保证：
    - 最终输出的变量名、变量数量不变；
    - 不会把不同银行卡的序列直接拼接后做 lag / streak / autocorr，避免含义失真。
    """
    _validate_columns(s_l_a_d, REQUIRED_COLS_BASE)

    s_l_a_d = _ensure_sample_datetime_column(s_l_a_d)

    s_l_a_d = s_l_a_d.copy()
    s_l_a_d["sample_datetime"] = _safe_to_datetime(s_l_a_d["sample_datetime"])
    s_l_a_d["balance_date"] = _safe_to_datetime(s_l_a_d["balance_date"])
    s_l_a_d["balance"] = pd.to_numeric(s_l_a_d["balance"], errors="coerce")
    s_l_a_d["bank_account_id"] = s_l_a_d["bank_account_id"].astype(str)

    # 只保留时间可用的记录
    s_l_a_d = s_l_a_d.loc[
        s_l_a_d["sample_datetime"].notna() & s_l_a_d["balance_date"].notna()
    ].copy()

    # 先按“单卡-单日”去重：若同一卡同一天有多条余额，保留最后一个非空 balance
    acct_day_cols = ["user_id", "sample_datetime", "bank_account_id", "balance_date"]
    s_l_a_d = s_l_a_d.sort_values(acct_day_cols, kind="stable").reset_index(drop=True)
    account_daily = (
        s_l_a_d.groupby(acct_day_cols, dropna=False, as_index=False)
        .agg(balance=("balance", _last_valid))
    )

    # 再按“用户-样本时点-日期”聚合多卡余额，构造成用户级日余额序列
    user_day_cols = ["user_id", "sample_datetime", "balance_date"]
    user_daily = (
        account_daily.groupby(user_day_cols, dropna=False, as_index=False)
        .agg(
            balance=("balance", "sum"),
            active_account_cnt=("bank_account_id", "nunique"),
        )
    )

    # 在用户级余额序列上统一做 lag / streak / 波动等特征
    out = _build_balance_series_features(
        user_daily,
        group_cols=["user_id", "sample_datetime"],
    )

    return out

# ===================== 工具函数（聚合用） =====================
def _safe_std(x):
    x = pd.Series(x).astype(float)
    if x.size <= 1: return np.nan
    return x.std(ddof=1)

def _nanmean_or_nan(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else np.nan

def _nanstd_or_nan(x: np.ndarray, ddof: int = 1) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= ddof:
        return np.nan
    return float(arr.std(ddof=ddof))

def _nanmax_or_nan(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size > 0 else np.nan

def _nanmin_or_nan(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size > 0 else np.nan

def _pearson_preserve_nan(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return np.nan
    if _nanstd_or_nan(a, ddof=1) > 0 and _nanstd_or_nan(b, ddof=1) > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return np.nan

def _to_fortnightly(value, w):
    if pd.isna(value): return np.nan
    return float(value) * 14.0 / float(w)

def _hurst_exponent(series: pd.Series) -> float:
    x = pd.Series(series).astype(float).dropna().values
    n = len(x)
    if n < 16: return np.nan
    max_lag = min(20, n // 3)
    lags = np.arange(2, max_lag + 1, dtype=int)
    if lags.size < 3: return np.nan
    tau = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        s = _safe_std(diff)
        tau.append(s)
    tau = np.array(tau, dtype=float)
    mask = np.isfinite(tau) & (tau > 0)
    if mask.sum() < 3: return np.nan
    xx = np.log(lags[mask]); yy = np.log(tau[mask])
    try:
        H = np.polyfit(xx, yy, 1)[0]
    except Exception:
        H = np.nan
    return H

def _ols_linear(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 or np.all(~np.isfinite(x)) or np.all(~np.isfinite(y)):
        return (np.nan, np.nan, np.nan)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return (np.nan, np.nan, np.nan)
    X = np.column_stack([np.ones_like(x), x])
    XtX = X.T.dot(X)
    try:
        beta = np.linalg.pinv(XtX).dot(X.T).dot(y)
    except Exception:
        return (np.nan, np.nan, np.nan)
    a, b = beta[0], beta[1]
    yhat = X.dot(beta)
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.sum((y - yhat)**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (b, a, r2)

def _poly_design(x, degree):
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(np.power(x, d))
    return np.column_stack(cols)

def _ridge_regression(x, y, degree=1, lam=1.0):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < (degree + 1):
        return (np.nan, np.nan, np.nan)
    X = _poly_design(x, degree)
    XtX = X.T.dot(X)
    I = np.eye(XtX.shape[0])
    beta = np.linalg.pinv(XtX + lam * I).dot(X.T).dot(y)
    yhat = X.dot(beta)
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.sum((y - yhat)**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    intercept = float(beta[0])
    slope1 = float(beta[1]) if beta.shape[0] > 1 else np.nan
    return (slope1, intercept, r2)

def _theil_sen_slope(x, y, max_pairs=20000):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 2: return np.nan
    rng = np.random.RandomState(7)
    if n*(n-1)//2 > max_pairs:
        idx_i = rng.randint(0, n-1, size=max_pairs)
        idx_j = rng.randint(idx_i+1, n, size=max_pairs)
    else:
        idx_i, idx_j = np.triu_indices(n, 1)
    dx = x[idx_j] - x[idx_i]
    valid = np.isfinite(dx) & (dx != 0)
    slopes = (y[idx_j] - y[idx_i]) / dx
    slopes = slopes[valid & np.isfinite(slopes)]
    if slopes.size == 0:
        return np.nan
    return float(np.median(slopes))

def _ar1_coef(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3 or np.nanstd(y) == 0:
        return (np.nan, np.nan, np.nan)
    yt = y[1:]; yl = y[:-1]
    X = np.column_stack([np.ones_like(yl), yl])
    XtX = X.T.dot(X)
    try:
        beta = np.linalg.pinv(XtX).dot(X.T).dot(yt)
    except Exception:
        return (np.nan, np.nan, np.nan)
    c, phi = beta[0], beta[1]
    resid = yt - X.dot(beta)
    resid_std = float(np.sqrt(np.mean(resid**2))) if resid.size else np.nan
    return (float(phi), float(c), resid_std)

def _ewma_residual_stats(y, alpha=0.3):
    y = pd.Series(y).astype(float)
    y_valid = y.dropna()
    if y_valid.size < 3:
        return (np.nan, np.nan, np.nan)
    s = []
    m = None
    for val in y:
        if not np.isfinite(val):
            s.append(np.nan if m is None else m)
            continue
        if m is None:
            m = val
        else:
            m = alpha * val + (1 - alpha) * m
        s.append(m)
    s = np.asarray(s, dtype=float)
    resid = y.to_numpy(dtype=float) - s
    resid = resid[np.isfinite(resid)]
    if resid.size < 3:
        return (np.nan, np.nan, np.nan)
    mean_abs = float(np.mean(np.abs(resid)))
    std = float(np.std(resid, ddof=1)) if resid.size > 1 else np.nan
    if resid.size > 2 and np.nanstd(resid[:-1], ddof=1) > 0 and np.nanstd(resid[1:], ddof=1) > 0:
        ac1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    else:
        ac1 = np.nan
    return (mean_abs, std, ac1)

def _kmeans_1d(x, k=2, max_iter=30):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < k:
        return (np.array([np.nan]*k), np.array([0]*k), np.nan, np.nan)
    qs = np.linspace(0, 1, k+2)[1:-1]
    c = np.quantile(x, qs)
    for _ in range(max_iter):
        dist = np.abs(x[:, None] - c[None, :])
        lab = np.argmin(dist, axis=1)
        c_new = np.array([x[lab == j].mean() if np.any(lab==j) else c[j] for j in range(k)], dtype=float)
        if np.allclose(c_new, c, atol=1e-8, rtol=1e-6):
            c = c_new
            break
        c = c_new
    dist = np.abs(x[:, None] - c[None, :])
    lab = np.argmin(dist, axis=1)
    wcss = float(np.sum((x - c[lab])**2))
    sizes = np.array([(lab == j).sum() for j in range(k)], dtype=int)
    order = np.argsort(c)
    c_sorted = c[order]
    sizes_sorted = sizes[order]
    separation = np.min(np.diff(c_sorted)) if k > 1 else np.nan
    return (c_sorted, sizes_sorted, wcss, separation)

def _gini_nonneg(arr: pd.Series) -> float:
    x = pd.Series(arr).astype(float).dropna()
    x = x[x >= 0]
    n = x.size
    s = x.sum()
    if n < 2 or s <= 0:
        return np.nan
    xs = np.sort(x.values)
    i = np.arange(1, n + 1, dtype=float)
    return (2.0 * (i * xs).sum() / (n * s)) - (n + 1.0) / n

def _entropy_share(arr: pd.Series) -> float:
    x = pd.Series(arr).astype(float).dropna()
    x = x[x > 0]
    s = x.sum()
    if s <= 0: return np.nan
    p = x / s
    k = p.size
    if k <= 1: return 0.0
    h = -np.sum(p * np.log(p))
    return float(h / np.log(k))

def _topk_share(arr: pd.Series, k=1) -> float:
    x = pd.Series(arr).astype(float).dropna()
    x = x[x > 0]
    s = x.sum()
    if s <= 0: return np.nan
    topk = np.sort(x.values)[-k:].sum()
    return float(topk / s)

def _autocorr_lag(x: np.ndarray, lag: int) -> float:
    if x is None or len(x) <= lag or lag <= 0:
        return np.nan
    a = x[:-lag]
    b = x[lag:]
    if a.size < 2:
        return np.nan
    if np.nanstd(a, ddof=1) == 0 or np.nanstd(b, ddof=1) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def _gap_stats(active_mask: pd.Series, dd: np.ndarray):
    m = pd.Series(active_mask).fillna(False).to_numpy(dtype=bool)
    if m.sum() <= 1:
        return (np.nan, np.nan)
    pos = dd[m]
    gaps = np.diff(pos)
    if gaps.size == 0:
        return (np.nan, np.nan)
    return (float(np.nanmean(gaps)), float(np.nanmax(gaps)))

def _kgram_counts(symbols, k=2):
    s = np.asarray(symbols, dtype=float)
    s = s[np.isfinite(s)].astype(int)
    n = s.size
    out = {}
    if n < k:
        return out
    for i in range(n - k + 1):
        key = tuple(s[i:i+k].tolist())
        out[key] = out.get(key, 0) + 1
    total = float(max(1, n - k + 1))
    for k_ in list(out.keys()):
        out[k_] = out[k_] / total
    return out

# ===================== 窗口聚合（finalvalue） =====================
def agg_for_windows(df: pd.DataFrame, categories=None) -> pd.Series:
    """
    categories: None 或 list[str]，从 VALID_CATEGORIES 里选。
    None 表示计算全量；list 则仅计算/输出所选类别。
    """
    cats = None if (categories is None) else set(categories)
    if cats is not None:
        unknown = cats - VALID_CATEGORIES
        if unknown:
            raise ValueError(f"存在未知类别: {unknown}. 可选: {sorted(VALID_CATEGORIES)}")

    def _on(name: str) -> bool:
        return (cats is None) or (name in cats)

    out = {}

    # 规范/计算 date_diff
    if "date_diff" in df.columns:
        sub = df.sort_values("date_diff", ascending=True, kind="stable")
    elif "balance_date" in df.columns and "sample_datetime" in df.columns:
        tmp_dd = (
            pd.to_datetime(df["sample_datetime"], errors="coerce").dt.normalize()
            - pd.to_datetime(df["balance_date"], errors="coerce").dt.normalize()
        ).dt.days
        sub = df.assign(date_diff=tmp_dd).sort_values("date_diff", ascending=True, kind="stable")
    else:
        sub = df.copy()
        if "date_diff" not in sub.columns:
            sub["date_diff"] = np.nan

    sub = sub.reset_index(drop=True)

    n_rows = len(sub)

    # 安全转换：兼容 pandas nullable dtype，比如 Float64 / Int64 / Int8 里的 pd.NA
    # 把 pd.NA 统一转成 np.nan，避免：
    # cannot convert to '<class 'float'>'-dtype NumPy array with missing values
    def _to_float_array(col_name, default=None):
        if col_name not in sub.columns:
            return default

        return pd.to_numeric(sub[col_name], errors="coerce").to_numpy(
            dtype="float64",
            na_value=np.nan
        )

    dd = _to_float_array("date_diff", np.full(n_rows, np.nan, dtype=float))

    dd_clean = dd.copy()
    dd_clean[~np.isfinite(dd_clean)] = np.inf
    dd_clean = np.maximum(dd_clean, 0.0)
    ks = np.searchsorted(dd_clean, np.asarray(WINDOWS, dtype=float), side="right").astype(int)

    has = {c: (c in sub.columns) for c in [
        "is_balance_anomaly", "is_fluctuation_anomaly",
        "increasing_streak", "decreasing_streak",
        "increase_rate", "decrease_rate",
        "balance_deviation", "balance_unchanged",
        "is_big_expense", "drop_after_big_expense",
        "is_big_income", "net_inflow", "trend", "balance_change"
    ]}

    balance_arr = _to_float_array("balance", np.full(n_rows, np.nan, dtype=float))
    debit_arr = _to_float_array("expense_amount", np.zeros(n_rows, dtype=float))
    credit_arr = _to_float_array("income_amount", np.zeros(n_rows, dtype=float))

    net_inflow_arr = _to_float_array("net_inflow", None)
    trend_arr = _to_float_array("trend", None)
    balance_change_arr = _to_float_array("balance_change", None)

    anomaly_arr = _to_float_array("is_balance_anomaly", None)
    fluctuation_arr = _to_float_array("is_fluctuation_anomaly", None)

    increasing_streak_arr = _to_float_array("increasing_streak", None)
    decreasing_streak_arr = _to_float_array("decreasing_streak", None)

    increase_rate_arr = _to_float_array("increase_rate", None)
    decrease_rate_arr = _to_float_array("decrease_rate", None)

    balance_deviation_arr = _to_float_array("balance_deviation", None)
    unchanged_arr = _to_float_array("balance_unchanged", None)

    big_expense_arr = _to_float_array("is_big_expense", None)
    drop_after_big_expense_arr = _to_float_array("drop_after_big_expense", None)
    big_income_arr = _to_float_array("is_big_income", None)

    for w, k in zip(WINDOWS, ks):
        # --- 空窗口：仅为启用的类别生成空占位 ---
        if k == 0:
            if _on("basic"):
                out.update({
                    f"bank_txn_balance_max_{w}d": np.nan,
                    f"bank_txn_balance_min_{w}d": np.nan,
                    f"bank_txn_balance_mean_{w}d": np.nan,
                    f"bank_txn_balance_std_{w}d": np.nan,
                    f"bank_txn_balance_anomaly_days_{w}d": 0,
                    f"bank_txn_balance_fluctuation_anomaly_days_{w}d": 0,
                    f"bank_txn_balance_streak_inc_max_{w}d": np.nan,
                    f"bank_txn_balance_streak_dec_max_{w}d": np.nan,
                    f"bank_txn_balance_rate_inc_max_{w}d": np.nan,
                    f"bank_txn_balance_rate_dec_max_{w}d": np.nan,
                    f"bank_txn_balance_dev_from_hist_mean_{w}d": np.nan,
                    f"bank_txn_balance_unchanged_days_{w}d": 0,
                })
            if _on("debit"):
                out.update({
                    f"bank_txn_balance_debit_sum_{w}d": 0.0,
                    f"bank_txn_balance_debit_mean_{w}d": np.nan,
                    f"bank_txn_balance_debit_max_{w}d": np.nan,
                    f"bank_txn_balance_debit_days_{w}d": 0,
                    f"bank_txn_balance_debit_freq_{w}d": np.nan,
                    f"bank_txn_balance_debit_no_txn_ratio_{w}d": np.nan,
                    f"bank_txn_balance_debit_std_{w}d": np.nan,
                    f"bank_txn_balance_debit_cv_{w}d": np.nan,
                    f"bank_txn_balance_debit_max_ratio_{w}d": np.nan,
                    f"bank_txn_balance_debit_concentration_{w}d": np.nan,
                    f"bank_txn_balance_debit_fortnight_amount_{w}d": np.nan,
                    f"bank_txn_balance_debit_fortnight_days_{w}d": np.nan,
                })
                if has.get("is_big_expense", False):
                    out[f"bank_txn_balance_debit_big_ratio_{w}d"] = np.nan
            if _on("credit"):
                out.update({
                    f"bank_txn_balance_credit_sum_{w}d": 0.0,
                    f"bank_txn_balance_credit_mean_{w}d": np.nan,
                    f"bank_txn_balance_credit_max_{w}d": np.nan,
                    f"bank_txn_balance_credit_days_{w}d": 0,
                    f"bank_txn_balance_credit_freq_{w}d": np.nan,
                    f"bank_txn_balance_credit_no_txn_ratio_{w}d": np.nan,
                    f"bank_txn_balance_credit_std_{w}d": np.nan,
                    f"bank_txn_balance_credit_cv_{w}d": np.nan,
                    f"bank_txn_balance_credit_max_ratio_{w}d": np.nan,
                    f"bank_txn_balance_credit_concentration_{w}d": np.nan,
                    f"bank_txn_balance_credit_fortnight_amount_{w}d": np.nan,
                    f"bank_txn_balance_credit_fortnight_days_{w}d": np.nan,
                })
                if has.get("is_big_income", False):
                    out[f"bank_txn_balance_credit_big_ratio_{w}d"] = np.nan
            if _on("ratios"):
                out[f"bank_txn_balance_global_balance_to_debit_ratio_mean_{w}d"] = np.nan
                out[f"bank_txn_balance_debit_after_big_avg_drop_{w}d"] = np.nan

            if _on("timeseries"):
                for kname in [
                    "slope","extreme_ratio","debit_corr","netinflow_corr","autocorr1",
                    "quad_curvature","monotonicity","cv","volatility","max_drawdown",
                    "skewness","kurtosis","hurst"
                ]:
                    out[f"bank_txn_balance_{kname}_{w}d"] = np.nan

            if _on("regression"):
                out[f"bank_txn_balance_ols_slope_{w}d"] = np.nan
                out[f"bank_txn_balance_ols_r2_{w}d"] = np.nan
                out[f"bank_txn_balance_poly2_slope1_{w}d"] = np.nan
                out[f"bank_txn_balance_poly2_r2_{w}d"] = np.nan
                out[f"bank_txn_balance_ridge1_slope_{w}d"] = np.nan
                out[f"bank_txn_balance_ridge1_r2_{w}d"] = np.nan
                out[f"bank_txn_balance_theilsen_slope_{w}d"] = np.nan

            if _on("ar_ewma"):
                out[f"bank_txn_balance_ar1_phi_{w}d"] = np.nan
                out[f"bank_txn_balance_ar1_c_{w}d"] = np.nan
                out[f"bank_txn_balance_ar1_resid_std_{w}d"] = np.nan
                out[f"bank_txn_balance_ewma_resid_mae_{w}d"] = np.nan
                out[f"bank_txn_balance_ewma_resid_std_{w}d"] = np.nan
                out[f"bank_txn_balance_ewma_resid_ac1_{w}d"] = np.nan

            if _on("kmeans"):
                out[f"bank_txn_balance_kmeans2_sep_{w}d"]   = np.nan
                out[f"bank_txn_balance_kmeans2_wcss_{w}d"]  = np.nan
                out[f"bank_txn_balance_kmeans2_c0_{w}d"]    = np.nan
                out[f"bank_txn_balance_kmeans2_c1_{w}d"]    = np.nan
                out[f"bank_txn_balance_kmeans2_size0_{w}d"] = 0
                out[f"bank_txn_balance_kmeans2_size1_{w}d"] = 0

            if _on("pca"):
                out[f"bank_txn_balance_global_pca_var_exp_{w}d"] = np.nan
                out[f"bank_txn_balance_global_pca_pc1_loading_balance_{w}d"] = np.nan
                out[f"bank_txn_balance_global_pca_pc1_loading_credit_{w}d"]  = np.nan
                out[f"bank_txn_balance_global_pca_pc1_loading_debit_{w}d"]   = np.nan

            if _on("kgram"):
                out[f"bank_txn_balance_kgram1_pos_{w}d"]  = np.nan
                out[f"bank_txn_balance_kgram1_neg_{w}d"]  = np.nan
                out[f"bank_txn_balance_kgram1_zero_{w}d"] = np.nan
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        out[f"bank_txn_balance_kgram2{a}_{b}_{w}d"] = np.nan

            if _on("direction"):
                out[f"bank_txn_balance_sign_change_count_{w}d"] = np.nan
                out[f"bank_txn_balance_positive_return_ratio_{w}d"] = np.nan
                out[f"bank_txn_balance_below_avg_ratio_{w}d"] = np.nan

            if _on("pct_change"):
                out[f"bank_txn_balance_pct_change_mean_{w}d"] = np.nan
                out[f"bank_txn_balance_pct_change_std_{w}d"]  = np.nan

            if _on("autocorr"):
                out[f"bank_txn_balance_autocorr7_{w}d"] = np.nan
                out[f"bank_txn_balance_credit_autocorr7_{w}d"]  = np.nan
                out[f"bank_txn_balance_debit_autocorr7_{w}d"]   = np.nan

            if _on("snapshot"):
                out[f"bank_txn_balance_snapshot_last_{w}d"] = np.nan
                out[f"bank_txn_balance_credit_snapshot_last_{w}d"]  = np.nan
                out[f"bank_txn_balance_debit_snapshot_last_{w}d"]   = np.nan

            if _on("inequality"):
                out[f"bank_txn_balance_credit_gini_{w}d"]    = np.nan
                out[f"bank_txn_balance_debit_gini_{w}d"]     = np.nan
                out[f"bank_txn_balance_credit_entropy_{w}d"] = np.nan
                out[f"bank_txn_balance_debit_entropy_{w}d"]  = np.nan
                out[f"bank_txn_balance_credit_top1_share_{w}d"] = np.nan
                out[f"bank_txn_balance_debit_top1_share_{w}d"]  = np.nan

            if _on("gaps"):
                out[f"bank_txn_balance_credit_gap_mean_{w}d"] = np.nan
                out[f"bank_txn_balance_credit_gap_max_{w}d"]  = np.nan
                out[f"bank_txn_balance_debit_gap_mean_{w}d"]  = np.nan
                out[f"bank_txn_balance_debit_gap_max_{w}d"]   = np.nan

            if _on("tails"):
                out[f"bank_txn_balance_debit_tail_ratio_{w}d"]  = np.nan
                out[f"bank_txn_balance_credit_tail_ratio_{w}d"] = np.nan

            if _on("global"):
                out[f"bank_txn_balance_global_debit_credit_ratio_{w}d"] = np.nan
                out[f"bank_txn_balance_global_coverage_ratio_{w}d"]     = np.nan
                out[f"bank_txn_balance_global_savings_rate_{w}d"]       = np.nan
                out[f"bank_txn_balance_global_net_burn_per_day_{w}d"]   = np.nan

            continue  # 下一个窗口

        # --- 非空窗口 ---
        y = balance_arr[:k]
        deb = debit_arr[:k]
        cre = credit_arr[:k]
        net_w = net_inflow_arr[:k] if net_inflow_arr is not None else None
        trend_w = trend_arr[:k] if trend_arr is not None else None
        balance_change_w = balance_change_arr[:k] if balance_change_arr is not None else None
        n = y.size
        x_idx = np.arange(n, dtype=float)

        # —— 基础/异常/连续性
        if _on("basic"):
            out[f"bank_txn_balance_max_{w}d"] = _nanmax_or_nan(y)
            out[f"bank_txn_balance_min_{w}d"] = _nanmin_or_nan(y)
            out[f"bank_txn_balance_mean_{w}d"] = _nanmean_or_nan(y)
            out[f"bank_txn_balance_std_{w}d"] = _nanstd_or_nan(y, ddof=1)
            out[f"bank_txn_balance_anomaly_days_{w}d"] = int(np.nansum(anomaly_arr[:k])) if anomaly_arr is not None else 0
            out[f"bank_txn_balance_fluctuation_anomaly_days_{w}d"] = int(np.nansum(fluctuation_arr[:k])) if fluctuation_arr is not None else 0
            out[f"bank_txn_balance_streak_inc_max_{w}d"] = _nanmax_or_nan(increasing_streak_arr[:k]) if increasing_streak_arr is not None else np.nan
            out[f"bank_txn_balance_streak_dec_max_{w}d"] = _nanmax_or_nan(decreasing_streak_arr[:k]) if decreasing_streak_arr is not None else np.nan
            out[f"bank_txn_balance_rate_inc_max_{w}d"] = _nanmax_or_nan(increase_rate_arr[:k]) if increase_rate_arr is not None else np.nan
            out[f"bank_txn_balance_rate_dec_max_{w}d"] = _nanmax_or_nan(decrease_rate_arr[:k]) if decrease_rate_arr is not None else np.nan
            out[f"bank_txn_balance_dev_from_hist_mean_{w}d"] = _nanmean_or_nan(balance_deviation_arr[:k]) if balance_deviation_arr is not None else np.nan
            out[f"bank_txn_balance_unchanged_days_{w}d"] = int(np.nansum(unchanged_arr[:k])) if unchanged_arr is not None else 0

        # —— debit
        total_debit = float(np.nansum(deb))
        debit_days = int(np.sum(deb > 0))
        mean_deb = _nanmean_or_nan(deb)
        std_deb = _nanstd_or_nan(deb, ddof=1)
        if _on("debit"):
            debit_max = _nanmax_or_nan(deb)
            out[f"bank_txn_balance_debit_sum_{w}d"] = total_debit
            out[f"bank_txn_balance_debit_mean_{w}d"] = mean_deb
            out[f"bank_txn_balance_debit_max_{w}d"] = debit_max
            out[f"bank_txn_balance_debit_days_{w}d"] = int(debit_days)
            out[f"bank_txn_balance_debit_freq_{w}d"] = float(np.mean(deb > 0))
            out[f"bank_txn_balance_debit_no_txn_ratio_{w}d"] = float(np.mean(deb == 0))
            out[f"bank_txn_balance_debit_std_{w}d"] = std_deb
            out[f"bank_txn_balance_debit_cv_{w}d"] = (std_deb / mean_deb) if (pd.notna(mean_deb) and mean_deb != 0) else np.nan
            out[f"bank_txn_balance_debit_max_ratio_{w}d"] = (debit_max / total_debit) if total_debit > 0 else np.nan
            out[f"bank_txn_balance_debit_fortnight_amount_{w}d"] = _to_fortnightly(total_debit, w)
            out[f"bank_txn_balance_debit_fortnight_days_{w}d"] = _to_fortnightly(debit_days, w)
            out[f"bank_txn_balance_debit_concentration_{w}d"] = float(np.nansum((deb / total_debit) ** 2)) if total_debit > 0 else np.nan
            if big_expense_arr is not None:
                out[f"bank_txn_balance_debit_big_ratio_{w}d"] = _nanmean_or_nan(big_expense_arr[:k])

        # —— credit
        total_credit = float(np.nansum(cre))
        credit_days = int(np.sum(cre > 0))
        mean_cre = _nanmean_or_nan(cre)
        std_cre = _nanstd_or_nan(cre, ddof=1)
        if _on("credit"):
            credit_max = _nanmax_or_nan(cre)
            out[f"bank_txn_balance_credit_sum_{w}d"] = total_credit
            out[f"bank_txn_balance_credit_mean_{w}d"] = mean_cre
            out[f"bank_txn_balance_credit_max_{w}d"] = credit_max
            out[f"bank_txn_balance_credit_days_{w}d"] = int(credit_days)
            out[f"bank_txn_balance_credit_freq_{w}d"] = float(np.mean(cre > 0))
            out[f"bank_txn_balance_credit_no_txn_ratio_{w}d"] = float(np.mean(cre == 0))
            out[f"bank_txn_balance_credit_std_{w}d"] = std_cre
            out[f"bank_txn_balance_credit_cv_{w}d"] = (std_cre / mean_cre) if (pd.notna(mean_cre) and mean_cre != 0) else np.nan
            out[f"bank_txn_balance_credit_max_ratio_{w}d"] = (credit_max / total_credit) if total_credit > 0 else np.nan
            out[f"bank_txn_balance_credit_fortnight_amount_{w}d"] = _to_fortnightly(total_credit, w)
            out[f"bank_txn_balance_credit_fortnight_days_{w}d"] = _to_fortnightly(credit_days, w)
            out[f"bank_txn_balance_credit_concentration_{w}d"] = float(np.nansum((cre / total_credit) ** 2)) if total_credit > 0 else np.nan
            if big_income_arr is not None:
                out[f"bank_txn_balance_credit_big_ratio_{w}d"] = _nanmean_or_nan(big_income_arr[:k])

        # —— 比率/大额影响
        if _on("ratios"):
            deb_nonzero = deb.copy()
            deb_nonzero[deb_nonzero == 0] = np.nan
            out[f"bank_txn_balance_global_balance_to_debit_ratio_mean_{w}d"] = _nanmean_or_nan(y / deb_nonzero)
            if big_expense_arr is not None and drop_after_big_expense_arr is not None:
                out[f"bank_txn_balance_debit_after_big_avg_drop_{w}d"] = _nanmean_or_nan(
                    drop_after_big_expense_arr[:k][big_expense_arr[:k] == 1]
                )
            else:
                out[f"bank_txn_balance_debit_after_big_avg_drop_{w}d"] = np.nan

        # —— 时序/算法特征
        if _on("timeseries"):
            if n > 1:
                try:
                    slope = np.polyfit(x_idx, y, 1)[0]
                except Exception:
                    slope = np.nan
                out[f"bank_txn_balance_slope_{w}d"] = slope

                max_val, min_val = np.nanmax(y), np.nanmin(y)
                extreme_days = np.sum((y == max_val) | (y == min_val))
                out[f"bank_txn_balance_extreme_ratio_{w}d"] = extreme_days / n if n > 0 else np.nan

                out[f"bank_txn_balance_debit_corr_{w}d"] = _pearson_preserve_nan(y, deb)
                out[f"bank_txn_balance_netinflow_corr_{w}d"] = _pearson_preserve_nan(y, net_w) if net_w is not None else np.nan

                try:
                    out[f"bank_txn_balance_autocorr1_{w}d"] = pd.Series(y).autocorr(lag=1)
                except Exception:
                    out[f"bank_txn_balance_autocorr1_{w}d"] = np.nan

                if n >= 3 and np.nanstd(y, ddof=1) > 0:
                    x_norm = (x_idx - x_idx.min()) / (x_idx.max() - x_idx.min()) if n > 1 else x_idx
                    y_std  = (y - np.nanmean(y)) / np.nanstd(y, ddof=1)
                    try:
                        a2, b2 = np.polyfit(x_norm, y_std, 2)[:2]
                        quad_curv = 2 * a2
                    except Exception:
                        quad_curv = np.nan
                else:
                    quad_curv = np.nan
                out[f"bank_txn_balance_quad_curvature_{w}d"] = quad_curv

                diffs = np.diff(y)
                nonzero = diffs[np.abs(diffs) > 0]
                mono = np.abs(np.sign(nonzero).sum()) / nonzero.size if nonzero.size >= 1 else np.nan
                out[f"bank_txn_balance_monotonicity_{w}d"] = mono

                mu = np.nanmean(y)
                sd = _safe_std(y)
                out[f"bank_txn_balance_cv_{w}d"] = (sd / np.abs(mu)) if (pd.notna(sd) and np.isfinite(sd) and pd.notna(mu) and mu != 0) else np.nan

                r = pd.Series(y).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                out[f"bank_txn_balance_volatility_{w}d"] = r.std(ddof=1) if len(r) >= 2 else np.nan

                v = y
                if np.isfinite(v).sum() >= 2:
                    cummax = np.maximum.accumulate(np.nan_to_num(v, nan=-np.inf))
                    valid = cummax > 0
                    if valid.any():
                        dd_ratio = np.full_like(v, np.nan, dtype=float)
                        dd_ratio[valid] = v[valid] / cummax[valid] - 1.0
                        out[f"bank_txn_balance_max_drawdown_{w}d"] = np.nanmin(dd_ratio)
                    else:
                        out[f"bank_txn_balance_max_drawdown_{w}d"] = np.nan
                else:
                    out[f"bank_txn_balance_max_drawdown_{w}d"] = np.nan

                s = pd.Series(y)
                out[f"bank_txn_balance_skewness_{w}d"] = s.skew() if n >= 3 else np.nan
                out[f"bank_txn_balance_kurtosis_{w}d"] = s.kurt() if n >= 4 else np.nan
                out[f"bank_txn_balance_hurst_{w}d"] = _hurst_exponent(pd.Series(y))
            else:
                for kname in [
                    "slope","extreme_ratio","debit_corr","netinflow_corr","autocorr1",
                    "quad_curvature","monotonicity","cv","volatility","max_drawdown",
                    "skewness","kurtosis","hurst"
                ]:
                    out[f"bank_txn_balance_{kname}_{w}d"] = np.nan

        # —— 回归/稳健回归
        if _on("regression"):
            if n >= 3 and np.nanstd(y) > 0:
                b_ols, a_ols, r2_ols = _ols_linear(x_idx, y)
                out[f"bank_txn_balance_ols_slope_{w}d"] = b_ols
                out[f"bank_txn_balance_ols_r2_{w}d"] = r2_ols

                slope1_poly2, intercept_poly2, r2_poly2 = _ridge_regression(x_idx, y, degree=2, lam=0.0)
                out[f"bank_txn_balance_poly2_slope1_{w}d"] = slope1_poly2
                out[f"bank_txn_balance_poly2_r2_{w}d"] = r2_poly2

                slope1_ridge, intercept_ridge, r2_ridge = _ridge_regression(x_idx, y, degree=1, lam=1.0)
                out[f"bank_txn_balance_ridge1_slope_{w}d"] = slope1_ridge
                out[f"bank_txn_balance_ridge1_r2_{w}d"] = r2_ridge

                out[f"bank_txn_balance_theilsen_slope_{w}d"] = _theil_sen_slope(x_idx, y, max_pairs=20000)
            else:
                out[f"bank_txn_balance_ols_slope_{w}d"] = np.nan
                out[f"bank_txn_balance_ols_r2_{w}d"] = np.nan
                out[f"bank_txn_balance_poly2_slope1_{w}d"] = np.nan
                out[f"bank_txn_balance_poly2_r2_{w}d"] = np.nan
                out[f"bank_txn_balance_ridge1_slope_{w}d"] = np.nan
                out[f"bank_txn_balance_ridge1_r2_{w}d"] = np.nan
                out[f"bank_txn_balance_theilsen_slope_{w}d"] = np.nan

        # —— AR(1) + EWMA 残差
        if _on("ar_ewma"):
            phi, c_ar1, resid_std = _ar1_coef(y)
            out[f"bank_txn_balance_ar1_phi_{w}d"] = phi
            out[f"bank_txn_balance_ar1_c_{w}d"] = c_ar1
            out[f"bank_txn_balance_ar1_resid_std_{w}d"] = resid_std
            mae_e, std_e, ac1_e = _ewma_residual_stats(y, alpha=0.3)
            out[f"bank_txn_balance_ewma_resid_mae_{w}d"] = mae_e
            out[f"bank_txn_balance_ewma_resid_std_{w}d"] = std_e
            out[f"bank_txn_balance_ewma_resid_ac1_{w}d"] = ac1_e

        # —— 1D K-Means（金额多模态）
        if _on("kmeans"):
            if balance_change_w is not None and np.isfinite(balance_change_w).any():
                amt = np.abs(balance_change_w.astype(float))
            else:
                a1 = cre[cre > 0]
                a2 = deb[deb > 0]
                amt = np.concatenate([a1, a2]) if (a1.size + a2.size) > 0 else np.asarray([], dtype=float)
            c2, sz2, wcss2, sep2 = _kmeans_1d(np.asarray(amt), k=2)
            out[f"bank_txn_balance_kmeans2_sep_{w}d"]   = sep2
            out[f"bank_txn_balance_kmeans2_wcss_{w}d"]  = wcss2
            out[f"bank_txn_balance_kmeans2_c0_{w}d"]    = float(c2[0]) if np.isfinite(c2[0]) else np.nan
            out[f"bank_txn_balance_kmeans2_c1_{w}d"]    = float(c2[1]) if c2.size > 1 and np.isfinite(c2[1]) else np.nan
            out[f"bank_txn_balance_kmeans2_size0_{w}d"] = int(sz2[0]) if sz2.size > 0 else 0
            out[f"bank_txn_balance_kmeans2_size1_{w}d"] = int(sz2[1]) if sz2.size > 1 else 0

        # —— PCA（balance/credit/debit）
        if _on("pca"):
            mat = np.column_stack([
                y.astype(float, copy=False),
                cre.astype(float, copy=False),
                deb.astype(float, copy=False)
            ])
            mvalid = np.all(np.isfinite(mat), axis=1)
            mat = mat[mvalid]
            if mat.shape[0] >= 3 and np.all(np.nanstd(mat, axis=0) > 0):
                mu = mat.mean(axis=0)
                sd = mat.std(axis=0, ddof=1)
                Z = (mat - mu) / sd
                C = (Z.T @ Z) / (Z.shape[0] - 1)
                eigvals, eigvecs = np.linalg.eig(C)
                order = np.argsort(eigvals)[::-1]
                eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
                var_exp = float(np.real(eigvals[0]) / np.sum(np.real(eigvals))) if np.isfinite(eigvals).all() else np.nan
                pc1 = np.real(eigvecs[:, 0])
                if pc1[0] < 0:
                    pc1 = -pc1
                out[f"bank_txn_balance_global_pca_var_exp_{w}d"] = var_exp
                out[f"bank_txn_balance_global_pca_pc1_loading_balance_{w}d"] = float(pc1[0])
                out[f"bank_txn_balance_global_pca_pc1_loading_credit_{w}d"]  = float(pc1[1])
                out[f"bank_txn_balance_global_pca_pc1_loading_debit_{w}d"]   = float(pc1[2])
            else:
                out[f"bank_txn_balance_global_pca_var_exp_{w}d"] = np.nan
                out[f"bank_txn_balance_global_pca_pc1_loading_balance_{w}d"] = np.nan
                out[f"bank_txn_balance_global_pca_pc1_loading_credit_{w}d"]  = np.nan
                out[f"bank_txn_balance_global_pca_pc1_loading_debit_{w}d"]   = np.nan

        # —— k-gram（余额符号序列）
        if _on("kgram"):
            if trend_w is not None:
                sym = trend_w
            else:
                dif = np.diff(y)
                sym = np.sign(dif)
            c1 = _kgram_counts(sym, k=1)
            out[f"bank_txn_balance_kgram1_pos_{w}d"]  = c1.get((1,), np.nan)
            out[f"bank_txn_balance_kgram1_neg_{w}d"]  = c1.get((-1,), np.nan)
            out[f"bank_txn_balance_kgram1_zero_{w}d"] = c1.get((0,), np.nan)
            c2g = _kgram_counts(sym, k=2)
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    out[f"bank_txn_balance_kgram2{a}_{b}_{w}d"] = c2g.get((a, b), np.nan)

        # —— 方向性/稳定性
        if _on("direction"):
            if n >= 2:
                diffs2 = np.diff(y)
                sign_changes = np.sum(np.sign(diffs2[1:]) != np.sign(diffs2[:-1]))
                pos_ret_ratio = float((pd.Series(y).pct_change() > 0).mean())
            else:
                sign_changes = np.nan
                pos_ret_ratio = np.nan
            out[f"bank_txn_balance_sign_change_count_{w}d"] = sign_changes
            out[f"bank_txn_balance_positive_return_ratio_{w}d"] = pos_ret_ratio
            out[f"bank_txn_balance_below_avg_ratio_{w}d"] = float(np.mean(y < _nanmean_or_nan(y))) if y.size > 0 else np.nan

        # —— 余额变化率统计
        if _on("pct_change"):
            pct = pd.Series(y).pct_change().replace([np.inf, -np.inf], np.nan)
            out[f"bank_txn_balance_pct_change_mean_{w}d"] = float(pct.mean()) if pct.size > 0 else np.nan
            out[f"bank_txn_balance_pct_change_std_{w}d"]  = float(pct.std(ddof=1)) if pct.size > 1 else np.nan

        # —— 周期性（lag=7）
        if _on("autocorr"):
            out[f"bank_txn_balance_autocorr7_{w}d"] = _autocorr_lag(y, 7)
            out[f"bank_txn_balance_credit_autocorr7_{w}d"]  = _autocorr_lag(cre.astype(float, copy=False), 7)
            out[f"bank_txn_balance_debit_autocorr7_{w}d"]   = _autocorr_lag(deb.astype(float, copy=False), 7)

        # —— 期末快照
        if _on("snapshot"):
            out[f"bank_txn_balance_snapshot_last_{w}d"] = float(y[-1]) if y.size else np.nan
            out[f"bank_txn_balance_credit_snapshot_last_{w}d"]  = float(cre[-1]) if cre.size else np.nan
            out[f"bank_txn_balance_debit_snapshot_last_{w}d"]   = float(deb[-1]) if deb.size else np.nan

        # —— 不均衡/熵/份额
        if _on("inequality"):
            out[f"bank_txn_balance_credit_gini_{w}d"]    = _gini_nonneg(cre)
            out[f"bank_txn_balance_debit_gini_{w}d"]     = _gini_nonneg(deb)
            out[f"bank_txn_balance_credit_entropy_{w}d"] = _entropy_share(cre)
            out[f"bank_txn_balance_debit_entropy_{w}d"]  = _entropy_share(deb)
            out[f"bank_txn_balance_credit_top1_share_{w}d"] = _topk_share(cre, 1)
            out[f"bank_txn_balance_debit_top1_share_{w}d"]  = _topk_share(deb, 1)

        # —— 间隔
        if _on("gaps"):
            dd_w = dd[:k]
            ig_mean, ig_max = _gap_stats((cre > 0), dd_w)
            eg_mean, eg_max = _gap_stats((deb > 0), dd_w)
            out[f"bank_txn_balance_credit_gap_mean_{w}d"] = ig_mean
            out[f"bank_txn_balance_credit_gap_max_{w}d"]  = ig_max
            out[f"bank_txn_balance_debit_gap_mean_{w}d"]  = eg_mean
            out[f"bank_txn_balance_debit_gap_max_{w}d"]   = eg_max

        # —— 尾部比例
        if _on("tails"):
            debit_tail = np.nan
            if pd.notna(mean_deb) and pd.notna(std_deb) and std_deb > 0 and deb.size > 0:
                debit_tail = float(((deb > (mean_deb + 2 * std_deb)).sum()) / deb.size)
            credit_tail = np.nan
            if pd.notna(mean_cre) and pd.notna(std_cre) and std_cre > 0 and cre.size > 0:
                credit_tail = float(((cre > (mean_cre + 2 * std_cre)).sum()) / cre.size)
            out[f"bank_txn_balance_debit_tail_ratio_{w}d"]  = debit_tail
            out[f"bank_txn_balance_credit_tail_ratio_{w}d"] = credit_tail

        # —— 全局概览
        if _on("global"):
            eps = 1e-12
            debit_to_credit = total_debit / (total_credit + eps) if total_credit > 0 else np.nan
            coverage   = total_credit / (total_debit + eps) if total_debit > 0 else np.nan
            savings    = (total_credit - total_debit) / total_credit if total_credit > 0 else np.nan
            net_burn   = (total_debit - total_credit) / float(w)
            out[f"bank_txn_balance_global_debit_credit_ratio_{w}d"] = debit_to_credit
            out[f"bank_txn_balance_global_coverage_ratio_{w}d"]     = coverage
            out[f"bank_txn_balance_global_savings_rate_{w}d"]       = savings
            out[f"bank_txn_balance_global_net_burn_per_day_{w}d"]   = net_burn

    return pd.Series(out)

# ===================== 主函数：输入DF → 输出DF =====================
def compute_balance_timeseries_features(df_input: pd.DataFrame, categories=None) -> pd.DataFrame:
    """
    参数
    ----
    df_input : pd.DataFrame
        必需列：
            user_id, bank_account_id, balance, balance_date
        其一列必须存在：
            sample_datetime 或 flowTime（若给 flowTime 会自动映射到 sample_datetime）
        可选列：
            application_id（当前函数不输出该字段，但会保留原有主键粒度 user_id + sample_datetime）

    categories : None 或 list[str]
        从 VALID_CATEGORIES 里选择要计算的类别。
        None 表示全量；list 表示仅计算/输出所选类别。

    返回
    ----
    pd.DataFrame
        按 (user_id, sample_datetime) 聚合的特征表（变量名和数量不变）
    """
    # 1) 先生成“用户级日余额序列”的明细派生结果
    df_detail = transform_s_l_a_d(df_input)

    # 2) 在用户级序列上做窗口聚合，避免多卡行直接混算
    groups = df_detail.groupby(["user_id", "sample_datetime"], sort=False, dropna=False)

    rows = []
    for (uid, sdt), g in groups:
        row = agg_for_windows(g, categories=categories)
        row["user_id"] = uid
        row["sample_datetime"] = sdt
        rows.append(row)

    features_df = pd.DataFrame(rows).reset_index(drop=True)

    cols = ["user_id", "sample_datetime"] + [
        c for c in features_df.columns if c not in ("user_id", "sample_datetime")
    ]
    return features_df.loc[:, cols]

# ===================== 用法示例（注释示例，非执行） =====================

# 你的这批数据只有一个 user_id、一个 job_id、多张 bank_account_id 且 flowTime 相同
# 可直接把 flowTime 当做 sample_datetime 传入（两者任何一个存在即可）：

# 示例：
# df_raw = pd.DataFrame({
#     "user_id":[10039,10039,10039],
#     "job_id":[663278,663278,663278],
#     "balance_date":["2023-07-14 00:00:00.000000 UTC","2023-07-15 00:00:00.000000 UTC","2023-07-17 00:00:00.000000 UTC"],
#     "balance":[21309.21,21060.76,20910.76],
#     "bank_account_id":[1807393,1807393,1807393],
#     "flowTime":["\"2024-01-10 06:59:11.0\"","\"2024-01-10 06:59:11.0\"","\"2024-01-10 06:59:11.0\""]
# })
#features_all = compute_balance_timeseries_features(df_raw, categories=None)
# features_basic_credit = compute_balance_timeseries_features(df_raw, categories=["basic","credit","inequality"])
