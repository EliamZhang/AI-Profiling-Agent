import pandas as pd
import numpy as np
import re
from . import load_refdata


class TxnListSampleVar:
    """给定样本的txn lender相关变量
    """
    def __init__(self, user_id, send_time, raw_data):
        self.user_id = user_id
        self.send_time = pd.to_datetime(send_time)
        self.raw_data = raw_data
        self.variables = {"user_id": user_id, "sample_datetime": self.send_time}
        self.prod_dict = {
            'personal loan': 'loan',
            'BNPL': 'bnpl',
            'cash/wage advance': 'advance',
            'bank': 'bank'
        }
        self.ALLOWED_ACCOUNT_TYPES = ["transaction", "investments", "savings", "trading", "Unlabeled"]
        # 以下属性将在方法中赋值
        self.raw_cate_list_df = None
        self.raw_cate_g_list_df = None

    # ----------------------------------------------------------------------
    # 内部辅助方法
    # ----------------------------------------------------------------------
    def _var_name(self, prefix, category_key, suffix, day=None):
        """生成标准化的变量名（用于产品类别）"""
        base = f"bank_txn_lender_{prefix}_{self.prod_dict[category_key]}_{suffix}"
        return f"{base}_l{day}d" if day else base

    def _competitor_var_name(self, prefix, competitor, suffix, day=None):
        """生成标准化的变量名（用于具体机构）"""
        # 清洗机构名：保留中文、字母、数字，其余替换为下划线
        cleaned = re.sub(r'[^\w\u4e00-\u9fff]+', '_', str(competitor))
        base = f"bank_txn_lender_{prefix}_competitor_{cleaned}_{suffix}"
        return f"{base}_l{day}d" if day else base

    def _filter_by_day(self, df, day):
        """按距离天数筛选数据"""
        if day is not None:
            return df.loc[df["trac_diffdays"] <= day].copy()
        return df.copy()

    # ----------------------------------------------------------------------
    # 数据清洗与准备
    # ----------------------------------------------------------------------
    def clean_txn_step1(self):
        """清洗txn lender数据第1步：选择和LOAN相关的数据"""
        df = self.raw_data.copy()
        df = df.loc[df.category.str.lower().str.contains('loan', na=False), :]  # 筛选loan
        df = df.loc[
            df["account_type"].isin(self.ALLOWED_ACCOUNT_TYPES)
            | df["account_type"].isna()
            | (df["account_type"] == "")
        ]#筛选卡类型
        df['amount'] = df.amount.apply(lambda x: abs(x))  # 金额取绝对值
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['trac_diffdays'] = (self.send_time - df['transaction_date']).dt.days
        self.raw_data = df

    def create_lender_cate_list(self, lender_cate_mapping_df):
        """生成交易对手类别List数据：匹配类别标签，去重无类别APP"""
        df = self.raw_data.copy()
        df = pd.merge(df, lender_cate_mapping_df, on=["third_party"], how="left")
        self.raw_cate_list_df = df

    def grouped_app_cate_list(self):
        """聚合lender类别List数据（按用户、机构、产品类型、日期、方向、类别）"""
        df = self.raw_cate_list_df.copy()
        df = df.groupby(
            ["user_id", "competitor", "product_type", "transaction_date", "dr_cr", "category"]
        ).agg(
            amount=pd.NamedAgg(column="amount", aggfunc="sum"),
            transaction_cnt=pd.NamedAgg(column="transaction_id", aggfunc="count"),
        ).reset_index()
        self.raw_cate_g_list_df = df

    # ----------------------------------------------------------------------
    # 还款/放款金额、次数等统计（按产品类别）
    # ----------------------------------------------------------------------
    def _calc_lender_stat(self, dr_cr, prefix, cate_list, day_list):
        """
        通用方法：计算指定方向（debit/credit）下各产品类别的统计量（总和、平均、次数、最大、最小）
        """
        def __calc(df, df_cate, day):
            # 构造变量名映射
            varnames_sum = {x: self._var_name(prefix, x, 'amount_sum', day) for x in cate_list}
            varnames_avg = {x: self._var_name(prefix, x, 'amount_avg', day) for x in cate_list}
            varnames_cnt = {x: self._var_name(prefix, x, 'count', day) for x in cate_list}
            varnames_max = {x: self._var_name(prefix, x, 'amount_max', day) for x in cate_list}
            varnames_min = {x: self._var_name(prefix, x, 'amount_min', day) for x in cate_list}

            variables = {}
            for d in [varnames_sum, varnames_avg, varnames_cnt, varnames_max, varnames_min]:
                variables.update({v: -1 for v in d.values()})

            if day is not None:
                df = df.loc[df["trac_diffdays"] <= day].copy()
                df_cate = df_cate.loc[df_cate["trac_diffdays"] <= day].copy()

            if df.shape[0] > 0:
                df_cate_sub = df_cate.loc[df_cate["product_type"].isin(cate_list)].copy()
                if df_cate_sub.shape[0] > 0:
                    # 求和
                    sum_vals = df_cate_sub.groupby("product_type")["amount"].sum().to_dict()
                    for cat, val in sum_vals.items():
                        variables[varnames_sum[cat]] = val
                    # 平均
                    avg_vals = df_cate_sub.groupby("product_type")["amount"].mean().to_dict()
                    for cat, val in avg_vals.items():
                        variables[varnames_avg[cat]] = val
                    # 计数
                    cnt_vals = df_cate_sub.groupby("product_type")["transaction_id"].count().to_dict()
                    for cat, val in cnt_vals.items():
                        variables[varnames_cnt[cat]] = val
                    # 最大
                    max_vals = df_cate_sub.groupby("product_type")["amount"].max().to_dict()
                    for cat, val in max_vals.items():
                        variables[varnames_max[cat]] = val
                    # 最小
                    min_vals = df_cate_sub.groupby("product_type")["amount"].min().to_dict()
                    for cat, val in min_vals.items():
                        variables[varnames_min[cat]] = val
            return variables

        df = self.raw_data.loc[self.raw_data['dr_cr'] == dr_cr].copy()
        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr'] == dr_cr].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df, df_cate, day))
        self.variables.update(variables)

    def calc_vars_repay_stat(self, cate_list, day_list=[None]):
        """计算近多段时间内指定类别还款（debit）次数和金额（包含总和、平均、次数、最大、最小）"""
        self._calc_lender_stat('debit', 'repay', cate_list, day_list)

    def calc_vars_disburse_stat(self, cate_list, day_list=[None]):
        """计算近多段时间内指定类别放款（credit）次数和金额（包含总和、平均、次数、最大、最小）"""
        self._calc_lender_stat('credit', 'disburse', cate_list, day_list)

    # ----------------------------------------------------------------------
    # 还款/放款整体统计（不分产品类别）
    # ----------------------------------------------------------------------
    def _calc_summary_stat(self, dr_cr, prefix, day_list):
        """通用方法：计算还款/放款整体的总和、平均、次数、最小、最大"""
        def __calc(df, day):
            df_day = self._filter_by_day(df, day)
            suffix = f"l{day}d" if day else ""
            varname_sum = f"bank_txn_lender_{prefix}_amount_sum" + (f"_{suffix}" if suffix else "")
            varname_avg = f"bank_txn_lender_{prefix}_amount_avg" + (f"_{suffix}" if suffix else "")
            varname_cnt = f"bank_txn_lender_{prefix}_amount_count" + (f"_{suffix}" if suffix else "")
            varname_min = f"bank_txn_lender_{prefix}_amount_min" + (f"_{suffix}" if suffix else "")
            varname_max = f"bank_txn_lender_{prefix}_amount_max" + (f"_{suffix}" if suffix else "")

            variables = {
                varname_sum: -1,
                varname_avg: -1,
                varname_cnt: -1,
                varname_min: -1,
                varname_max: -1
            }
            if df_day.shape[0] > 0:
                variables.update({
                    varname_sum: df_day.amount.sum(),
                    varname_avg: df_day.amount.mean(),
                    varname_cnt: df_day.shape[0],
                    varname_min: df_day.amount.min(),
                    varname_max: df_day.amount.max()
                })
            return variables

        df = self.raw_data.loc[self.raw_data['dr_cr'] == dr_cr].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df, day))
        self.variables.update(variables)

    def calc_vars_summary_repay_stat(self, day_list=[None, 1]):
        """计算近多段时间内还款（debit）的整体统计"""
        self._calc_summary_stat('debit', 'repay', day_list)

    def calc_vars_summary_disburse_stat(self, day_list=[None, 1]):
        """计算近多段时间内放款（credit）的整体统计"""
        self._calc_summary_stat('credit', 'disburse', day_list)

    # ----------------------------------------------------------------------
    # 单机构最大累计还款/借款（按机构聚合后取最大）
    # ----------------------------------------------------------------------
    def _calc_single_stat(self, dr_cr, prefix, cate_list):
        """通用方法：计算历史单机构最大累计还款/借款金额和次数（按产品类别）"""
        raw_grouped = self.raw_cate_g_list_df.loc[self.raw_cate_g_list_df['dr_cr'] == dr_cr].copy()
        if raw_grouped.shape[0] == 0:
            # 无数据，直接设置默认值并返回
            variables = {}
            for x in cate_list:
                variables[self._var_name(prefix, x, 'count_max')] = -1
                variables[self._var_name(prefix, x, 'amount_max')] = -1
            variables.update({
                f"bank_txn_lender_{prefix}_single_count_max": -1,
                f"bank_txn_lender_{prefix}_single_amount_max": -1
            })
            self.variables.update(variables)
            return

        # 按机构聚合（competitor+product_type）
        agg_df = raw_grouped.groupby(["user_id", "competitor", "product_type"]).agg(
            amount=pd.NamedAgg(column="amount", aggfunc="sum"),
            transaction_cnt=pd.NamedAgg(column="transaction_cnt", aggfunc="sum"),
        ).reset_index()

        # 构建变量
        varnames_cnt = {x: self._var_name(prefix, x, 'count_max') for x in cate_list}
        varnames_amt = {x: self._var_name(prefix, x, 'amount_max') for x in cate_list}
        variables = {varnames_cnt[x]: -1 for x in cate_list}
        variables.update({varnames_amt[x]: -1 for x in cate_list})
        variables.update({
            f"bank_txn_lender_{prefix}_single_count_max": -1,
            f"bank_txn_lender_{prefix}_single_amount_max": -1
        })

        if agg_df.shape[0] > 0:
            df_sub = agg_df.loc[agg_df["product_type"].isin(cate_list)].copy()
            if df_sub.shape[0] > 0:
                cnt_max_by_cat = df_sub.groupby("product_type")["transaction_cnt"].max().to_dict()
                for cat, val in cnt_max_by_cat.items():
                    variables[varnames_cnt[cat]] = val
                amt_max_by_cat = df_sub.groupby("product_type")["amount"].max().to_dict()
                for cat, val in amt_max_by_cat.items():
                    variables[varnames_amt[cat]] = val

            # 汇总：使用原始分组数据（raw_grouped）中的最大值
            if raw_grouped.shape[0] > 0:
                variables[f"bank_txn_lender_{prefix}_single_count_max"] = raw_grouped["transaction_cnt"].max()
                variables[f"bank_txn_lender_{prefix}_single_amount_max"] = raw_grouped["amount"].max()

        self.variables.update(variables)

    def calc_vars_repay_single_stat(self, cate_list=['personal loan', 'BNPL', 'cash/wage advance', 'bank']):
        """计算历史单机构最大累计还款金额/次数"""
        self._calc_single_stat('debit', 'repay', cate_list)

    def calc_vars_disburse_single_stat(self, cate_list=['personal loan', 'BNPL', 'cash/wage advance', 'bank']):
        """计算历史单机构最大累计借款金额/次数"""
        self._calc_single_stat('credit', 'disburse', cate_list)

    # ----------------------------------------------------------------------
    # 机构数统计（整体和分产品类别）
    # ----------------------------------------------------------------------
    def _calc_summary_jgs_stat(self, dr_cr, prefix, day_list):
        """通用方法：计算整体机构数（不分产品类别）"""
        def __calc(df_cate, day):
            df_day = self._filter_by_day(df_cate, day)
            varname = f"bank_txn_lender_{prefix}_jgs"
            if day:
                varname += f"_l{day}d"
            variables = {varname: -1}
            if df_day.shape[0] > 0:
                variables[varname] = df_day["competitor"].nunique()
            return variables

        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr'] == dr_cr].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df_cate, day))
        self.variables.update(variables)

    def calc_vars_summary_repay_jgs_stat(self, day_list=[None]):
        """计算近多段时间内还款机构数（整体）"""
        self._calc_summary_jgs_stat('debit', 'repay', day_list)

    def calc_vars_summary_disburse_jgs_stat(self, day_list=[None]):
        """计算近多段时间内放款机构数（整体）"""
        self._calc_summary_jgs_stat('credit', 'disburse', day_list)

    def _calc_jgs_stat(self, dr_cr, prefix, cate_list, day_list):
        """通用方法：计算各产品类别的机构数"""
        def __calc(df_cate, day):
            df_day = self._filter_by_day(df_cate, day)
            varnames = {x: self._var_name(prefix, x, 'jgs', day) for x in cate_list}
            variables = {varnames[x]: -1 for x in cate_list}
            if df_day.shape[0] > 0:
                df_sub = df_day.loc[df_day["product_type"].isin(cate_list)].copy()
                if df_sub.shape[0] > 0:
                    jgs_by_cat = df_sub.groupby("product_type")["competitor"].nunique().to_dict()
                    for cat, val in jgs_by_cat.items():
                        variables[varnames[cat]] = val
            return variables

        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr'] == dr_cr].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df_cate, day))
        self.variables.update(variables)

    def calc_vars_repay_jgs_stat(self, cate_list, day_list=[None]):
        """计算近多段时间内指定类别还款机构数"""
        self._calc_jgs_stat('debit', 'repay', cate_list, day_list)

    def calc_vars_disburse_jgs_stat(self, cate_list, day_list=[None]):
        """计算近多段时间内指定类别放款机构数"""
        self._calc_jgs_stat('credit', 'disburse', cate_list, day_list)

    # ----------------------------------------------------------------------
    # 针对指定机构列表（competitor）的统计
    # ----------------------------------------------------------------------
    def _calc_competitor_stat(self, dr_cr, prefix, competitor_list, day_list):
        """
        通用方法：计算指定方向下，给定机构列表的统计量（总和、平均、次数、最大、最小）
        """
        def __calc(df, df_cate, day):
            # 构造变量名映射
            varnames_sum = {comp: self._competitor_var_name(prefix, comp, 'amount_sum', day) for comp in competitor_list}
            varnames_avg = {comp: self._competitor_var_name(prefix, comp, 'amount_avg', day) for comp in competitor_list}
            varnames_cnt = {comp: self._competitor_var_name(prefix, comp, 'count', day) for comp in competitor_list}
            varnames_max = {comp: self._competitor_var_name(prefix, comp, 'amount_max', day) for comp in competitor_list}
            varnames_min = {comp: self._competitor_var_name(prefix, comp, 'amount_min', day) for comp in competitor_list}

            variables = {}
            for d in [varnames_sum, varnames_avg, varnames_cnt, varnames_max, varnames_min]:
                variables.update({v: -1 for v in d.values()})

            if day is not None:
                df = df.loc[df["trac_diffdays"] <= day].copy()
                df_cate = df_cate.loc[df_cate["trac_diffdays"] <= day].copy()

            if df.shape[0] > 0:
                # 只保留指定的competitor
                df_cate_sub = df_cate.loc[df_cate["competitor"].isin(competitor_list)].copy()
                if df_cate_sub.shape[0] > 0:
                    # 总和
                    sum_vals = df_cate_sub.groupby("competitor")["amount"].sum().to_dict()
                    for comp, val in sum_vals.items():
                        variables[varnames_sum[comp]] = val
                    # 平均
                    avg_vals = df_cate_sub.groupby("competitor")["amount"].mean().to_dict()
                    for comp, val in avg_vals.items():
                        variables[varnames_avg[comp]] = val
                    # 计数
                    cnt_vals = df_cate_sub.groupby("competitor")["transaction_id"].count().to_dict()
                    for comp, val in cnt_vals.items():
                        variables[varnames_cnt[comp]] = val
                    # 最大
                    max_vals = df_cate_sub.groupby("competitor")["amount"].max().to_dict()
                    for comp, val in max_vals.items():
                        variables[varnames_max[comp]] = val
                    # 最小
                    min_vals = df_cate_sub.groupby("competitor")["amount"].min().to_dict()
                    for comp, val in min_vals.items():
                        variables[varnames_min[comp]] = val
            return variables

        df = self.raw_data.loc[self.raw_data['dr_cr'] == dr_cr].copy()
        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr'] == dr_cr].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df, df_cate, day))
        self.variables.update(variables)

    def calc_vars_repay_competitor_stat(self, competitor_list, day_list=[None]):
        """
        计算近多段时间内指定机构（competitor_list）的还款（debit）统计量
        （包含总和、平均、次数、最大、最小）
        """
        self._calc_competitor_stat('debit', 'repay', competitor_list, day_list)

    def calc_vars_disburse_competitor_stat(self, competitor_list, day_list=[None]):
        """
        计算近多段时间内指定机构（competitor_list）的放款（credit）统计量
        （包含总和、平均、次数、最大、最小）
        """
        self._calc_competitor_stat('credit', 'disburse', competitor_list, day_list)