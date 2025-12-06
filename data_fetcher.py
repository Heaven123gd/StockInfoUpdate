"""
数据获取模块
封装所有 akshare 数据获取逻辑，每个函数接收日期参数
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict
from config import (
    FINANCE_KEYWORDS, BANK_KEYWORDS,
    INDEX_SYMBOLS_AKSHARE, INDEX_SYMBOLS_LOCAL,
    INDEX_START_DATE, find_local_index_file,
    TRADING_DAYS_PER_YEAR, BASE_CPI_VALUE,
    DJI_SYMBOL, DJI_NAME, COMPARISON_START_DATE,
    FEATURED_STOCKS, STOCK_EXCHANGE_SUFFIX
)


def retry_request(func, max_retries=3, delay=2):
    """
    带重试机制的请求包装器

    Args:
        func: 要执行的函数
        max_retries: 最大重试次数
        delay: 重试间隔（秒）

    Returns:
        函数执行结果
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # 递增延迟
    raise last_error


def fetch_sse_summary() -> pd.DataFrame:
    """
    获取上交所市场统计数据

    Returns:
        DataFrame: 上交所统计数据（流通股本、总市值、上市公司数等）
    """
    try:
        df = retry_request(lambda: ak.stock_sse_summary())
        return df
    except Exception as e:
        raise Exception(f"获取上交所数据失败: {str(e)}")


def fetch_szse_summary(date: str) -> pd.DataFrame:
    """
    获取深交所市场统计数据

    Args:
        date: 日期字符串，格式 'YYYYMMDD'

    Returns:
        DataFrame: 深交所统计数据
    """
    try:
        df = retry_request(lambda: ak.stock_szse_summary(date=date))
        return df
    except Exception as e:
        raise Exception(f"获取深交所数据失败: {str(e)}")


def fetch_stock_yjbb(date: str) -> pd.DataFrame:
    """
    获取A股业绩报表数据

    Args:
        date: 报告期日期，格式 'YYYYMMDD'（如 '20250930' 表示三季报）

    Returns:
        DataFrame: 业绩报表数据
    """
    try:
        df = retry_request(lambda: ak.stock_yjbb_em(date=date), max_retries=5, delay=3)
        return df
    except Exception as e:
        raise Exception(f"获取业绩报表数据失败: {str(e)}")


def fetch_stock_code_name() -> pd.DataFrame:
    """
    获取A股股票代码和名称列表

    Returns:
        DataFrame: 包含 code 和 name 列的数据
    """
    try:
        df = retry_request(lambda: ak.stock_info_a_code_name(), max_retries=5, delay=3)
        return df
    except Exception as e:
        raise Exception(f"获取股票代码列表失败: {str(e)}")


def fetch_profit_top100(report_date: str) -> Tuple[pd.DataFrame, dict]:
    """
    获取A股归母净利润前100强及统计信息

    Args:
        report_date: 报告期日期，格式 'YYYYMMDD'

    Returns:
        Tuple[DataFrame, dict]: (净利润前100强数据, 统计信息字典)
        统计信息字典包含:
        - top100_total: 100强归母净利润总计（元）
        - all_total: 全A股归母净利润总计（元）
        - ratio: 100强占比（百分比）
        - all_count: 全A股公司数量
        - profit_count: 盈利公司数量
        - loss_count: 亏损公司数量
    """
    try:
        # 获取股票代码名称
        stock_info = fetch_stock_code_name()
        # 获取业绩报表
        yjbb = fetch_stock_yjbb(report_date)
        # 合并数据
        merged = pd.merge(
            stock_info,
            yjbb,
            left_on='code',
            right_on='股票代码',
            how='inner'
        )
        # 按净利润排序取前100
        profit_col = '净利润-净利润'
        if profit_col in merged.columns:
            merged[profit_col] = pd.to_numeric(merged[profit_col], errors='coerce')

            # 计算全A股统计信息
            all_total = merged[profit_col].sum(skipna=True)
            all_count = len(merged)
            profit_count = (merged[profit_col] > 0).sum()
            loss_count = (merged[profit_col] < 0).sum()

            # 排序取前100
            top100 = merged.sort_values(by=profit_col, ascending=False).head(100)
            top100_total = top100[profit_col].sum(skipna=True)

            # 计算占比
            ratio = (top100_total / all_total * 100) if all_total != 0 else 0

            stats = {
                'top100_total': top100_total,
                'all_total': all_total,
                'ratio': ratio,
                'all_count': all_count,
                'profit_count': profit_count,
                'loss_count': loss_count
            }

            return top100, stats
        else:
            raise Exception(f"未找到净利润列，可用列: {list(merged.columns)}")
    except Exception as e:
        raise Exception(f"获取净利润100强失败: {str(e)}")


def calculate_profit_distribution(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    计算净利润行业分布（银行、其他金融、其他）

    Args:
        df: 包含净利润和行业信息的 DataFrame

    Returns:
        Tuple[DataFrame, dict]: 分布统计表和饼图数据
    """
    try:
        # 找到净利润列和行业列
        profit_col = next((c for c in df.columns if '净利' in c), None)
        industry_col = next((c for c in df.columns if '行业' in c or '所属' in c), None)

        if profit_col is None:
            raise Exception(f"未找到净利润列，可用列: {list(df.columns)}")

        # 转为数值型
        df = df.copy()
        df[profit_col] = pd.to_numeric(df[profit_col], errors='coerce')
        total_profit = df[profit_col].sum(skipna=True)

        if pd.isna(total_profit) or total_profit == 0:
            raise Exception("总净利润为0或缺失，无法计算占比")

        if industry_col is None:
            # 没有行业列，全部归为其他
            bank_profit = 0.0
            other_fin_profit = 0.0
            others_profit = total_profit
        else:
            df[industry_col] = df[industry_col].astype(str)
            # 银行业
            bank_mask = df[industry_col].str.contains('|'.join(BANK_KEYWORDS), na=False)
            # 其他金融（排除银行）
            other_fin_mask = df[industry_col].str.contains(
                '|'.join(FINANCE_KEYWORDS), na=False
            ) & ~bank_mask

            bank_profit = df.loc[bank_mask, profit_col].sum(skipna=True)
            other_fin_profit = df.loc[other_fin_mask, profit_col].sum(skipna=True)
            others_profit = total_profit - bank_profit - other_fin_profit

        # 构建结果
        labels = ['银行', '其他金融', '其他行业']
        values = [bank_profit, other_fin_profit, others_profit]
        percentages = [v / total_profit * 100 for v in values]

        result_df = pd.DataFrame({
            '类别': labels,
            '净利润(亿元)': [round(v / 1e8, 2) for v in values],
            '占比(%)': [round(p, 2) for p in percentages]
        })

        pie_data = {'labels': labels, 'values': values, 'percentages': percentages}

        return result_df, pie_data

    except Exception as e:
        raise Exception(f"计算利润分布失败: {str(e)}")


# =====================
# 指数数据获取函数
# =====================

def fetch_index_daily_akshare(symbol: str, start_date: str = None) -> pd.DataFrame:
    """
    通过 akshare 获取指数日线数据（上证/深证）

    Args:
        symbol: 指数代码，如 'sh000001'
        start_date: 起始日期，格式 'YYYYMMDD'

    Returns:
        DataFrame: 包含 date, open, high, low, close, volume 列
    """
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        # 标准化列名
        df = df.rename(columns={
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        # 确保日期格式
        df['date'] = pd.to_datetime(df['date'])
        # 按日期筛选
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        raise Exception(f"获取指数 {symbol} 数据失败: {str(e)}")


def parse_uploaded_index_file(uploaded_file, start_date: str = None) -> pd.DataFrame:
    """
    解析用户上传的指数数据文件

    Args:
        uploaded_file: Streamlit 上传的文件对象
        start_date: 起始日期，格式 'YYYYMMDD'

    Returns:
        DataFrame: 包含 date, close 列
    """
    try:
        # 根据文件类型读取
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            raise Exception("不支持的文件格式，请上传 .xlsx、.xls 或 .csv 文件")

        # 标准化列名
        df = df.rename(columns={
            '交易日期': 'date',
            '收盘价': 'close',
            '开盘点位': 'open',
            '最高点位': 'high',
            '最低点位': 'low'
        })

        # 检查必要列
        if 'date' not in df.columns:
            # 尝试找日期列
            for col in df.columns:
                if '日期' in str(col) or 'date' in str(col).lower():
                    df = df.rename(columns={col: 'date'})
                    break
        if 'close' not in df.columns:
            # 尝试找收盘价列
            for col in df.columns:
                if '收盘' in str(col) or 'close' in str(col).lower():
                    df = df.rename(columns={col: 'close'})
                    break

        if 'date' not in df.columns or 'close' not in df.columns:
            raise Exception("文件缺少必要列：需要包含'交易日期'和'收盘价'列")

        # 处理收盘价：可能是带逗号的字符串
        if df['close'].dtype == object:
            df['close'] = df['close'].astype(str).str.replace(',', '')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # 确保日期格式
        df['date'] = pd.to_datetime(df['date'])

        # 按日期筛选
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        return df

    except Exception as e:
        raise Exception(f"解析上传文件失败: {str(e)}")


def validate_uploaded_index_file(uploaded_file) -> Tuple[bool, str, pd.DataFrame]:
    """
    验证上传的指数数据文件格式

    Args:
        uploaded_file: Streamlit 上传的文件对象

    Returns:
        Tuple: (是否有效, 错误/成功消息, 数据预览DataFrame)
    """
    try:
        df = parse_uploaded_index_file(uploaded_file)

        # 检查数据是否为空
        if df.empty:
            return False, "上传的文件数据为空", pd.DataFrame()

        # 检查必要列
        if 'date' not in df.columns or 'close' not in df.columns:
            return False, "缺少必要列（date, close）", pd.DataFrame()

        # 检查数据有效性
        valid_rows = df['close'].notna().sum()
        total_rows = len(df)

        msg = f"✅ 文件验证成功！共 {total_rows} 行数据，{valid_rows} 行有效收盘价"
        return True, msg, df.head(5)

    except Exception as e:
        return False, f"❌ 文件验证失败: {str(e)}", pd.DataFrame()


def fetch_index_daily_local(symbol: str, start_date: str = None,
                            data_dir: str = '.') -> pd.DataFrame:
    """
    从本地 Excel 文件获取道琼斯指数数据

    Args:
        symbol: 指数代码，如 'DJCHINA.GI'
        start_date: 起始日期，格式 'YYYYMMDD'
        data_dir: 数据文件目录

    Returns:
        DataFrame: 包含 date, close 列
    """
    try:
        file_path = find_local_index_file(symbol, data_dir)
        if file_path is None:
            raise Exception(f"未找到指数 {symbol} 的本地数据文件")

        df = pd.read_excel(file_path)
        # 标准化列名
        df = df.rename(columns={
            '交易日期': 'date',
            '收盘价': 'close',
            '开盘点位': 'open',
            '最高点位': 'high',
            '最低点位': 'low'
        })
        # 确保日期格式
        df['date'] = pd.to_datetime(df['date'])
        # 按日期筛选
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        raise Exception(f"读取本地指数 {symbol} 数据失败: {str(e)}")


def fetch_all_index_data(start_date: str = None,
                         data_dir: str = '.') -> Dict[str, pd.DataFrame]:
    """
    获取所有指数的历史数据

    Args:
        start_date: 起始日期，格式 'YYYYMMDD'，默认使用 INDEX_START_DATE
        data_dir: 本地数据文件目录

    Returns:
        Dict: {指数名称: DataFrame}，获取失败的指数值为 None
    """
    if start_date is None:
        start_date = INDEX_START_DATE

    result = {}
    errors = {}

    # 获取 akshare 数据源的指数
    for name, symbol in INDEX_SYMBOLS_AKSHARE.items():
        try:
            df = fetch_index_daily_akshare(symbol, start_date)
            result[name] = df
        except Exception as e:
            result[name] = None
            errors[name] = str(e)

    # 获取本地文件的道琼斯指数
    for name, symbol in INDEX_SYMBOLS_LOCAL.items():
        try:
            df = fetch_index_daily_local(symbol, start_date, data_dir)
            result[name] = df
        except Exception as e:
            result[name] = None
            errors[name] = str(e)

    return result, errors


# =====================
# CPI 数据获取与处理函数
# =====================

def fetch_cpi_yearly() -> pd.DataFrame:
    """
    获取中国年度 CPI 数据

    Returns:
        DataFrame: 包含年份和 CPI 值的数据
    """
    try:
        df = ak.macro_china_cpi_yearly()
        return df
    except Exception as e:
        raise Exception(f"获取CPI数据失败: {str(e)}")


def calculate_daily_cpi_growth(cpi_yearly_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每年的日复利增长率

    注意：ak.macro_china_cpi_yearly() 返回的是 CPI 同比增长率（百分比形式）
    例如：1995年1月的值为 25.5，表示相对于1994年1月增长了25.5%

    计算公式：
    (1 + CPI_daily)^250 = 1 + (annual_rate / 100)
    CPI_daily = (1 + annual_rate / 100)^(1/250) - 1

    Args:
        cpi_yearly_df: 年度 CPI 同比增长率 DataFrame

    Returns:
        DataFrame: 包含年份、年度增长率(%)、日复利增长率
    """
    try:
        df = cpi_yearly_df.copy()

        # 识别列名（akshare 返回的列名可能是 '日期' 或 '统计时间' 等）
        date_col = None
        value_col = None
        for col in df.columns:
            if '日期' in col or '时间' in col or '年' in col:
                date_col = col
            elif '今值' in col or 'CPI' in col.upper() or '值' in col or '同比' in col:
                value_col = col

        if date_col is None:
            date_col = df.columns[0]
        if value_col is None:
            value_col = df.columns[1]

        # 提取年份（数据中的年份表示该年相对于上一年的增长率）
        df['year'] = pd.to_datetime(df[date_col]).dt.year

        # CPI 同比增长率（百分比形式，如 25.5 表示 25.5%）
        df['annual_rate_pct'] = pd.to_numeric(df[value_col], errors='coerce')

        # 按年份排序并去重（取每年的数据）
        df = df.sort_values('year').drop_duplicates(subset='year', keep='first').reset_index(drop=True)

        # 过滤无效数据
        df = df[df['annual_rate_pct'].notna()].copy()

        # 将百分比转换为小数（25.5% -> 0.255）
        df['annual_rate_decimal'] = df['annual_rate_pct'] / 100.0

        # 计算日复利增长率
        # 公式: CPI_daily = (1 + annual_rate)^(1/250) - 1
        # 注意：年度增长率可能为负，需要确保 1 + annual_rate > 0
        df['growth_factor'] = 1 + df['annual_rate_decimal']

        # 过滤掉增长因子为负或零的异常数据
        df = df[df['growth_factor'] > 0].copy()

        # 计算日复利增长率
        df['daily_rate'] = df['growth_factor'] ** (1 / TRADING_DAYS_PER_YEAR) - 1

        # 返回结果
        # year: 该年份（如1995年的数据表示1994->1995这一年的增长率）
        # annual_rate_pct: 年度CPI同比增长率（百分比形式）
        # daily_rate: 对应的日复利增长率
        result = df[['year', 'annual_rate_pct', 'daily_rate']].copy()

        return result.reset_index(drop=True)

    except Exception as e:
        raise Exception(f"计算CPI日增长率失败: {str(e)}")


def build_daily_cpi_series(cpi_growth_df: pd.DataFrame,
                           trading_dates: pd.Series,
                           base_cpi: float = None) -> pd.DataFrame:
    """
    构建每日累计 CPI 序列

    Args:
        cpi_growth_df: 日增长率 DataFrame（包含 year, daily_rate 列）
        trading_dates: 交易日日期序列
        base_cpi: 基准 CPI 值，默认使用配置中的值

    Returns:
        DataFrame: 包含 date, cpi 列的每日 CPI 数据
    """
    if base_cpi is None:
        base_cpi = BASE_CPI_VALUE

    try:
        # 创建年份到日增长率的映射
        rate_map = dict(zip(cpi_growth_df['year'], cpi_growth_df['daily_rate']))

        # 获取最早和最晚年份的增长率，用于外推
        min_year = cpi_growth_df['year'].min()
        max_year = cpi_growth_df['year'].max()
        first_rate = cpi_growth_df.loc[cpi_growth_df['year'] == min_year, 'daily_rate'].values[0]
        last_rate = cpi_growth_df.loc[cpi_growth_df['year'] == max_year, 'daily_rate'].values[0]

        # 准备日期序列
        dates = pd.to_datetime(trading_dates).sort_values().reset_index(drop=True)

        # 计算每日累计 CPI
        cpi_values = []
        current_cpi = base_cpi

        for i, date in enumerate(dates):
            year = date.year

            # 获取该年的日增长率
            if year in rate_map:
                daily_rate = rate_map[year]
            elif year < min_year:
                daily_rate = first_rate  # 使用最早年份的增长率
            else:
                daily_rate = last_rate   # 使用最晚年份的增长率

            if i == 0:
                cpi_values.append(current_cpi)
            else:
                current_cpi = current_cpi * (1 + daily_rate)
                cpi_values.append(current_cpi)

        result = pd.DataFrame({
            'date': dates,
            'cpi': cpi_values
        })

        return result

    except Exception as e:
        raise Exception(f"构建每日CPI序列失败: {str(e)}")


def adjust_index_for_inflation(index_df: pd.DataFrame,
                                cpi_daily_df: pd.DataFrame,
                                base_cpi: float = None) -> pd.DataFrame:
    """
    对指数进行通胀调整和对数化处理

    Args:
        index_df: 指数数据（包含 date, close 列）
        cpi_daily_df: 每日 CPI 数据（包含 date, cpi 列）
        base_cpi: 基准 CPI 值

    Returns:
        DataFrame: 包含 date, close, cpi, real_close, log_close 列
    """
    if base_cpi is None:
        base_cpi = BASE_CPI_VALUE

    try:
        # 确保日期格式一致
        index_df = index_df.copy()
        cpi_daily_df = cpi_daily_df.copy()
        index_df['date'] = pd.to_datetime(index_df['date'])
        cpi_daily_df['date'] = pd.to_datetime(cpi_daily_df['date'])

        # 合并数据
        merged = pd.merge(index_df, cpi_daily_df, on='date', how='inner')

        # 计算实际指数值（去除通胀）
        merged['real_close'] = merged['close'] / merged['cpi'] # * base_cpi

        # 对数化处理
        merged['log_close'] = np.log(merged['real_close'])

        # 清理无效值
        merged = merged.dropna(subset=['real_close', 'log_close'])
        merged = merged[merged['real_close'] > 0]

        return merged[['date', 'close', 'cpi', 'real_close', 'log_close']]

    except Exception as e:
        raise Exception(f"通胀调整失败: {str(e)}")


def calculate_ols_regression(df: pd.DataFrame) -> dict:
    """
    对对数化指数进行 OLS 线性回归

    Args:
        df: 包含 date 和 log_close 列的 DataFrame

    Returns:
        dict: {'slope': 斜率, 'intercept': 截距, 'std': 残差标准差,
               'r_squared': R², 'days': 天数数组, 'fitted': 拟合值}
    """
    try:
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # 将日期转换为从起始日开始的天数
        start_date = df['date'].iloc[0]
        df['days'] = (df['date'] - start_date).dt.days

        # 准备数据
        x = df['days'].values
        y = df['log_close'].values

        # OLS 回归
        slope, intercept = np.polyfit(x, y, 1)

        # 计算拟合值
        fitted = slope * x + intercept

        # 计算残差和标准差
        residuals = y - fitted
        std = np.std(residuals)

        # 计算 R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 计算年化收益率（基于斜率）
        # 斜率是每日对数收益率，年化 = slope * 250
        annual_return = slope * TRADING_DAYS_PER_YEAR

        # 计算年化波动率
        annual_volatility = std * np.sqrt(TRADING_DAYS_PER_YEAR)

        return {
            'slope': slope,
            'intercept': intercept,
            'std': std,
            'r_squared': r_squared,
            'days': x,
            'fitted': fitted,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility
        }

    except Exception as e:
        raise Exception(f"OLS回归计算失败: {str(e)}")


def process_all_indices_inflation(index_data: Dict[str, pd.DataFrame],
                                   cpi_yearly_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """
    对所有指数进行通胀调整和回归分析

    Args:
        index_data: {指数名称: DataFrame} 字典
        cpi_yearly_df: 年度 CPI 数据

    Returns:
        Tuple: (adjusted_data, regression_results, errors)
    """
    adjusted_data = {}
    regression_results = {}
    errors = {}

    try:
        # 计算 CPI 日增长率
        cpi_growth = calculate_daily_cpi_growth(cpi_yearly_df)

        # 收集所有指数的交易日期
        all_dates = set()
        for name, df in index_data.items():
            if df is not None and not df.empty:
                all_dates.update(pd.to_datetime(df['date']).tolist())

        if not all_dates:
            raise Exception("没有有效的指数数据")

        # 构建每日 CPI 序列
        all_dates_series = pd.Series(sorted(list(all_dates)))
        cpi_daily = build_daily_cpi_series(cpi_growth, all_dates_series)

        # 处理每个指数
        for name, df in index_data.items():
            if df is None or df.empty:
                errors[name] = "无数据"
                continue

            try:
                # 通胀调整
                adjusted_df = adjust_index_for_inflation(df, cpi_daily)
                adjusted_data[name] = adjusted_df

                # OLS 回归
                if len(adjusted_df) > 10:  # 确保有足够的数据点
                    regression = calculate_ols_regression(adjusted_df)
                    regression_results[name] = regression
                else:
                    errors[name] = "数据点不足，无法进行回归分析"

            except Exception as e:
                errors[name] = str(e)

    except Exception as e:
        errors['系统错误'] = str(e)

    return adjusted_data, regression_results, errors


# =====================
# 指数对比分析函数
# =====================

def fetch_dji_index(start_date: str = None, data_dir: str = '.') -> pd.DataFrame:
    """
    获取道琼斯工业指数数据（从本地Excel文件）

    Args:
        start_date: 起始日期，格式 'YYYYMMDD'
        data_dir: 数据文件目录

    Returns:
        DataFrame: 包含 date, close 列
    """
    if start_date is None:
        start_date = COMPARISON_START_DATE.replace('-', '')

    try:
        file_path = find_local_index_file(DJI_SYMBOL, data_dir)
        if not file_path:
            raise Exception(f"未找到道琼斯工业指数文件: {DJI_SYMBOL}")

        df = pd.read_excel(file_path)

        # 识别日期和收盘价列
        date_col = None
        close_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if '日期' in col_lower or 'date' in col_lower:
                date_col = col
            elif '收盘' in col_lower or 'close' in col_lower:
                close_col = col

        if date_col is None:
            date_col = df.columns[0]
        if close_col is None:
            # 尝试找收盘价相关列
            for col in df.columns:
                if '收盘' in str(col) or 'close' in str(col).lower():
                    close_col = col
                    break
            if close_col is None:
                close_col = df.columns[4] if len(df.columns) > 4 else df.columns[1]

        # 处理收盘价：可能是带逗号的字符串（如 "47,416.91"）
        close_values = df[close_col]
        if close_values.dtype == object:
            # 如果是字符串类型，去掉逗号再转换
            close_values = close_values.astype(str).str.replace(',', '')
        close_numeric = pd.to_numeric(close_values, errors='coerce')

        result = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'close': close_numeric
        })

        # 过滤日期范围
        start_dt = pd.to_datetime(start_date)
        result = result[result['date'] >= start_dt]
        result = result.dropna().sort_values('date').reset_index(drop=True)

        if result.empty:
            raise Exception(f"过滤后无有效数据（起始日期: {start_date}）")

        return result

    except Exception as e:
        raise Exception(f"获取道琼斯工业指数失败: {str(e)}")


def normalize_index_pair(base_index_df: pd.DataFrame,
                          compare_index_df: pd.DataFrame,
                          start_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, float, str]:
    """
    将对比指数归一化到基准指数的起始点位

    Args:
        base_index_df: 基准指数数据（中国指数）
        compare_index_df: 对比指数数据（道琼斯指数）
        start_date: 起始日期，格式 'YYYY-MM-DD'，如果为 None 则使用配置中的默认值

    Returns:
        Tuple: (基准指数df, 归一化后的对比指数df, 归一化因子, 实际使用的起始日期)
    """
    if start_date is None:
        start_date = COMPARISON_START_DATE

    try:
        base_df = base_index_df.copy()
        compare_df = compare_index_df.copy()

        base_df['date'] = pd.to_datetime(base_df['date'])
        compare_df['date'] = pd.to_datetime(compare_df['date'])

        start_dt = pd.to_datetime(start_date)

        # 找到两个指数的最早可用日期
        base_min_date = base_df['date'].min()
        compare_min_date = compare_df['date'].min()

        # 使用较晚的起始日期作为共同起始点
        common_start = max(base_min_date, compare_min_date, start_dt)

        # 找到起始日期或最接近的交易日
        base_start = base_df[base_df['date'] >= common_start].head(1)
        compare_start = compare_df[compare_df['date'] >= common_start].head(1)

        if base_start.empty:
            raise Exception(f"基准指数在 {common_start.strftime('%Y-%m-%d')} 后无数据")
        if compare_start.empty:
            raise Exception(f"对比指数在 {common_start.strftime('%Y-%m-%d')} 后无数据")

        # 使用实际的共同起始日期
        actual_start_date = max(base_start['date'].values[0], compare_start['date'].values[0])
        actual_start_date = pd.to_datetime(actual_start_date)

        # 重新获取起始点数据
        base_start = base_df[base_df['date'] >= actual_start_date].head(1)
        compare_start = compare_df[compare_df['date'] >= actual_start_date].head(1)

        base_start_value = base_start['close'].values[0]
        compare_start_value = compare_start['close'].values[0]

        # 计算归一化因子
        normalize_factor = compare_start_value / base_start_value

        # 归一化对比指数
        compare_df['close_normalized'] = compare_df['close'] / normalize_factor

        # 过滤日期范围
        base_df = base_df[base_df['date'] >= actual_start_date].reset_index(drop=True)
        compare_df = compare_df[compare_df['date'] >= actual_start_date].reset_index(drop=True)

        return base_df, compare_df, normalize_factor, actual_start_date.strftime('%Y-%m-%d')

    except Exception as e:
        raise Exception(f"归一化处理失败: {str(e)}")


def prepare_all_index_comparisons(china_indices: Dict[str, pd.DataFrame],
                                   dji_df: pd.DataFrame,
                                   start_date: str = None) -> Tuple[Dict, Dict]:
    """
    准备所有中国指数与道琼斯工业指数的对比数据

    Args:
        china_indices: {指数名称: DataFrame} 字典
        dji_df: 道琼斯工业指数数据
        start_date: 归一化起始日期

    Returns:
        Tuple: (comparison_data, errors)
        comparison_data: {指数名称: {'base': df, 'compare': df, 'factor': float, 'start_date': str}}
    """
    if start_date is None:
        start_date = COMPARISON_START_DATE

    comparison_data = {}
    errors = {}

    for name, base_df in china_indices.items():
        if base_df is None or base_df.empty:
            errors[name] = "无数据"
            continue

        try:
            base_normalized, compare_normalized, factor, actual_start = normalize_index_pair(
                base_df, dji_df, start_date
            )
            comparison_data[name] = {
                'base': base_normalized,
                'compare': compare_normalized,
                'factor': factor,
                'start_date': actual_start,
                'base_name': name,
                'compare_name': DJI_NAME
            }
        except Exception as e:
            errors[name] = str(e)

    return comparison_data, errors


# =====================
# 主要股票基本面数据获取函数（多年年报数据）
# =====================

# 年报日期列表（从2020年开始）
ANNUAL_REPORT_DATES = ['20201231', '20211231', '20221231', '20231231']


def get_annual_report_dates():
    """获取从2020年到最近年报的日期列表"""
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month

    # 如果当前月份>=5月，可以使用上一年的年报
    latest_year = current_year - 1 if current_month >= 5 else current_year - 2

    years = list(range(2020, latest_year + 1))
    return [f"{year}1231" for year in years]


def fetch_stock_financial_abstract_ths_yearly(symbol: str) -> pd.DataFrame:
    """
    获取同花顺财务摘要数据（多年年报）

    Args:
        symbol: 股票代码，如 '002594'

    Returns:
        DataFrame: 包含多年的营业总收入、每股净资产、销售净利润、销售毛利率、净资产收益率等
    """
    try:
        df = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
        if df.empty:
            return pd.DataFrame()

        # 筛选年报数据（报告期以12-31结尾）
        df['报告期'] = df['报告期'].astype(str)
        annual_df = df[df['报告期'].str.endswith('12-31')].copy()

        # 筛选2020年及以后的数据
        annual_df = annual_df[annual_df['报告期'] >= '2020-12-31']

        # 选择需要的列
        columns_map = {
            '报告期': '报告期',
            '营业总收入': '营业总收入',
            '每股净资产': '每股净资产',
            '销售净利率': '销售净利率',
            '销售毛利率': '销售毛利率',
        }

        # 处理净资产收益率（可能有不同列名）
        if '净资产收益率-摊薄' in annual_df.columns:
            columns_map['净资产收益率-摊薄'] = '净资产收益率'
        elif '净资产收益率' in annual_df.columns:
            columns_map['净资产收益率'] = '净资产收益率'

        # 只保留存在的列
        available_cols = [col for col in columns_map.keys() if col in annual_df.columns]
        result_df = annual_df[available_cols].copy()
        result_df.columns = [columns_map[col] for col in available_cols]

        return result_df
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})


def fetch_stock_financial_abstract_yearly(symbol: str) -> pd.DataFrame:
    """
    获取新浪财务摘要数据（多年年报）

    数据结构: 每行是一个指标，报告期作为列名（如 '20201231', '20211231' 等）
    - 第一行: 归母净利润
    - 最后一行: 应付账款周转率

    Args:
        symbol: 股票代码，如 '002594'

    Returns:
        DataFrame: 包含多年的归母净利润、应付账款周转率等
    """
    try:
        df = ak.stock_financial_abstract(symbol=symbol)
        if df.empty:
            return pd.DataFrame()

        # 获取年报列（以1231结尾的列名）
        report_dates = get_annual_report_dates()
        available_dates = [col for col in df.columns if col in report_dates]

        if not available_dates:
            return pd.DataFrame()

        # 需要的指标（与接口返回的指标名称匹配）
        target_indicators = {
            '归母净利润': '归母净利润',
            '营业总收入': '营业总收入',
            '营业成本': '营业成本',
            '净利润': '净利润',
            '应付账款周转率': '应付账款周转率',
            '权益乘数': '权益乘数',  # 新增：权益乘数
        }

        # 构建结果数据 - 转置：每行一个年报
        all_data = []
        for report_date in available_dates:
            row_data = {'报告期': f"{report_date[:4]}-12-31"}

            for _, row in df.iterrows():
                indicator_name = row.get('指标', '')
                if indicator_name in target_indicators:
                    col_name = target_indicators[indicator_name]
                    value = row.get(report_date)
                    row_data[col_name] = value

            all_data.append(row_data)

        return pd.DataFrame(all_data)
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})


def fetch_stock_balance_sheet_yearly(symbol: str) -> pd.DataFrame:
    """
    获取资产负债表数据（多年年报）

    数据结构: 返回指定日期所有股票的资产负债表，每行是一只股票
    列名示例: '股票代码', '资产-应收账款', '资产-存货', '负债-应付账款' 等

    Args:
        symbol: 股票代码，如 '002594'

    Returns:
        DataFrame: 包含多年的存货、应付账款、应收账款
    """
    all_data = []
    report_dates = get_annual_report_dates()

    for report_date in report_dates:
        try:
            df = ak.stock_zcfz_em(date=report_date)
            if df.empty:
                continue

            # 筛选指定股票
            stock_data = df[df['股票代码'] == symbol]
            if stock_data.empty:
                continue

            row = stock_data.iloc[0]
            year = report_date[:4]

            # 正确的列名（带"资产-"或"负债-"前缀）
            all_data.append({
                '报告期': f"{year}-12-31",
                '存货': row.get('资产-存货', None),
                '应付账款': row.get('负债-应付账款', None),
                '应收账款': row.get('资产-应收账款', None),
            })
        except Exception:
            continue

    return pd.DataFrame(all_data)


def fetch_stock_financial_indicator_yearly(symbol: str) -> pd.DataFrame:
    """
    获取财务分析指标数据（多年年报）

    数据结构: 每行对应一个报告期，列为各指标
    列名示例: '日期', '主营业务成本率(%)', '存货周转率(次)', '存货周转天数(天)', '总资产(元)'

    Args:
        symbol: 股票代码，如 '002594'

    Returns:
        DataFrame: 包含多年的主营业务成本率、存货周转率、存货周转天数、总资产、平均存货周期
    """
    try:
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2020")
        if df.empty:
            return pd.DataFrame()

        # 筛选年报数据 - 日期列是datetime.date对象
        if '日期' in df.columns:
            # 转换为字符串格式 YYYY-MM-DD
            df['日期_str'] = df['日期'].apply(lambda x: str(x) if x else '')
            annual_df = df[df['日期_str'].str.endswith('12-31')].copy()
        else:
            annual_df = df.copy()

        if annual_df.empty:
            return pd.DataFrame()

        # 列名映射（处理带单位后缀的列名）
        column_mapping = {
            '日期': '报告期',
            '主营业务成本率(%)': '主营业务成本率',
            '存货周转率(次)': '存货周转率',
            '存货周转天数(天)': '存货周转天数',
            '总资产(元)': '总资产',
        }

        # 选择需要的列
        result_data = []
        for _, row in annual_df.iterrows():
            row_data = {}
            # 报告期
            if '日期' in annual_df.columns:
                date_val = row['日期']
                row_data['报告期'] = str(date_val) if date_val else ''

            # 其他指标
            for orig_col, new_col in column_mapping.items():
                if orig_col in annual_df.columns and orig_col != '日期':
                    row_data[new_col] = row.get(orig_col)

            result_data.append(row_data)

        result_df = pd.DataFrame(result_data)

        # 计算平均存货周期 = 365 / 存货周转率
        if '存货周转率' in result_df.columns:
            result_df['平均存货周期'] = result_df['存货周转率'].apply(
                lambda x: 365 / float(x) if pd.notna(x) and float(x) > 0 else None
            )

        return result_df
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})


def fetch_stock_financial_indicator_em_yearly(symbol: str) -> pd.DataFrame:
    """
    获取东方财富财务指标数据（多年年报）

    数据结构: 每行对应一个报告期，列名为英文
    - REPORT_DATE: 报告日期 (datetime格式 如 2024-12-31 00:00:00)
    - YSZKZZL: 应收账款周转率
    - ACCOUNTS_PAYABLE_TR: 应付账款周转率

    Args:
        symbol: 股票代码，如 '002594'

    Returns:
        DataFrame: 包含多年的应收账款周转率、应付账款周转率
    """
    try:
        # 添加交易所后缀
        if '.' not in symbol:
            suffix = STOCK_EXCHANGE_SUFFIX.get(symbol, '.SZ')
            symbol_with_suffix = symbol + suffix
        else:
            symbol_with_suffix = symbol

        df = ak.stock_financial_analysis_indicator_em(symbol=symbol_with_suffix)
        if df.empty:
            return pd.DataFrame()

        # 筛选年报数据（REPORT_DATE列是datetime格式）
        if 'REPORT_DATE' in df.columns:
            # 转换为字符串格式
            df['报告期_str'] = df['REPORT_DATE'].apply(
                lambda x: str(x)[:10] if pd.notna(x) else ''
            )
            # 筛选年报 (12-31结尾) 且 >= 2020
            annual_df = df[
                (df['报告期_str'].str.endswith('12-31')) &
                (df['报告期_str'] >= '2020')
            ].copy()
        else:
            annual_df = df.copy()

        if annual_df.empty:
            return pd.DataFrame()

        # 列名映射（英文 -> 中文）
        result_data = []
        for _, row in annual_df.iterrows():
            row_data = {
                '报告期': row.get('报告期_str', ''),
                '应收账款周转率': row.get('YSZKZZL'),
                '应付账款周转率': row.get('ACCOUNTS_PAYABLE_TR'),
            }
            result_data.append(row_data)

        return pd.DataFrame(result_data)
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})


def _fetch_single_stock_data(name: str, symbol: str) -> Tuple[pd.DataFrame, Dict]:
    """
    获取单只股票的基本面数据（用于并行请求）

    Args:
        name: 股票名称
        symbol: 股票代码

    Returns:
        Tuple: (股票DataFrame, 错误信息Dict)
    """
    errors = {}

    try:
        # 使用 ThreadPoolExecutor 并行获取5个数据源
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures_map = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures_map['ths'] = executor.submit(fetch_stock_financial_abstract_ths_yearly, symbol)
            futures_map['sina'] = executor.submit(fetch_stock_financial_abstract_yearly, symbol)
            futures_map['balance'] = executor.submit(fetch_stock_balance_sheet_yearly, symbol)
            futures_map['indicator'] = executor.submit(fetch_stock_financial_indicator_yearly, symbol)
            futures_map['em'] = executor.submit(fetch_stock_financial_indicator_em_yearly, symbol)

        # 收集结果
        results = {}
        for key, future in futures_map.items():
            try:
                results[key] = future.result(timeout=30)
            except Exception as e:
                errors[f"{name}_{key}"] = str(e)
                results[key] = pd.DataFrame()

        # 处理结果
        ths_df = results.get('ths', pd.DataFrame())
        sina_df = results.get('sina', pd.DataFrame())
        balance_df = results.get('balance', pd.DataFrame())
        indicator_df = results.get('indicator', pd.DataFrame())
        em_df = results.get('em', pd.DataFrame())

        # 检查错误
        if not ths_df.empty and 'error' in ths_df.columns:
            errors[f"{name}_ths"] = ths_df['error'].iloc[0]
            ths_df = pd.DataFrame()
        if not sina_df.empty and 'error' in sina_df.columns:
            errors[f"{name}_sina"] = sina_df['error'].iloc[0]
            sina_df = pd.DataFrame()
        if not indicator_df.empty and 'error' in indicator_df.columns:
            errors[f"{name}_indicator"] = indicator_df['error'].iloc[0]
            indicator_df = pd.DataFrame()
        if not em_df.empty and 'error' in em_df.columns:
            errors[f"{name}_em"] = em_df['error'].iloc[0]
            em_df = pd.DataFrame()

        # 合并数据 - 以同花顺数据为基础，按报告期合并
        if not ths_df.empty:
            merged_df = ths_df.copy()
            merged_df['报告期'] = merged_df['报告期'].astype(str).str.replace('-', '')

            if not sina_df.empty:
                sina_df['报告期'] = sina_df['报告期'].astype(str).str.replace('-', '')
                merged_df = pd.merge(merged_df, sina_df, on='报告期', how='outer')

            if not balance_df.empty:
                balance_df['报告期'] = balance_df['报告期'].astype(str).str.replace('-', '')
                merged_df = pd.merge(merged_df, balance_df, on='报告期', how='outer')

            if not indicator_df.empty:
                indicator_df['报告期'] = indicator_df['报告期'].astype(str).str.replace('-', '')
                merged_df = pd.merge(merged_df, indicator_df, on='报告期', how='outer')

            if not em_df.empty:
                em_df['报告期'] = em_df['报告期'].astype(str).str.replace('-', '')
                merged_df = pd.merge(merged_df, em_df, on='报告期', how='outer')

            merged_df['股票名称'] = name
            merged_df['股票代码'] = symbol
            merged_df['报告期'] = merged_df['报告期'].apply(
                lambda x: f"{str(x)[:4]}年年报" if len(str(x)) >= 8 else str(x)
            )
            return merged_df, errors
        elif not sina_df.empty:
            sina_df['股票名称'] = name
            sina_df['股票代码'] = symbol
            sina_df['报告期'] = sina_df['报告期'].apply(
                lambda x: f"{str(x)[:4]}年年报" if len(str(x)) >= 8 else str(x)
            )
            return sina_df, errors

    except Exception as e:
        errors[name] = str(e)

    return pd.DataFrame(), errors


def fetch_all_featured_stocks_data(progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
    """
    获取所有重点关注股票的多年年报基本面数据（并行优化版本）

    Args:
        progress_callback: 进度回调函数，接收 (当前进度, 总数, 股票名称)

    Returns:
        Tuple: (汇总DataFrame, 错误信息Dict)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_data = []
    errors = {}
    stocks_list = list(FEATURED_STOCKS.items())
    total = len(stocks_list)

    # 并行获取所有股票数据
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_stock = {
            executor.submit(_fetch_single_stock_data, name, symbol): (name, symbol)
            for name, symbol in stocks_list
        }

        completed = 0
        for future in as_completed(future_to_stock):
            name, symbol = future_to_stock[future]
            completed += 1

            if progress_callback:
                progress_callback(completed, total, name)

            try:
                stock_df, stock_errors = future.result(timeout=60)
                errors.update(stock_errors)
                if not stock_df.empty:
                    all_data.append(stock_df)
            except Exception as e:
                errors[name] = str(e)

    # 合并所有股票数据
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        first_cols = ['股票名称', '股票代码', '报告期']
        other_cols = [col for col in final_df.columns if col not in first_cols]
        final_df = final_df[first_cols + other_cols]
        final_df = final_df.sort_values(['股票名称', '报告期'], ascending=[True, False])
    else:
        final_df = pd.DataFrame()

    return final_df, errors


def load_prepared_stock_data(file_path: str = 'prepared_stock_data.csv') -> Tuple[pd.DataFrame, Dict]:
    """
    从本地 CSV 文件加载预先准备好的股票数据（用于快速首次加载）

    Args:
        file_path: CSV 文件路径

    Returns:
        Tuple: (DataFrame, 错误信息Dict)
    """
    import os

    errors = {}

    try:
        if not os.path.exists(file_path):
            return pd.DataFrame(), {'文件错误': f'预加载文件 {file_path} 不存在'}

        df = pd.read_csv(file_path, encoding='utf-8')

        # 验证必要的列
        required_cols = ['股票名称', '股票代码', '报告期']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), {'格式错误': f'缺少必要列: {missing_cols}'}

        # 按股票名称和报告期排序
        df = df.sort_values(['股票名称', '报告期'], ascending=[True, False])

        return df, errors

    except Exception as e:
        return pd.DataFrame(), {'读取错误': str(e)}


def load_prepared_us_stock_data(file_path: str = 'prepared_us_stock_data.csv') -> Tuple[pd.DataFrame, Dict]:
    """
    从本地 CSV 文件加载预先准备好的美股数据（用于快速首次加载）

    Args:
        file_path: CSV 文件路径

    Returns:
        Tuple: (DataFrame, 错误信息Dict)
    """
    import os

    errors = {}

    try:
        if not os.path.exists(file_path):
            # 美股预加载文件不存在时返回空，不报错
            return pd.DataFrame(), {}

        df = pd.read_csv(file_path, encoding='utf-8')

        # 验证必要的列
        required_cols = ['股票名称', '股票代码', '报告期', '存货周转天数', '应收账款周转天数', '经营周期']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), {'格式错误': f'缺少必要列: {missing_cols}'}

        df = df.sort_values(['股票名称', '报告期'], ascending=[True, False])

        return df, errors

    except Exception as e:
        return pd.DataFrame(), {'读取错误': str(e)}


# =====================
# 美股数据获取
# =====================

# 美股重点关注股票
US_FEATURED_STOCKS = {
    '特斯拉': 'TSLA',
    '丰田': 'TM'
}


def fetch_us_stock_operating_cycle(symbol: str, name: str = None) -> pd.DataFrame:
    """
    获取美股的经营周期数据（存货周转天数 + 应收账款周转天数）

    Args:
        symbol: 股票代码（如 'TSLA', 'TM'）
        name: 股票名称（可选）

    Returns:
        DataFrame: 包含报告期、存货周转天数、应收账款周转天数、经营周期
    """
    try:
        df = ak.stock_financial_us_analysis_indicator_em(symbol=symbol, indicator='年报')

        if df.empty:
            return pd.DataFrame({'error': [f'{symbol} 数据为空']})

        # 筛选需要的列
        required_cols = ['REPORT_DATE', 'REPORT_DATA_TYPE', 'INVENTORY_TDAYS', 'ACCOUNTS_RECE_TDAYS']
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame({'error': [f'{symbol} 缺少列: {col}']})

        result = df[required_cols].copy()

        # 使用 REPORT_DATA_TYPE 作为报告期（如 "2024年 年报"）
        result['报告期'] = result['REPORT_DATA_TYPE']

        # 筛选2020年及以后的数据
        result['REPORT_DATE'] = pd.to_datetime(result['REPORT_DATE'])
        result = result[result['REPORT_DATE'] >= '2020-01-01']

        # 重命名列
        result = result.rename(columns={
            'INVENTORY_TDAYS': '存货周转天数',
            'ACCOUNTS_RECE_TDAYS': '应收账款周转天数'
        })

        # 计算经营周期
        result['经营周期'] = result['存货周转天数'] + result['应收账款周转天数']

        # 添加股票信息
        if name:
            result['股票名称'] = name
        result['股票代码'] = symbol

        # 选择最终列并排序
        final_cols = ['股票名称', '股票代码', '报告期', '存货周转天数', '应收账款周转天数', '经营周期']
        if name:
            result = result[final_cols]
        else:
            result = result[['股票代码', '报告期', '存货周转天数', '应收账款周转天数', '经营周期']]

        result = result.sort_values('报告期', ascending=False).reset_index(drop=True)

        return result

    except Exception as e:
        return pd.DataFrame({'error': [f'{symbol} 获取失败: {str(e)}']})


def fetch_all_us_stocks_data(progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
    """
    获取所有美股重点关注股票的经营周期数据（并行优化版本）

    Args:
        progress_callback: 进度回调函数，接收 (当前进度, 总数, 股票名称)

    Returns:
        Tuple: (汇总DataFrame, 错误信息Dict)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_data = []
    errors = {}
    stocks_list = list(US_FEATURED_STOCKS.items())
    total = len(stocks_list)

    # 并行获取所有美股数据
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_stock = {
            executor.submit(fetch_us_stock_operating_cycle, symbol, name): (name, symbol)
            for name, symbol in stocks_list
        }

        completed = 0
        for future in as_completed(future_to_stock):
            name, _ = future_to_stock[future]
            completed += 1

            if progress_callback:
                progress_callback(completed, total, name)

            try:
                df = future.result(timeout=30)
                if not df.empty and 'error' in df.columns:
                    errors[name] = df['error'].iloc[0]
                elif not df.empty:
                    all_data.append(df)
            except Exception as e:
                errors[name] = str(e)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values(['股票名称', '报告期'], ascending=[True, False])
    else:
        final_df = pd.DataFrame()

    return final_df, errors
