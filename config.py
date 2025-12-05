"""
配置模块
集中管理应用的默认参数和配置项
"""

from datetime import datetime, timedelta
import os

# =====================
# 日期相关配置
# =====================

# 获取当前日期
def get_today() -> str:
    """获取今天日期，格式 YYYYMMDD"""
    return datetime.now().strftime('%Y%m%d')

def get_default_report_date() -> str:
    """
    获取默认报告期日期
    根据当前月份自动推算最近的季报日期
    """
    now = datetime.now()
    year = now.year
    month = now.month

    # 根据月份确定最近的报告期
    if month >= 11:  # 11-12月，使用三季报
        return f"{year}0930"
    elif month >= 9:  # 9-10月，使用中报
        return f"{year}0630"
    elif month >= 5:  # 5-8月，使用一季报
        return f"{year}0331"
    elif month >= 2:  # 2-4月，使用上年年报
        return f"{year-1}1231"
    else:  # 1月，使用上年三季报
        return f"{year-1}0930"

# =====================
# 行业分类关键词
# =====================

# 银行业关键词
BANK_KEYWORDS = ['银行']

# 其他金融行业关键词（不含银行）
FINANCE_KEYWORDS = ['证券', '保险', '券商', '基金', '信托', '金融']

# =====================
# 指数配置
# =====================

# 指数默认起始日期
INDEX_START_DATE = "19940103"

# 指数符号配置
# akshare 接口支持的指数（使用 stock_zh_index_daily）
INDEX_SYMBOLS_AKSHARE = {
    '上证综合指数': 'sh000001',
    '深证成份指数': 'sz399001',
    '深圳综指': 'sz399106',
}

# 本地 Excel 文件的道琼斯指数
# 文件名格式: {symbol}-行情统计-YYYYMMDD.xlsx
INDEX_SYMBOLS_LOCAL = {
    '道琼斯中国指数': 'DJCHINA.GI',
    '道琼斯上海指数': 'DJSH.GI',
    '道琼斯深圳指数': 'DJSZ.GI',
}

# 道琼斯工业指数（用于对比分析）
DJI_SYMBOL = 'DJI.GI'
DJI_NAME = '道琼斯工业指数'

# 指数对比分析配置
COMPARISON_START_DATE = "1994-01-03"  # 归一化起始日期

# 指数图表颜色
INDEX_COLORS = {
    '上证综合指数': '#e41a1c',
    '深证成份指数': '#377eb8',
    '深圳综指': '#17becf',
    '道琼斯中国指数': '#4daf4a',
    '道琼斯上海指数': '#984ea3',
    '道琼斯深圳指数': '#ff7f00',
    '道琼斯工业指数': '#a65628',
}

def find_local_index_file(symbol: str, data_dir: str = '.') -> str:
    """
    查找本地指数 Excel 文件

    Args:
        symbol: 指数代码，如 'DJCHINA.GI'
        data_dir: 数据目录

    Returns:
        文件路径，未找到则返回 None
    """
    for f in os.listdir(data_dir):
        if f.startswith(symbol) and f.endswith('.xlsx'):
            return os.path.join(data_dir, f)
    return None

# =====================
# CPI 与通胀调整配置
# =====================

# 交易日配置
TRADING_DAYS_PER_YEAR = 250  # 每年交易日天数

# CPI 基准配置
BASE_CPI_VALUE = 1  # 基准 CPI 值（1994年1月1日）
BASE_CPI_YEAR = 1994   # 基准年份

# 分布图配置
HISTOGRAM_BINS = 50     # 直方图分箱数量
CONFIDENCE_SIGMA = 1.0  # 标准差倍数（用于绘制边界线）

# =====================
# 图表样式配置
# =====================

# 饼图配色方案
PIE_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb']

# 页面标题
APP_TITLE = "A股市场数据可视化工具"
APP_ICON = "📈"

# =====================
# 数据展示配置
# =====================

# 净利润100强显示的列
PROFIT_TOP100_DISPLAY_COLS = [
    '股票代码', '股票简称', '每股收益', '营业总收入-营业总收入',
    '净利润-净利润', '净利润-同比增长', '每股净资产', '净资产收益率',
    '销售毛利率', '所处行业'
]

# 数值格式化
def format_large_number(num):
    """将大数字格式化为亿/万"""
    if num is None or pd.isna(num):
        return "N/A"
    if abs(num) >= 1e8:
        return f"{num/1e8:.2f}亿"
    elif abs(num) >= 1e4:
        return f"{num/1e4:.2f}万"
    else:
        return f"{num:.2f}"

# 需要导入 pandas 用于 isna 检查
import pandas as pd

# =====================
# 主要股票配置
# =====================

# 重点关注的股票列表
FEATURED_STOCKS = {
    '比亚迪': '002594',
    '美的集团': '000333',
    '海尔智家': '600690',  # 注：海尔智家代码是600690，不是600601
    '格力电器': '000651',
}

# 股票对应的交易所后缀（用于某些接口）
STOCK_EXCHANGE_SUFFIX = {
    '002594': '.SZ',
    '000333': '.SZ',
    '600690': '.SH',
    '000651': '.SZ',
}

