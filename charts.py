"""
图表模块
封装所有可视化逻辑
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib import font_manager
from typing import Dict
from config import PIE_COLORS, INDEX_COLORS, CONFIDENCE_SIGMA, COMPARISON_START_DATE
import os

# 设置 matplotlib 中文字体
# 优先使用项目自带的 SimHei 字体（用于云端部署）
FONT_PATH = os.path.join(os.path.dirname(__file__), 'fonts', 'SimHei.ttf')
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
else:
    # 本地开发时使用系统字体
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


def display_market_summary(sse_df: pd.DataFrame, szse_df: pd.DataFrame):
    """
    展示市场统计数据（上交所+深交所）

    Args:
        sse_df: 上交所统计数据
        szse_df: 深交所统计数据
    """
    st.subheader("上交所市场统计")

    # 提取关键指标用于展示
    if sse_df is not None and not sse_df.empty:
        # 使用 columns 布局展示关键数据
        col1, col2, col3, col4 = st.columns(4)

        # 尝试从数据中提取指标
        try:
            # 上交所数据结构：项目列 + 股票列 + 主板列 + 科创板列
            sse_dict = dict(zip(sse_df['项目'], sse_df['股票']))

            with col1:
                st.metric("上市公司", f"{sse_dict.get('上市公司', 'N/A')}")
            with col2:
                total_mv = sse_dict.get('总市值', 0)
                if isinstance(total_mv, (int, float)):
                    st.metric("总市值(亿)", f"{float(total_mv):.2f}亿")
                else:
                    st.metric("总市值(亿)", str(total_mv))
            with col3:
                st.metric("流通市值(亿)", f"{sse_dict.get('流通市值', 'N/A')}")
            with col4:
                st.metric("平均市盈率", f"{sse_dict.get('平均市盈率', 'N/A')}")
        except Exception:
            pass

        # 显示完整表格
        with st.expander("查看上交所完整数据", expanded=False):
            st.dataframe(sse_df, use_container_width=True)

    st.subheader("深交所市场统计")

    if szse_df is not None and not szse_df.empty:
        query_date = szse_df.attrs.get('query_date')
        actual_date = szse_df.attrs.get('actual_date')
        if query_date and actual_date and query_date != actual_date:
            st.caption(
                f"所选日期 {query_date} 暂无可用数据，已自动使用最近可用交易日 {actual_date}。"
            )

        # 提取股票行数据
        try:
            stock_row = szse_df[szse_df['证券类别'] == '股票']
            if not stock_row.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("股票数量", f"{stock_row['数量'].values[0]}")
                with col2:
                    mv = stock_row['总市值'].values[0]
                    st.metric("总市值", f"{mv/1e12:.2f}万亿" if mv else "N/A")
                with col3:
                    lv = stock_row['流通市值'].values[0]
                    st.metric("流通市值", f"{lv/1e12:.2f}万亿" if lv else "N/A")
                with col4:
                    vol = stock_row['成交金额'].values[0]
                    st.metric("成交金额", f"{vol/1e8:.2f}亿" if vol else "N/A")
        except Exception:
            pass

        with st.expander("查看深交所完整数据", expanded=False):
            st.dataframe(szse_df, use_container_width=True)


def display_profit_top100(df: pd.DataFrame):
    """
    展示净利润100强数据

    Args:
        df: 净利润100强数据
    """
    if df is None or df.empty:
        st.warning("暂无数据")
        return

    st.subheader("A股归母净利润100强")

    # 获取股票简称列表
    stock_names = df['股票简称'].head(100).tolist() if '股票简称' in df.columns else []

    # 补齐到100个（如果不足）
    while len(stock_names) < 100:
        stock_names.append('-')

    # 创建10x10表格
    st.markdown("**归母净利润100强股票名单**")

    # 构建10x10表格数据（竖向排列：第1列为1-10名，第2列为11-20名...）
    grid_data = {}
    for col_idx in range(10):
        col_values = []
        for row_idx in range(10):
            rank = col_idx * 10 + row_idx + 1
            if rank <= len(stock_names):
                # 只显示股票名称
                col_values.append(stock_names[rank - 1])
            else:
                col_values.append("-")
        grid_data[f"{col_idx * 10 + 1}-{col_idx * 10 + 10}"] = col_values

    grid_df = pd.DataFrame(grid_data)

    # 使用自定义CSS使表格更紧凑美观
    st.markdown("""
    <style>
    .top100-grid td {
        text-align: center !important;
        padding: 8px 4px !important;
        font-size: 13px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.dataframe(
        grid_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # 显示前10名详细信息
    st.markdown("---")
    st.markdown("**前10名详细信息**")
    top10 = df.head(10).copy()

    display_cols = ['股票代码', '股票简称', '净利润-净利润', '净利润-同比增长', '所处行业']
    available_cols = [c for c in display_cols if c in top10.columns]

    if available_cols:
        show_df = top10[available_cols].copy()
        # 格式化净利润列
        if '净利润-净利润' in show_df.columns:
            show_df['净利润(亿元)'] = show_df['净利润-净利润'].apply(
                lambda x: f"{x/1e8:.2f}" if pd.notna(x) else "N/A"
            )
            show_df = show_df.drop(columns=['净利润-净利润'])

        # 重命名列
        col_rename = {
            '股票代码': '代码',
            '股票简称': '名称',
            '净利润-同比增长': '同比增长(%)',
            '所处行业': '行业'
        }
        show_df = show_df.rename(columns=col_rename)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    # 完整数据可折叠
    with st.expander("查看完整100强详细数据", expanded=False):
        st.dataframe(df, use_container_width=True)

    # 提供下载
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载完整数据 (CSV)",
        data=csv,
        file_name="profit_top100.csv",
        mime="text/csv"
    )


def display_profit_distribution(result_df: pd.DataFrame, pie_data: dict):
    """
    展示利润分布饼图

    Args:
        result_df: 分布统计表
        pie_data: 饼图数据 {'labels': [], 'values': [], 'percentages': []}
    """
    st.subheader("🍰 净利润行业分布")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**分布统计：**")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    with col2:
        # 绘制饼图 - 使用柔和的配色方案
        pastel_colors = [
            '#FF6B6B',  # 珊瑚红
            '#F7DC6F',  # 金黄
            '#45B7D1',  # 天蓝
            '#96CEB4',  # 薄荷绿
            '#FFEAA7',  # 柠檬黄
            '#DDA0DD',  # 梅红
            '#98D8C8',  # 浅绿
            '#BB8FCE',  # 淡紫
            '#85C1E9',  # 浅蓝
            '#F8B500',  # 橙黄
            '#82E0AA',  # 嫩绿
        ]

        fig, ax = plt.subplots(figsize=(6, 7))
        wedges, texts, autotexts = ax.pie(
            pie_data['values'],
            autopct='%.1f%%',
            startangle=140,
            colors=pastel_colors[:len(pie_data['values'])],
            wedgeprops=dict(edgecolor='white', linewidth=1.5)
        )

        # 设置百分比文字样式
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        # 图例放在下方，往上移
        ax.legend(
            wedges,
            pie_data['labels'],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            fontsize=9
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def display_index_charts(index_data: Dict[str, pd.DataFrame],
                         errors: Dict[str, str] = None):
    """
    展示指数历史走势图

    Args:
        index_data: {指数名称: DataFrame} 字典
        errors: {指数名称: 错误信息} 字典
    """
    st.subheader("📈 股票指数历史走势")

    # 显示错误信息
    if errors:
        for name, err in errors.items():
            if err:
                st.warning(f"⚠️ {name}: {err}")

    # 过滤有效数据
    valid_data = {k: v for k, v in index_data.items() if v is not None and not v.empty}

    if not valid_data:
        st.error("未能获取任何指数数据")
        return

    # 创建选项卡：分别展示 or 合并展示
    view_mode = st.radio(
        "展示模式",
        ["分别展示", "合并展示"],
        horizontal=True
    )

    if view_mode == "分别展示":
        _display_index_separate(valid_data)
    else:
        _display_index_combined(valid_data)

    # 提供数据下载
    st.divider()
    st.markdown("**📥 下载指数数据**")
    cols = st.columns(len(valid_data))
    for i, (name, df) in enumerate(valid_data.items()):
        with cols[i]:
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label=f"{name}",
                data=csv,
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"download_{name}"
            )


def _display_index_separate(index_data: Dict[str, pd.DataFrame]):
    """分别展示每个指数的走势图"""

    # 分成两列展示
    index_names = list(index_data.keys())

    for i in range(0, len(index_names), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(index_names):
                name = index_names[idx]
                df = index_data[name]
                with col:
                    _plot_single_index(name, df)


def _format_chart_axis(ax, set_xlim=True):
    """统一格式化图表坐标轴：每年显示、去掉边框、只保留横向网格线"""
    from datetime import datetime

    # 格式化日期轴 - 每年显示
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45, fontsize=8)

    # 设置x轴范围为1994-当前年份
    if set_xlim:
        current_year = datetime.now().year
        ax.set_xlim(pd.Timestamp('1994-01-01'), pd.Timestamp(f'{current_year}-12-31'))

    # 去掉边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # 只保留横向网格线，去掉竖向网格线
    ax.grid(True, axis='y', alpha=0.3)
    ax.grid(False, axis='x')


def _plot_single_index(name: str, df: pd.DataFrame):
    """绘制单个指数的走势图"""
    fig, ax = plt.subplots(figsize=(8, 4))

    color = INDEX_COLORS.get(name, '#333333')
    ax.plot(df['date'], df['close'], color=color, linewidth=0.8)

    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('收盘价/点位')

    # 格式化坐标轴
    _format_chart_axis(ax)

    # 显示数据范围
    date_min = df['date'].min().strftime('%Y-%m-%d')
    date_max = df['date'].max().strftime('%Y-%m-%d')
    close_min = df['close'].min()
    close_max = df['close'].max()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 显示统计信息
    st.caption(f"📅 {date_min} ~ {date_max} | 最低: {close_min:.2f} | 最高: {close_max:.2f}")


def _display_index_combined(index_data: Dict[str, pd.DataFrame]):
    """合并展示所有指数（归一化对比）"""

    st.info("💡 为便于对比，各指数已归一化处理（以起始日期为基准 = 100）")

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, df in index_data.items():
        if df is not None and not df.empty:
            # 归一化：以第一个数据点为 100
            df_sorted = df.sort_values('date')
            first_close = df_sorted['close'].iloc[0]
            if first_close != 0:
                normalized = df_sorted['close'] / first_close * 100
            else:
                normalized = df_sorted['close']

            color = INDEX_COLORS.get(name, '#333333')
            ax.plot(df_sorted['date'], normalized,
                   label=name, color=color, linewidth=1)

    ax.set_title('各指数历史走势对比（归一化）', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('归一化指数（起始=100）')
    ax.legend(loc='upper left', fontsize=9)

    # 格式化坐标轴
    _format_chart_axis(ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =====================
# 概率分布图函数
# =====================

def display_index_distribution(adjusted_data: Dict[str, pd.DataFrame],
                                regression_results: Dict[str, dict],
                                errors: Dict[str, str] = None):
    """
    展示指数通胀调整后的对数化时间序列和回归分析

    Args:
        adjusted_data: {指数名称: 处理后的DataFrame}
        regression_results: {指数名称: 回归结果字典}
        errors: {指数名称: 错误信息}
    """
    st.subheader("📊 通胀调整与回归分析")

    # 显示错误信息
    if errors:
        for name, err in errors.items():
            if err:
                st.warning(f"⚠️ {name}: {err}")

    # 过滤有效数据
    valid_data = {k: v for k, v in adjusted_data.items()
                  if v is not None and not v.empty and k in regression_results}

    if not valid_data:
        st.error("未能获取有效的分析数据")
        return

    # 显示统计摘要
    _display_regression_summary(regression_results)

    st.divider()

    # 展示模式选择
    view_mode = st.radio(
        "图表展示模式",
        ["分别展示", "合并展示"],
        horizontal=True,
        key="dist_view_mode"
    )

    if view_mode == "分别展示":
        _display_distribution_separate(valid_data, regression_results)
    else:
        _display_distribution_combined(valid_data, regression_results)

    # 提供数据下载
    st.divider()
    st.markdown("**📥 下载处理后的数据**")
    cols = st.columns(min(len(valid_data), 5))
    for i, (name, df) in enumerate(valid_data.items()):
        with cols[i % 5]:
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label=f"{name[:6]}...",
                data=csv,
                file_name=f"{name}_adjusted.csv",
                mime="text/csv",
                key=f"download_adj_{name}"
            )


def _display_regression_summary(regression_results: Dict[str, dict]):
    """显示回归分析统计摘要"""
    st.markdown("### 📈 统计摘要")

    # 创建摘要表格
    summary_data = []
    for name, reg in regression_results.items():
        summary_data.append({
            '指数名称': name,
            '年化收益率': f"{reg['annual_return']*100:.2f}%",
            '年化波动率': f"{reg['annual_volatility']*100:.2f}%",
            'R²': f"{reg['r_squared']:.4f}",
            '日均斜率': f"{reg['slope']:.6f}",
            '残差标准差': f"{reg['std']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # 使用 metric 卡片显示关键指标
    st.markdown("#### 关键指标对比")
    cols = st.columns(len(regression_results))
    for i, (name, reg) in enumerate(regression_results.items()):
        with cols[i]:
            st.metric(
                label=name[:8],
                value=f"{reg['annual_return']*100:.1f}%",
                delta=f"波动率: {reg['annual_volatility']*100:.1f}%"
            )


def _display_distribution_separate(adjusted_data: Dict[str, pd.DataFrame],
                                    regression_results: Dict[str, dict]):
    """分别展示每个指数的对数化时间序列和回归线"""

    index_names = list(adjusted_data.keys())

    for i in range(0, len(index_names), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(index_names):
                name = index_names[idx]
                df = adjusted_data[name]
                reg = regression_results.get(name)
                with col:
                    _plot_single_distribution(name, df, reg)


def _plot_single_distribution(name: str, df: pd.DataFrame, reg: dict):
    """绘制单个指数的对数化时间序列和回归线"""

    fig, ax = plt.subplots(figsize=(8, 5))

    color = INDEX_COLORS.get(name, '#333333')

    # 绘制对数化指数值
    ax.plot(df['date'], df['log_close'], color=color, linewidth=0.8,
            alpha=0.7, label='对数化指数')

    if reg is not None:
        # 计算回归线的日期对应值
        df_sorted = df.sort_values('date').reset_index(drop=True)
        start_date = df_sorted['date'].iloc[0]
        days = (df_sorted['date'] - start_date).dt.days.values

        fitted = reg['slope'] * days + reg['intercept']
        upper = fitted + CONFIDENCE_SIGMA * reg['std']
        lower = fitted - CONFIDENCE_SIGMA * reg['std']

        # 绘制回归线
        ax.plot(df_sorted['date'], fitted, 'k-', linewidth=1.5,
                label='回归线')

        # 绘制标准差边界
        ax.plot(df_sorted['date'], upper, 'k--', linewidth=1,
                alpha=0.6, label=f'+{CONFIDENCE_SIGMA}σ')
        ax.plot(df_sorted['date'], lower, 'k--', linewidth=1,
                alpha=0.6, label=f'-{CONFIDENCE_SIGMA}σ')

        # 填充标准差区域
        ax.fill_between(df_sorted['date'], lower, upper,
                        alpha=0.1, color='gray')

    ax.set_title(f'{name}',fontsize=11, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('对数指数值（去通胀）')
    ax.legend(loc='upper left', fontsize=8)

    # 格式化坐标轴
    _format_chart_axis(ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _display_distribution_combined(adjusted_data: Dict[str, pd.DataFrame],
                                    regression_results: Dict[str, dict]):
    """合并展示所有指数的对数化时间序列"""

    st.info("💡 各指数对数化后的时间序列，便于对比长期增长趋势")

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, df in adjusted_data.items():
        if df is not None and not df.empty:
            df_sorted = df.sort_values('date')
            color = INDEX_COLORS.get(name, '#333333')

            # 绘制对数化指数
            ax.plot(df_sorted['date'], df_sorted['log_close'],
                   label=name, color=color, linewidth=1, alpha=0.8)

    ax.set_title('各指数对数化时间序列（去通胀）', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('对数指数值')
    ax.legend(loc='upper left', fontsize=9)

    # 格式化坐标轴
    _format_chart_axis(ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 绘制回归线对比图
    st.markdown("#### 回归趋势线对比")

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for name, df in adjusted_data.items():
        reg = regression_results.get(name)
        if df is not None and not df.empty and reg is not None:
            df_sorted = df.sort_values('date').reset_index(drop=True)
            start_date = df_sorted['date'].iloc[0]
            days = (df_sorted['date'] - start_date).dt.days.values

            fitted = reg['slope'] * days + reg['intercept']
            color = INDEX_COLORS.get(name, '#333333')

            ax2.plot(df_sorted['date'], fitted,
                    label=f"{name} (年化: {reg['annual_return']*100:.1f}%)",
                    color=color, linewidth=2)

    ax2.set_title('各指数回归趋势线对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('拟合值')
    ax2.legend(loc='upper left', fontsize=9)

    # 格式化坐标轴
    _format_chart_axis(ax2)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# =====================
# 指数对比分析函数
# =====================

def display_index_comparison(comparison_data: Dict, errors: Dict[str, str] = None):
    """
    展示道琼斯工业指数与中国指数的归一化对比图

    Args:
        comparison_data: {指数名称: {'base': df, 'compare': df, 'factor': float, ...}}
        errors: {指数名称: 错误信息}
    """
    st.subheader("📊 道琼斯工业指数 vs 中国指数")
    st.markdown(f"**归一化起始日期:** {COMPARISON_START_DATE}")
    st.info("💡 将道琼斯工业指数归一化至与中国指数相同的起始点位，便于对比长期相对表现")

    # 显示错误信息
    if errors:
        for name, err in errors.items():
            if err:
                st.warning(f"⚠️ {name}: {err}")

    if not comparison_data:
        st.error("未能获取有效的对比数据")
        return

    # 显示归一化因子摘要
    _display_normalization_summary(comparison_data)

    st.divider()

    # 展示模式选择
    view_mode = st.radio(
        "图表展示模式",
        ["分别展示", "合并展示"],
        horizontal=True,
        key="comparison_view_mode"
    )

    if view_mode == "分别展示":
        _display_comparison_separate(comparison_data)
    else:
        _display_comparison_combined(comparison_data)

    # 提供数据下载
    st.divider()
    _display_comparison_downloads(comparison_data)


def _display_normalization_summary(comparison_data: Dict):
    """显示归一化因子摘要"""
    st.markdown("### 📈 归一化因子")

    summary_data = []
    for name, data in comparison_data.items():
        base_df = data['base']
        compare_df = data['compare']

        if not base_df.empty and not compare_df.empty:
            # 中国指数信息
            base_start = base_df['close'].iloc[0]
            base_end = base_df['close'].iloc[-1]
            base_start_date = pd.to_datetime(base_df['date'].iloc[0]).strftime('%Y-%m-%d')
            base_return = (base_end / base_start - 1) * 100

            # 道琼斯指数信息
            compare_start = compare_df['close'].iloc[0]
            compare_end = compare_df['close'].iloc[-1]
            compare_start_date = pd.to_datetime(compare_df['date'].iloc[0]).strftime('%Y-%m-%d')
            compare_norm_end = compare_df['close_normalized'].iloc[-1]
            compare_return = (compare_end / compare_start - 1) * 100

            # 获取归一化因子
            factor = data['factor']

            # 添加中国指数行
            summary_data.append({
                '指数名称': name,
                '起始日期': base_start_date,
                '起始点位': f"{base_start:.2f}",
                '当前点位': f"{base_end:.2f}",
                '累计涨幅': f"{base_return:.1f}%",
                '归一化因子': f"{factor:.4f}",
                '归一化后点位': f"{base_end:.2f}"
            })

            # 添加道琼斯指数行
            summary_data.append({
                '指数名称': f"  └ 道琼斯工业指数",
                '起始日期': compare_start_date,
                '起始点位': f"{compare_start:.2f}",
                '当前点位': f"{compare_end:.2f}",
                '累计涨幅': f"{compare_return:.1f}%",
                '归一化因子': '-',
                '归一化后点位': f"{compare_norm_end:.2f}"
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _display_comparison_separate(comparison_data: Dict):
    """分别展示每个对比图"""

    index_names = list(comparison_data.keys())

    for i in range(0, len(index_names), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(index_names):
                name = index_names[idx]
                data = comparison_data[name]
                with col:
                    _plot_single_comparison(name, data)


def _plot_single_comparison(name: str, data: Dict):
    """绘制单个对比图"""

    base_df = data['base']
    compare_df = data['compare']
    start_date = data.get('start_date', COMPARISON_START_DATE)

    fig, ax = plt.subplots(figsize=(8, 5))

    # 中国指数颜色
    base_color = INDEX_COLORS.get(name, '#e41a1c')
    # 道琼斯指数颜色
    compare_color = INDEX_COLORS.get('道琼斯工业指数', '#a65628')

    # 绘制中国指数（实线）
    ax.plot(base_df['date'], base_df['close'],
            color=base_color, linewidth=1.2, label=name)

    # 绘制归一化后的道琼斯指数（虚线）
    ax.plot(compare_df['date'], compare_df['close_normalized'],
            color=compare_color, linewidth=1.2, linestyle='-',
            label=f"道琼斯工业指数（归一化）")

    ax.set_title(f'{name} vs 道琼斯工业指数\n（归一化至 {start_date}）',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('指数点位')
    ax.legend(loc='upper left', fontsize=9)

    # 格式化坐标轴
    _format_chart_axis(ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _display_comparison_combined(comparison_data: Dict):
    """合并展示所有对比"""

    st.markdown("#### 所有中国指数走势")

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, data in comparison_data.items():
        base_df = data['base']
        if not base_df.empty:
            color = INDEX_COLORS.get(name, '#333333')
            ax.plot(base_df['date'], base_df['close'],
                   label=name, color=color, linewidth=1)

    ax.set_title('中国主要指数走势对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('指数点位')
    ax.legend(loc='upper left', fontsize=9)

    # 格式化坐标轴
    _format_chart_axis(ax)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 归一化后的道琼斯指数对比
    st.markdown("#### 归一化后的道琼斯工业指数对比")
    st.info("💡 每条虚线表示道琼斯指数归一化到对应中国指数的起始点位")

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for name, data in comparison_data.items():
        compare_df = data['compare']
        if not compare_df.empty:
            color = INDEX_COLORS.get(name, '#333333')
            ax2.plot(compare_df['date'], compare_df['close_normalized'],
                    label=f"DJI→{name[:4]}", color=color, linewidth=1, linestyle='-')

    ax2.set_title('道琼斯工业指数（归一化至各中国指数起点）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('归一化后点位')
    ax2.legend(loc='upper left', fontsize=9)

    # 格式化坐标轴
    _format_chart_axis(ax2)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    

def _display_comparison_downloads(comparison_data: Dict):
    """提供对比数据下载"""
    st.markdown("**📥 下载对比数据**")

    cols = st.columns(min(len(comparison_data), 5))
    for i, (name, data) in enumerate(comparison_data.items()):
        with cols[i % 5]:
            # 合并基准和对比数据
            base_df = data['base'][['date', 'close']].copy()
            base_df.columns = ['date', f'{name}']

            compare_df = data['compare'][['date', 'close', 'close_normalized']].copy()
            compare_df.columns = ['date', 'DJI原始', 'DJI归一化']

            merged = pd.merge(base_df, compare_df, on='date', how='outer')
            merged = merged.sort_values('date')

            csv = merged.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label=f"{name[:6]}...",
                data=csv,
                file_name=f"{name}_vs_DJI.csv",
                mime="text/csv",
                key=f"download_cmp_{name}"
            )


# =====================
# 市净率展示
# =====================

def display_market_pb_monthly_trend(monthly_df: pd.DataFrame, loaded_at: str = None):
    """按板块分别展示市净率走势图，并在最后展示叠加对比图。"""
    if monthly_df is None or monthly_df.empty:
        st.warning("暂无市净率数据")
        return

    plot_df = monthly_df.copy()
    plot_df['日期'] = pd.to_datetime(plot_df['日期'])
    plot_df['市净率'] = pd.to_numeric(plot_df['市净率'], errors='coerce')
    if '板块' not in plot_df.columns:
        plot_df['板块'] = '市净率'
    plot_df = plot_df.dropna(subset=['日期', '市净率']).sort_values('日期')

    if plot_df.empty:
        st.warning("暂无有效的市净率数据")
        return

    start_date = plot_df['日期'].min().strftime('%Y-%m')
    end_date = plot_df['日期'].max().strftime('%Y-%m')
    caption = f"数据范围：{start_date} 至 {end_date}"
    if loaded_at:
        caption += f" | 缓存更新时间：{loaded_at}"
    st.caption(caption)

    preferred_markets = ['上证', '深证', '创业板', '科创板']
    ordered_markets = [market for market in preferred_markets if market in plot_df['板块'].unique()]
    remaining_markets = [market for market in plot_df['板块'].unique() if market not in ordered_markets]
    ordered_markets.extend(remaining_markets)

    for market in ordered_markets:
        market_df = plot_df[plot_df['板块'] == market].copy()
        if market_df.empty:
            continue

        st.markdown(f"### {market}")
        st.altair_chart(
            _build_market_pb_chart(
                market_df,
                title=f"{market}市净率（月度）",
                show_legend=False
            ),
            use_container_width=True
        )

        table_df = market_df[['日期', '市净率']].copy()
        table_df['日期'] = table_df['日期'].dt.strftime('%Y-%m-%d')
        table_df['市净率'] = table_df['市净率'].round(2)
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.divider()

    st.markdown("### 叠加对比")
    st.altair_chart(
        _build_market_pb_chart(
            plot_df,
            title='市净率叠加对比（月度）',
            show_legend=True
        ),
        use_container_width=True
    )


def _build_market_pb_chart(plot_df: pd.DataFrame, title: str, show_legend: bool = True):
    """构建市净率交互图。"""
    chart_df = plot_df.copy()
    chart_df['日期标签'] = chart_df['日期'].dt.strftime('%Y-%m-%d')

    selected_point = alt.selection_point(
        fields=['日期标签', '板块'],
        nearest=True,
        on='click',
        empty=False
    )

    tooltip = [
        alt.Tooltip('板块:N', title='板块'),
        alt.Tooltip('日期标签:N', title='日期'),
        alt.Tooltip('市净率:Q', title='市净率', format='.2f'),
    ]

    legend = alt.Legend(title='板块') if show_legend else None
    base = alt.Chart(chart_df).encode(
        x=alt.X('日期:T', title='月份'),
        y=alt.Y('市净率:Q', title='市净率', scale=alt.Scale(zero=False)),
        color=alt.Color(
            '板块:N',
            legend=legend,
            scale=alt.Scale(
                domain=['上证', '深证', '创业板', '科创板'],
                range=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
            )
        ),
    )
    line = base.mark_line(strokeWidth=2)
    points = base.mark_circle(size=34).encode(
        opacity=alt.condition(selected_point, alt.value(1), alt.value(0.3)),
        tooltip=tooltip
    ).add_params(selected_point)
    rule = base.mark_rule(color='#777777').encode(
        opacity=alt.condition(selected_point, alt.value(0.45), alt.value(0)),
        tooltip=tooltip
    )

    return (line + points + rule).properties(
        height=430,
        title=title
    ).interactive(bind_y=False)


# =====================
# 主要股票基本面数据展示（多年年报）
# =====================

def display_featured_stocks(df: pd.DataFrame, errors: Dict = None):
    """
    展示重点关注股票的多年年报基本面数据

    Args:
        df: 股票基本面数据 DataFrame（包含多年数据）
        errors: 错误信息字典
    """
    if df is None or df.empty:
        st.warning("暂无股票数据")
        return

    # 显示错误信息（折叠）
    if errors:
        with st.expander("⚠️ 数据获取警告", expanded=False):
            for name, err in errors.items():
                st.warning(f"{name}: {err}")

    # 获取股票列表
    stock_names = df['股票名称'].unique().tolist()

    # 显示模式选择
    col1, col2 = st.columns([1, 3])
    with col1:
        display_mode = st.radio(
            "展示模式",
            ["📊 分股票表格", "📈 汇总对比表"],
            key="stock_display_mode"
        )
    with col2:
        if display_mode == "📊 分股票表格":
            selected_stock = st.selectbox(
                "选择股票",
                options=["全部"] + stock_names,
                key="stock_selector"
            )
        else:
            selected_indicator = st.selectbox(
                "选择指标",
                options=["营业总收入", "营业成本", "归母净利润", "销售毛利率", "销售净利率",
                        "净资产收益率", "权益乘数", "存货周转率", "应收账款周转率", "每股净资产"],
                key="indicator_selector"
            )

    st.divider()

    if display_mode == "📊 分股票表格":
        _display_stocks_by_stock(df, selected_stock)
    else:
        _display_stocks_comparison(df, selected_indicator)

    # 数据下载
    st.divider()
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载全部数据 (CSV)",
        data=csv,
        file_name="featured_stocks_yearly_data.csv",
        mime="text/csv"
    )


def _display_stocks_by_stock(df: pd.DataFrame, selected_stock: str = "全部"):
    """按股票分别展示多年数据"""

    if selected_stock == "全部":
        stocks_to_show = df['股票名称'].unique()
    else:
        stocks_to_show = [selected_stock]

    for stock_name in stocks_to_show:
        stock_df = df[df['股票名称'] == stock_name].copy()
        stock_code = stock_df['股票代码'].iloc[0] if not stock_df.empty else ''

        st.markdown(f"### 📊 {stock_name} ({stock_code})")

        # 定义要展示的列及其显示名称
        display_cols = {
            '报告期': '报告期',
            '营业总收入': '营业总收入',
            '营业成本': '营业成本',
            '归母净利润': '归母净利润',
            '每股净资产': '每股净资产',
            '销售净利率': '销售净利率(%)',
            '销售毛利率': '销售毛利率(%)',
            '净资产收益率': 'ROE(%)',
            '权益乘数': '权益乘数',
            '存货': '存货',
            '应收账款': '应收账款',
            '应付账款': '应付账款',
            '存货周转率': '存货周转率',
            '存货周转天数': '存货周转天数',
            '应收账款周转率': '应收账款周转率',
            '应付账款周转率': '应付账款周转率',
            '主营业务成本率': '成本率(%)',
            '总资产': '总资产',
            '平均存货周期': '平均存货周期(天)',
        }

        # 筛选存在的列
        available_cols = [col for col in display_cols.keys() if col in stock_df.columns]
        display_df = stock_df[available_cols].copy()

        # 格式化数值列
        for col in display_df.columns:
            if col != '报告期':
                display_df[col] = display_df[col].apply(_format_table_value)

        # 重命名列
        display_df.columns = [display_cols.get(col, col) for col in available_cols]

        # 按报告期倒序排列
        display_df = display_df.sort_values('报告期', ascending=False)

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.divider()


def _display_stocks_comparison(df: pd.DataFrame, indicator: str):
    """横向对比各股票同一指标的多年数据"""

    st.markdown(f"### 📈 {indicator} 历年对比")

    # 获取所有股票和报告期
    stocks = df['股票名称'].unique()
    periods = sorted(df['报告期'].unique(), reverse=True)

    # 构建对比表格
    comparison_data = {'报告期': periods}

    for stock in stocks:
        stock_df = df[df['股票名称'] == stock]
        values = []
        for period in periods:
            period_data = stock_df[stock_df['报告期'] == period]
            if not period_data.empty and indicator in period_data.columns:
                val = period_data[indicator].iloc[0]
                values.append(_format_table_value(val))
            else:
                values.append('-')
        comparison_data[stock] = values

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # 绘制趋势图
    st.markdown("#### 📉 趋势图")
    _plot_indicator_trend(df, indicator, stocks)


def _plot_indicator_trend(df: pd.DataFrame, indicator: str, stocks):
    """绘制指标趋势图"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for idx, stock in enumerate(stocks):
        stock_df = df[df['股票名称'] == stock].copy()
        if indicator not in stock_df.columns:
            continue

        # 按报告期排序
        stock_df = stock_df.sort_values('报告期')

        # 提取年份
        stock_df['年份'] = stock_df['报告期'].str.extract(r'(\d{4})')[0].astype(int)

        # 转换为数值
        values = pd.to_numeric(stock_df[indicator], errors='coerce')
        years = stock_df['年份']

        # 过滤有效数据
        valid_mask = values.notna()
        if valid_mask.sum() > 0:
            ax.plot(years[valid_mask], values[valid_mask],
                   marker='o', label=stock, color=colors[idx % len(colors)], linewidth=2)

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel(indicator, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{indicator} 历年趋势', fontsize=14)

    # 设置x轴为整数年份
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _format_table_value(value):
    """格式化表格中的数值"""
    if value is None or pd.isna(value):
        return '-'
    try:
        num = float(value)
        if abs(num) >= 1e8:
            return f"{num/1e8:.2f}亿"
        elif abs(num) >= 1e4:
            return f"{num/1e4:.2f}万"
        elif abs(num) >= 100:
            return f"{num:.0f}"
        elif abs(num) >= 1:
            return f"{num:.2f}"
        else:
            return f"{num:.4f}"
    except:
        return str(value)
