"""
A股市场数据可视化工具
主应用入口
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os

# 设置请求超时时间（解决云端部署时的网络问题）
os.environ['AKSHARE_TIMEOUT'] = '60'

from config import APP_TITLE, APP_ICON
from data_fetcher import (
    fetch_sse_summary,
    fetch_szse_summary,
    fetch_profit_top100,
    calculate_profit_distribution,
    fetch_all_index_data,
    fetch_cpi_yearly,
    process_all_indices_inflation,
    fetch_dji_index,
    prepare_all_index_comparisons,
    fetch_all_featured_stocks_data,
    fetch_all_us_stocks_data,
    parse_uploaded_index_file,
    validate_uploaded_index_file,
    load_prepared_stock_data,
    load_prepared_us_stock_data
)
from charts import (
    display_market_summary,
    display_profit_top100,
    display_profit_distribution,
    display_index_charts,
    display_index_distribution,
    display_index_comparison,
    display_featured_stocks
)

# =====================
# 页面配置
# =====================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# 侧边栏：参数设置
# =====================
with st.sidebar:
    st.header("⚙️ 参数设置")

    # 市场统计日期
    st.subheader("市场统计")
    market_date = st.date_input(
        "选择日期",
        value=datetime.now() - timedelta(days=1),
        max_value=datetime.now(),
        help="用于获取深交所统计数据"
    )
    market_date_str = market_date.strftime('%Y%m%d')

    # 报告期选择
    st.subheader("业绩报告")
    current_year = datetime.now().year
    report_options = {
        f"{current_year}年三季报": f"{current_year}0930",
        f"{current_year}年中报": f"{current_year}0630",
        f"{current_year}年一季报": f"{current_year}0331",
        f"{current_year-1}年年报": f"{current_year-1}1231",
        f"{current_year-1}年三季报": f"{current_year-1}0930",
    }

    selected_report = st.selectbox(
        "选择报告期",
        options=list(report_options.keys()),
        index=0,
        help="选择要查询的财务报告期"
    )
    report_date_str = report_options[selected_report]

    # 指数走势日期范围
    st.subheader("指数走势")
    index_start_date = st.date_input(
        "起始日期",
        value=datetime(1994, 1, 3),
        min_value=datetime(1990, 1, 1),
        max_value=datetime.now(),
        help="指数数据起始日期"
    )
    index_start_str = index_start_date.strftime('%Y%m%d')

    st.divider()

    # =====================
    # 道琼斯指数文件上传
    # =====================
    st.subheader("道琼斯指数数据上传")

    # 显示当前使用的数据源
    using_uploaded_dji = 'uploaded_dji_data' in st.session_state and st.session_state.uploaded_dji_data is not None
    using_uploaded_djchina = 'uploaded_djchina_data' in st.session_state and st.session_state.uploaded_djchina_data is not None
    using_uploaded_djsh = 'uploaded_djsh_data' in st.session_state and st.session_state.uploaded_djsh_data is not None
    using_uploaded_djsz = 'uploaded_djsz_data' in st.session_state and st.session_state.uploaded_djsz_data is not None

    uploaded_count = sum([using_uploaded_dji, using_uploaded_djchina, using_uploaded_djsh, using_uploaded_djsz])
    if uploaded_count > 0:
        st.info(f"📤 正在使用上传的数据 ({uploaded_count}个)")
    else:
        st.caption("默认使用本地数据，截止时间为2025年12月5日")

    # 道琼斯工业指数上传
    with st.expander("🇺🇸 道琼斯工业指数 (DJI)", expanded=False):
        dji_file = st.file_uploader(
            "上传DJI数据文件",
            type=['xlsx', 'xls', 'csv'],
            key='dji_uploader',
            help="支持 Excel (.xlsx, .xls) 或 CSV 格式"
        )

        if dji_file is not None:
            is_valid, msg, preview_df = validate_uploaded_index_file(dji_file)
            if is_valid:
                st.success(msg)
                st.caption("数据预览:")
                st.dataframe(preview_df[['date', 'close']].head(3), hide_index=True)
                # 存储解析后的数据
                dji_file.seek(0)  # 重置文件指针
                st.session_state.uploaded_dji_data = parse_uploaded_index_file(dji_file, index_start_str)
            else:
                st.error(msg)
                st.session_state.uploaded_dji_data = None

        if using_uploaded_dji:
            if st.button("🔄 重置为默认数据", key='reset_dji'):
                st.session_state.uploaded_dji_data = None
                st.rerun()

    # 道琼斯中国指数上传
    with st.expander("🇨🇳 道琼斯中国指数 (DJCHINA)", expanded=False):
        djchina_file = st.file_uploader(
            "上传DJCHINA数据文件",
            type=['xlsx', 'xls', 'csv'],
            key='djchina_uploader',
            help="支持 Excel (.xlsx, .xls) 或 CSV 格式"
        )

        if djchina_file is not None:
            is_valid, msg, preview_df = validate_uploaded_index_file(djchina_file)
            if is_valid:
                st.success(msg)
                st.caption("数据预览:")
                st.dataframe(preview_df[['date', 'close']].head(3), hide_index=True)
                # 存储解析后的数据
                djchina_file.seek(0)
                st.session_state.uploaded_djchina_data = parse_uploaded_index_file(djchina_file, index_start_str)
            else:
                st.error(msg)
                st.session_state.uploaded_djchina_data = None

        if using_uploaded_djchina:
            if st.button("🔄 重置为默认数据", key='reset_djchina'):
                st.session_state.uploaded_djchina_data = None
                st.rerun()

    # 道琼斯上海指数上传
    with st.expander("🇨🇳 道琼斯上海指数 (DJSH)", expanded=False):
        djsh_file = st.file_uploader(
            "上传DJSH数据文件",
            type=['xlsx', 'xls', 'csv'],
            key='djsh_uploader',
            help="支持 Excel (.xlsx, .xls) 或 CSV 格式"
        )

        if djsh_file is not None:
            is_valid, msg, preview_df = validate_uploaded_index_file(djsh_file)
            if is_valid:
                st.success(msg)
                st.caption("数据预览:")
                st.dataframe(preview_df[['date', 'close']].head(3), hide_index=True)
                # 存储解析后的数据
                djsh_file.seek(0)
                st.session_state.uploaded_djsh_data = parse_uploaded_index_file(djsh_file, index_start_str)
            else:
                st.error(msg)
                st.session_state.uploaded_djsh_data = None

        if using_uploaded_djsh:
            if st.button("🔄 重置为默认数据", key='reset_djsh'):
                st.session_state.uploaded_djsh_data = None
                st.rerun()

    # 道琼斯深圳指数上传
    with st.expander("🇨🇳 道琼斯深圳指数 (DJSZ)", expanded=False):
        djsz_file = st.file_uploader(
            "上传DJSZ数据文件",
            type=['xlsx', 'xls', 'csv'],
            key='djsz_uploader',
            help="支持 Excel (.xlsx, .xls) 或 CSV 格式"
        )

        if djsz_file is not None:
            is_valid, msg, preview_df = validate_uploaded_index_file(djsz_file)
            if is_valid:
                st.success(msg)
                st.caption("数据预览:")
                st.dataframe(preview_df[['date', 'close']].head(3), hide_index=True)
                # 存储解析后的数据
                djsz_file.seek(0)
                st.session_state.uploaded_djsz_data = parse_uploaded_index_file(djsz_file, index_start_str)
            else:
                st.error(msg)
                st.session_state.uploaded_djsz_data = None

        if using_uploaded_djsz:
            if st.button("🔄 重置为默认数据", key='reset_djsz'):
                st.session_state.uploaded_djsz_data = None
                st.rerun()

    st.divider()

    # 刷新按钮
    refresh_btn = st.button("🔄 刷新所有数据", type="primary", use_container_width=True)

    st.divider()
    st.caption("数据来源: AKShare + 本地Excel")
    st.caption(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# =====================
# 主页面
# =====================
st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown(f"**市场日期:** {market_date_str} | **报告期:** {selected_report}")

# 使用 tabs 组织内容
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 市场统计",
    "🏆 归母净利润100强",
    "🍰 利润分布",
    "📈 指数走势",
    "🧾 概率分布图",
    "🌟 指数对比",
    "🏢 主要股票数据"
])

# =====================
# Tab 1: 市场统计
# =====================
with tab1:
    st.header("中国上市公司市场统计")

    # 懒加载：只有点击按钮时才加载数据
    load_tab1 = st.button("📥 加载市场统计数据", key='load_tab1') if 'sse_data' not in st.session_state else False

    if refresh_btn or load_tab1:
        with st.spinner("正在获取上交所数据..."):
            try:
                st.session_state.sse_data = fetch_sse_summary()
                st.session_state.sse_error = None
            except Exception as e:
                st.session_state.sse_data = None
                st.session_state.sse_error = str(e)

        with st.spinner("正在获取深交所数据..."):
            try:
                st.session_state.szse_data = fetch_szse_summary(market_date_str)
                st.session_state.szse_error = None
            except Exception as e:
                st.session_state.szse_data = None
                st.session_state.szse_error = str(e)

    # 显示数据或错误
    if st.session_state.get('sse_error'):
        st.error(f"上交所数据获取失败: {st.session_state.sse_error}")
    if st.session_state.get('szse_error'):
        st.error(f"深交所数据获取失败: {st.session_state.szse_error}")

    if st.session_state.get('sse_data') is not None or st.session_state.get('szse_data') is not None:
        display_market_summary(
            st.session_state.get('sse_data'),
            st.session_state.get('szse_data')
        )
    elif 'sse_data' not in st.session_state:
        st.info("👆 点击上方按钮加载数据")

# =====================
# Tab 2: 净利润100强
# =====================
with tab2:
    st.header(f"A股归母净利润100强 ({selected_report})")

    # 懒加载
    load_tab2 = st.button("📥 加载净利润100强数据", key='load_tab2') if 'profit_top100' not in st.session_state else False

    if refresh_btn or load_tab2:
        with st.spinner("正在获取业绩数据，请稍候..."):
            try:
                st.session_state.profit_top100 = fetch_profit_top100(report_date_str)
                st.session_state.profit_error = None
            except Exception as e:
                st.session_state.profit_top100 = None
                st.session_state.profit_error = str(e)

    if st.session_state.get('profit_error'):
        st.error(f"获取数据失败: {st.session_state.profit_error}")
    elif st.session_state.get('profit_top100') is not None:
        display_profit_top100(st.session_state.get('profit_top100'))
    elif 'profit_top100' not in st.session_state:
        st.info("👆 点击上方按钮加载数据")

# =====================
# Tab 3: 利润分布
# =====================
with tab3:
    st.header(f"净利润行业分布分析 ({selected_report})")

    # 使用 profit_top100 的原始完整数据计算分布（依赖 Tab2 的数据）
    if st.session_state.get('profit_top100') is not None:
        if refresh_btn or 'profit_dist' not in st.session_state:
            with st.spinner("正在计算利润分布..."):
                try:
                    result_df, pie_data = calculate_profit_distribution(
                        st.session_state.profit_top100
                    )
                    st.session_state.profit_dist = (result_df, pie_data)
                    st.session_state.dist_error = None
                except Exception as e:
                    st.session_state.profit_dist = None
                    st.session_state.dist_error = str(e)

        if st.session_state.get('dist_error'):
            st.error(f"计算分布失败: {st.session_state.dist_error}")
        elif st.session_state.get('profit_dist'):
            result_df, pie_data = st.session_state.profit_dist
            display_profit_distribution(result_df, pie_data)
    else:
        st.info("请先在「净利润100强」页面加载数据")

# =====================
# Tab 4: 指数走势
# =====================
with tab4:
    st.header("股票指数历史走势")
    st.markdown(f"**数据范围:** {index_start_date.strftime('%Y-%m-%d')} ~ 今")

    # 显示数据来源状态
    uploaded_indices = []
    if st.session_state.get('uploaded_djchina_data') is not None:
        uploaded_indices.append("道琼斯中国指数")
    if st.session_state.get('uploaded_djsh_data') is not None:
        uploaded_indices.append("道琼斯上海指数")
    if st.session_state.get('uploaded_djsz_data') is not None:
        uploaded_indices.append("道琼斯深圳指数")
    if uploaded_indices:
        st.success(f"📤 使用上传的数据: {', '.join(uploaded_indices)}")

    # 懒加载
    load_tab4 = st.button("📥 加载指数走势数据", key='load_tab4') if 'index_data' not in st.session_state else False

    if refresh_btn or load_tab4:
        with st.spinner("正在获取指数数据，请稍候（数据量较大）..."):
            try:
                index_data, index_errors = fetch_all_index_data(
                    start_date=index_start_str,
                    data_dir='.'
                )
                # 如果有上传的道琼斯指数数据，替换默认数据
                if st.session_state.get('uploaded_djchina_data') is not None:
                    index_data['道琼斯中国指数'] = st.session_state.uploaded_djchina_data
                    if '道琼斯中国指数' in index_errors:
                        del index_errors['道琼斯中国指数']
                if st.session_state.get('uploaded_djsh_data') is not None:
                    index_data['道琼斯上海指数'] = st.session_state.uploaded_djsh_data
                    if '道琼斯上海指数' in index_errors:
                        del index_errors['道琼斯上海指数']
                if st.session_state.get('uploaded_djsz_data') is not None:
                    index_data['道琼斯深圳指数'] = st.session_state.uploaded_djsz_data
                    if '道琼斯深圳指数' in index_errors:
                        del index_errors['道琼斯深圳指数']
                st.session_state.index_data = index_data
                st.session_state.index_errors = index_errors
            except Exception as e:
                st.session_state.index_data = {}
                st.session_state.index_errors = {'系统错误': str(e)}

    # 显示图表
    if st.session_state.get('index_data'):
        display_index_charts(
            st.session_state.index_data,
            st.session_state.get('index_errors', {})
        )
    elif 'index_data' not in st.session_state:
        st.info("👆 点击上方按钮加载数据")
    else:
        st.warning("暂无指数数据")

# =====================
# Tab 5: 概率分布图（通胀调整与回归分析）
# =====================
with tab5:
    st.header("指数通胀调整与概率分布分析")
    st.markdown(f"**数据范围:** {index_start_date.strftime('%Y-%m-%d')} ~ 今")
    st.markdown("对指数进行通胀调整（去除CPI影响）和对数化处理，然后进行OLS线性回归分析。")

    # 懒加载（依赖 Tab4 的指数数据）
    if st.session_state.get('index_data'):
        load_tab5 = st.button("📥 加载概率分布分析", key='load_tab5') if 'adjusted_data' not in st.session_state else False

        if refresh_btn or load_tab5:
            with st.spinner("正在获取CPI数据..."):
                try:
                    cpi_yearly = fetch_cpi_yearly()
                    st.session_state.cpi_yearly = cpi_yearly
                    st.session_state.cpi_error = None
                except Exception as e:
                    st.session_state.cpi_yearly = None
                    st.session_state.cpi_error = str(e)

            # 进行通胀调整和回归分析
            if st.session_state.get('cpi_yearly') is not None:
                with st.spinner("正在进行通胀调整和回归分析，请稍候..."):
                    try:
                        adjusted_data, regression_results, adj_errors = process_all_indices_inflation(
                            st.session_state.index_data,
                            st.session_state.cpi_yearly
                        )
                        st.session_state.adjusted_data = adjusted_data
                        st.session_state.regression_results = regression_results
                        st.session_state.adj_errors = adj_errors
                    except Exception as e:
                        st.session_state.adjusted_data = {}
                        st.session_state.regression_results = {}
                        st.session_state.adj_errors = {'系统错误': str(e)}

        # 显示错误信息
        if st.session_state.get('cpi_error'):
            st.error(f"CPI数据获取失败: {st.session_state.cpi_error}")

        # 显示分析结果
        if st.session_state.get('adjusted_data') and st.session_state.get('regression_results'):
            display_index_distribution(
                st.session_state.adjusted_data,
                st.session_state.regression_results,
                st.session_state.get('adj_errors', {})
            )
        elif 'adjusted_data' not in st.session_state:
            st.info("👆 点击上方按钮加载数据")
    else:
        st.info("请先在「指数走势」页面加载指数数据")

# =====================
# Tab 6: 指数对比分析
# =====================
with tab6:
    st.header("道琼斯工业指数 vs 中国指数对比分析")
    st.markdown(f"**起始日期:** 1994-01-03 | **归一化基准:** 中国指数起始点位")

    # 显示数据来源状态
    using_uploaded_dji = st.session_state.get('uploaded_dji_data') is not None
    if using_uploaded_dji:
        st.success("📤 道琼斯工业指数: 使用上传的数据")
    else:
        st.caption("📂 道琼斯工业指数: 使用默认本地数据")

    # 懒加载（依赖 Tab4 的指数数据）
    if st.session_state.get('index_data'):
        load_tab6 = st.button("📥 加载指数对比数据", key='load_tab6') if 'comparison_data' not in st.session_state else False

        if refresh_btn or load_tab6:
            # 获取道琼斯工业指数数据
            with st.spinner("正在获取道琼斯工业指数数据..."):
                try:
                    if st.session_state.get('uploaded_dji_data') is not None:
                        dji_data = st.session_state.uploaded_dji_data
                    else:
                        dji_data = fetch_dji_index(data_dir='.')
                    st.session_state.dji_data = dji_data
                    st.session_state.dji_error = None
                except Exception as e:
                    st.session_state.dji_data = None
                    st.session_state.dji_error = str(e)

            # 准备对比数据
            if st.session_state.get('dji_data') is not None:
                with st.spinner("正在计算归一化对比数据..."):
                    try:
                        comparison_data, comparison_errors = prepare_all_index_comparisons(
                            st.session_state.index_data,
                            st.session_state.dji_data
                        )
                        st.session_state.comparison_data = comparison_data
                        st.session_state.comparison_errors = comparison_errors
                    except Exception as e:
                        st.session_state.comparison_data = {}
                        st.session_state.comparison_errors = {'系统错误': str(e)}

        # 显示错误信息
        if st.session_state.get('dji_error'):
            st.error(f"道琼斯工业指数数据获取失败: {st.session_state.dji_error}")

        # 显示对比图表
        comparison_data = st.session_state.get('comparison_data', {})
        comparison_errors = st.session_state.get('comparison_errors', {})

        if comparison_data:
            display_index_comparison(comparison_data, comparison_errors)
        elif comparison_errors:
            st.error("数据处理出现问题：")
            for name, err in comparison_errors.items():
                st.warning(f"⚠️ {name}: {err}")
        elif 'comparison_data' not in st.session_state:
            st.info("👆 点击上方按钮加载数据")
    else:
        st.info("请先在「指数走势」页面加载指数数据")

# =====================
# Tab 7: 主要股票基本面数据
# =====================
with tab7:
    st.header("🏢 主要股票基本面数据")

    # 刷新按钮
    refresh_stocks_btn = st.button("🔄 刷新最新数据 (从API获取)", key='refresh_stocks')

    # =====================
    # A股数据
    # =====================
    st.subheader("🇨🇳 A股重点股票")
    st.markdown("比亚迪、美的集团、海尔智家、格力电器")

    # 数据加载逻辑：
    # 1. 首次加载：从预加载文件读取（快速）
    # 2. 点击刷新：从 API 获取最新数据
    if refresh_stocks_btn:
        # 用户点击刷新 -> 从 API 获取最新数据
        a_progress = st.progress(0, text="正在从API并行获取A股最新数据...")
        try:
            stocks_df, stocks_errors = fetch_all_featured_stocks_data()
            st.session_state.featured_stocks_data = stocks_df
            st.session_state.featured_stocks_errors = stocks_errors
            st.session_state.featured_stocks_source = 'api'  # 标记数据来源
            a_progress.progress(100, text="✅ A股最新数据加载完成")
        except Exception as e:
            st.session_state.featured_stocks_data = None
            st.session_state.featured_stocks_errors = {'系统错误': str(e)}
            st.session_state.featured_stocks_source = 'api_error'
            a_progress.progress(100, text="❌ A股数据加载失败")
        a_progress.empty()
    elif 'featured_stocks_data' not in st.session_state:
        # 首次加载 -> 从预加载文件读取
        stocks_df, stocks_errors = load_prepared_stock_data('prepared_stock_data.csv')
        if not stocks_df.empty:
            st.session_state.featured_stocks_data = stocks_df
            st.session_state.featured_stocks_errors = stocks_errors
            st.session_state.featured_stocks_source = 'preload'  # 标记数据来源
        else:
            # 预加载失败，回退到 API
            st.warning("预加载文件读取失败，正在从API获取数据...")
            try:
                stocks_df, stocks_errors = fetch_all_featured_stocks_data()
                st.session_state.featured_stocks_data = stocks_df
                st.session_state.featured_stocks_errors = stocks_errors
                st.session_state.featured_stocks_source = 'api'
            except Exception as e:
                st.session_state.featured_stocks_data = None
                st.session_state.featured_stocks_errors = {'系统错误': str(e)}
                st.session_state.featured_stocks_source = 'api_error'

    # 显示数据来源标识
    data_source = st.session_state.get('featured_stocks_source', 'unknown')
    if data_source == 'preload':
        st.caption("📁 当前显示：预加载数据 | 点击上方按钮获取最新数据")
    elif data_source == 'api':
        st.caption("🌐 当前显示：API 最新数据")

    # 显示A股数据
    stocks_df = st.session_state.get('featured_stocks_data')
    stocks_errors = st.session_state.get('featured_stocks_errors', {})

    if stocks_df is not None and not stocks_df.empty:
        display_featured_stocks(stocks_df, stocks_errors)
    elif stocks_errors:
        st.error("A股数据获取出现问题：")
        for name, err in stocks_errors.items():
            st.warning(f"⚠️ {name}: {err}")
    else:
        st.info("数据加载中...")

    st.divider()

    # =====================
    # 美股数据
    # =====================
    st.subheader("🇺🇸 美股重点股票")
    st.markdown("特斯拉、丰田 - 经营周期分析")

    # 美股数据加载逻辑（同A股）
    if refresh_stocks_btn:
        # 用户点击刷新 -> 从 API 获取最新数据
        us_progress = st.progress(0, text="正在从API并行获取美股最新数据...")
        try:
            us_df, us_errors = fetch_all_us_stocks_data()
            st.session_state.us_stocks_data = us_df
            st.session_state.us_stocks_errors = us_errors
            st.session_state.us_stocks_source = 'api'
            us_progress.progress(100, text="✅ 美股最新数据加载完成")
        except Exception as e:
            st.session_state.us_stocks_data = None
            st.session_state.us_stocks_errors = {'系统错误': str(e)}
            st.session_state.us_stocks_source = 'api_error'
            us_progress.progress(100, text="❌ 美股数据加载失败")
        us_progress.empty()
    elif 'us_stocks_data' not in st.session_state:
        # 首次加载 -> 尝试从预加载文件读取
        us_df, us_errors = load_prepared_us_stock_data('prepared_us_stock_data.csv')
        if not us_df.empty:
            st.session_state.us_stocks_data = us_df
            st.session_state.us_stocks_errors = us_errors
            st.session_state.us_stocks_source = 'preload'
        else:
            # 预加载文件不存在，从 API 获取
            try:
                us_df, us_errors = fetch_all_us_stocks_data()
                st.session_state.us_stocks_data = us_df
                st.session_state.us_stocks_errors = us_errors
                st.session_state.us_stocks_source = 'api'
            except Exception as e:
                st.session_state.us_stocks_data = None
                st.session_state.us_stocks_errors = {'系统错误': str(e)}
                st.session_state.us_stocks_source = 'api_error'

    # 显示数据来源标识
    us_data_source = st.session_state.get('us_stocks_source', 'unknown')
    if us_data_source == 'preload':
        st.caption("📁 当前显示：预加载数据 | 点击上方按钮获取最新数据")
    elif us_data_source == 'api':
        st.caption("🌐 当前显示：API 最新数据")

    # 显示美股数据
    us_df = st.session_state.get('us_stocks_data')
    us_errors = st.session_state.get('us_stocks_errors', {})

    if us_df is not None and not us_df.empty:
        # 显示错误信息（折叠）
        if us_errors:
            with st.expander("⚠️ 美股数据获取警告", expanded=False):
                for name, err in us_errors.items():
                    st.warning(f"{name}: {err}")

        # 分股票展示
        us_stock_names = us_df['股票名称'].unique().tolist()

        for stock_name in us_stock_names:
            stock_data = us_df[us_df['股票名称'] == stock_name]
            st.markdown(f"**{stock_name} ({stock_data['股票代码'].iloc[0]})**")

            # 显示表格
            display_df = stock_data[['报告期', '存货周转天数', '应收账款周转天数', '经营周期']].copy()

            # 格式化数值
            for col in ['存货周转天数', '应收账款周转天数', '经营周期']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    elif us_errors:
        st.error("美股数据获取出现问题：")
        for name, err in us_errors.items():
            st.warning(f"⚠️ {name}: {err}")
    else:
        st.info("数据加载中...")
