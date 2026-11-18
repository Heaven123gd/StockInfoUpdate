"""Apple 风格的 Streamlit 页面样式与顶部 Hero。"""

from html import escape

import streamlit as st


def inject_apple_ui() -> None:
    """注入全局 Apple 风格设计系统。"""
    st.markdown(
        """
        <style>
        :root {
            --apple-bg: #f5f5f7;
            --apple-surface: rgba(255, 255, 255, 0.88);
            --apple-surface-solid: #ffffff;
            --apple-text: #1d1d1f;
            --apple-secondary: #6e6e73;
            --apple-tertiary: #86868b;
            --apple-blue: #0071e3;
            --apple-blue-hover: #0077ed;
            --apple-blue-soft: rgba(0, 113, 227, 0.10);
            --apple-separator: rgba(29, 29, 31, 0.08);
            --apple-shadow: 0 18px 50px rgba(0, 0, 0, 0.055),
                            0 2px 10px rgba(0, 0, 0, 0.025);
            --apple-shadow-hover: 0 28px 70px rgba(0, 0, 0, 0.09),
                                  0 8px 24px rgba(0, 0, 0, 0.04);
            --apple-radius: 28px;
            --apple-radius-small: 18px;
            --apple-ease: cubic-bezier(0.22, 1, 0.36, 1);
            --apple-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        html {
            scroll-behavior: smooth;
            background: var(--apple-bg);
        }

        html, body, .stApp, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                         "SF Pro Text", "PingFang SC", "Helvetica Neue",
                         "Microsoft YaHei", Arial, sans-serif;
            color: var(--apple-text);
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }

        body {
            background: var(--apple-bg);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 78% -10%, rgba(0, 113, 227, 0.07), transparent 30rem),
                radial-gradient(circle at 10% 22%, rgba(135, 206, 250, 0.05), transparent 26rem),
                var(--apple-bg);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .block-container {
            max-width: 1480px;
            padding: 3.25rem 3.5rem 7rem;
        }

        header[data-testid="stHeader"] {
            background: rgba(245, 245, 247, 0.72);
            border-bottom: 1px solid rgba(255, 255, 255, 0.68);
            backdrop-filter: saturate(180%) blur(22px);
            -webkit-backdrop-filter: saturate(180%) blur(22px);
        }

        [data-testid="stToolbar"] {
            color: var(--apple-text);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--apple-text);
            letter-spacing: -0.035em;
            text-wrap: balance;
        }

        h1 {
            font-size: clamp(2.55rem, 5vw, 4.85rem);
            line-height: 0.98;
            font-weight: 760;
        }

        h2 {
            margin-top: 2.5rem;
            font-size: clamp(1.75rem, 2.8vw, 2.7rem);
            line-height: 1.08;
            font-weight: 720;
        }

        h3 {
            font-size: clamp(1.25rem, 1.8vw, 1.7rem);
            font-weight: 680;
        }

        p, label, [data-testid="stMarkdownContainer"] {
            letter-spacing: -0.008em;
        }

        [data-testid="stCaptionContainer"], .stCaption {
            color: var(--apple-secondary);
        }

        hr {
            margin: 2.4rem 0;
            border-color: var(--apple-separator) !important;
        }

        /* Hero + first Bento row */
        .apple-hero {
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1.65fr) minmax(260px, 0.65fr);
            min-height: 330px;
            margin: 0.25rem 0 1.25rem;
            padding: clamp(2rem, 5vw, 4.75rem);
            overflow: hidden;
            border-radius: 36px;
            background:
                linear-gradient(130deg, rgba(255, 255, 255, 0.98), rgba(250, 252, 255, 0.88));
            box-shadow: var(--apple-shadow);
            isolation: isolate;
        }

        .apple-hero::before {
            content: "";
            position: absolute;
            inset: auto -8% -68% auto;
            width: min(56vw, 680px);
            aspect-ratio: 1;
            border-radius: 50%;
            background: radial-gradient(circle at 36% 32%, #6bc6ff, #0071e3 45%, #0045a5 74%);
            filter: blur(1px);
            opacity: 0.94;
            z-index: -2;
        }

        .apple-hero::after {
            content: "";
            position: absolute;
            right: 5%;
            bottom: -28%;
            width: min(34vw, 420px);
            aspect-ratio: 1;
            border-radius: 50%;
            border: 1px solid rgba(255, 255, 255, 0.48);
            box-shadow: inset 0 0 70px rgba(255, 255, 255, 0.22),
                        0 0 90px rgba(0, 113, 227, 0.22);
            z-index: -1;
        }

        .apple-hero-copy {
            align-self: center;
            max-width: 800px;
        }

        .apple-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 1.25rem;
            color: var(--apple-blue);
            font-size: 0.75rem;
            font-weight: 760;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .apple-live-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: #34c759;
            box-shadow: 0 0 0 5px rgba(52, 199, 89, 0.11);
        }

        .apple-hero h1 {
            max-width: 760px;
            margin: 0;
        }

        .apple-hero h1 span {
            color: var(--apple-secondary);
        }

        .apple-hero p {
            max-width: 650px;
            margin: 1.5rem 0 0;
            color: var(--apple-secondary);
            font-size: clamp(1rem, 1.45vw, 1.28rem);
            line-height: 1.62;
        }

        .apple-context-grid {
            display: grid;
            grid-template-columns: 1.25fr 1fr 1fr;
            gap: 1rem;
            margin: 0 0 1.6rem;
        }

        .apple-context-card {
            position: relative;
            min-height: 134px;
            padding: 1.45rem 1.6rem;
            overflow: hidden;
            border-radius: var(--apple-radius);
            background: var(--apple-surface);
            box-shadow: var(--apple-shadow);
            backdrop-filter: blur(20px) saturate(150%);
            -webkit-backdrop-filter: blur(20px) saturate(150%);
            transition: transform 520ms var(--apple-ease),
                        box-shadow 520ms var(--apple-ease),
                        background-color 520ms var(--apple-ease);
        }

        .apple-context-card:hover {
            transform: translateY(-6px) scale(1.008);
            box-shadow: var(--apple-shadow-hover);
            background: #ffffff;
        }

        .apple-context-card--accent {
            background: linear-gradient(145deg, #087be8, #0062c7);
            color: white;
        }

        .apple-context-card--accent:hover {
            background: linear-gradient(145deg, #1284ee, #0069d6);
        }

        .apple-context-label {
            display: block;
            margin-bottom: 0.7rem;
            color: var(--apple-tertiary);
            font-size: 0.72rem;
            font-weight: 720;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .apple-context-card--accent .apple-context-label {
            color: rgba(255, 255, 255, 0.72);
        }

        .apple-context-value {
            display: block;
            font-size: clamp(1.15rem, 2vw, 1.6rem);
            font-weight: 690;
            letter-spacing: -0.035em;
            line-height: 1.16;
        }

        .apple-context-note {
            display: block;
            margin-top: 0.45rem;
            color: var(--apple-secondary);
            font-size: 0.84rem;
        }

        .apple-context-card--accent .apple-context-note {
            color: rgba(255, 255, 255, 0.72);
        }

        /* Glass tab navigation */
        div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
            position: sticky;
            top: 3.15rem;
            z-index: 90;
            gap: 0.35rem;
            margin: 0 -0.35rem 2rem;
            padding: 0.45rem;
            overflow-x: auto;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.70);
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.055);
            backdrop-filter: saturate(180%) blur(24px);
            -webkit-backdrop-filter: saturate(180%) blur(24px);
            scrollbar-width: none;
        }

        div[data-testid="stTabs"] > div[data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }

        button[data-baseweb="tab"] {
            min-height: 42px;
            padding: 0.55rem 1rem !important;
            border-radius: 14px;
            color: var(--apple-secondary);
            font-size: 0.89rem;
            font-weight: 590;
            letter-spacing: -0.012em;
            transition: color 360ms var(--apple-ease),
                        background-color 360ms var(--apple-ease),
                        transform 360ms var(--apple-spring);
        }

        button[data-baseweb="tab"]:hover {
            color: var(--apple-text);
            background: rgba(0, 0, 0, 0.035);
            transform: translateY(-1px);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--apple-text);
            background: #ffffff;
            box-shadow: 0 5px 16px rgba(0, 0, 0, 0.07);
        }

        div[data-baseweb="tab-highlight"] {
            display: none;
        }

        div[data-baseweb="tab-panel"] {
            padding-top: 0.15rem;
        }

        /* Bento cards: metrics, charts, tables and bordered containers */
        [data-testid="stHorizontalBlock"] {
            gap: 1rem;
            align-items: stretch;
        }

        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stPyplotGlobalUse"],
        div[data-testid="stImage"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 0 !important;
            border-radius: var(--apple-radius) !important;
            background: var(--apple-surface-solid);
            box-shadow: var(--apple-shadow);
            transition: transform 520ms var(--apple-ease),
                        box-shadow 520ms var(--apple-ease);
        }

        div[data-testid="stMetric"]:hover,
        div[data-testid="stDataFrame"]:hover,
        div[data-testid="stVegaLiteChart"]:hover,
        div[data-testid="stPlotlyChart"]:hover,
        div[data-testid="stPyplotGlobalUse"]:hover,
        div[data-testid="stImage"]:hover,
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-5px);
            box-shadow: var(--apple-shadow-hover);
        }

        div[data-testid="stMetric"] {
            min-height: 132px;
            padding: 1.45rem 1.55rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--apple-secondary);
            font-size: 0.84rem;
            font-weight: 590;
        }

        [data-testid="stMetricValue"] {
            color: var(--apple-text);
            font-size: clamp(1.55rem, 2.5vw, 2.3rem);
            font-weight: 710;
            letter-spacing: -0.045em;
        }

        [data-testid="stMetricDelta"] {
            font-weight: 600;
        }

        div[data-testid="stDataFrame"] {
            padding: 0.55rem;
            overflow: hidden;
        }

        div[data-testid="stVegaLiteChart"] {
            padding: clamp(1rem, 2vw, 1.5rem);
            overflow: visible;
        }

        div[data-testid="stPlotlyChart"],
        div[data-testid="stPyplotGlobalUse"],
        div[data-testid="stImage"] {
            padding: clamp(0.75rem, 2vw, 1.35rem);
            overflow: hidden;
        }

        canvas, svg, img {
            border-radius: calc(var(--apple-radius) - 8px);
        }

        /* Expanders */
        details[data-testid="stExpander"] {
            overflow: hidden;
            border: 0;
            border-radius: var(--apple-radius);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 10px 34px rgba(0, 0, 0, 0.045);
            transition: box-shadow 420ms var(--apple-ease),
                        transform 420ms var(--apple-ease);
        }

        details[data-testid="stExpander"]:hover {
            transform: translateY(-3px);
            box-shadow: var(--apple-shadow);
        }

        details[data-testid="stExpander"] summary {
            padding: 0.35rem 0.65rem;
            font-weight: 610;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFileUploader"] button {
            min-height: 44px;
            padding: 0.62rem 1.15rem;
            border: 0 !important;
            border-radius: 999px !important;
            color: var(--apple-text);
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 5px 16px rgba(0, 0, 0, 0.065);
            font-weight: 630;
            letter-spacing: -0.012em;
            transition: transform 360ms var(--apple-spring),
                        box-shadow 360ms var(--apple-ease),
                        background-color 360ms var(--apple-ease),
                        color 360ms var(--apple-ease);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFileUploader"] button:hover {
            transform: translateY(-2px) scale(1.015);
            background: #ffffff;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.10);
        }

        .stButton > button:active,
        .stDownloadButton > button:active,
        [data-testid="stFileUploader"] button:active {
            transform: scale(0.965);
            transition-duration: 100ms;
        }

        .stButton > button[kind="primary"] {
            color: #ffffff;
            background: var(--apple-blue);
            box-shadow: 0 9px 24px rgba(0, 113, 227, 0.23);
        }

        .stButton > button[kind="primary"]:hover {
            color: #ffffff;
            background: var(--apple-blue-hover);
            box-shadow: 0 13px 30px rgba(0, 113, 227, 0.30);
        }

        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible {
            outline: 3px solid rgba(0, 113, 227, 0.25) !important;
            outline-offset: 2px;
        }

        /* Inputs */
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="select"] > div,
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            border-color: transparent !important;
            border-radius: 14px !important;
            background: rgba(118, 118, 128, 0.08) !important;
            box-shadow: none !important;
            transition: background-color 300ms var(--apple-ease),
                        box-shadow 300ms var(--apple-ease);
        }

        [data-baseweb="input"]:focus-within,
        [data-baseweb="select"] > div:focus-within {
            background: #ffffff !important;
            box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.16) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 0;
            border-radius: var(--apple-radius);
            background: rgba(118, 118, 128, 0.06);
        }

        /* Radio / segmented controls */
        [data-testid="stRadio"] > div {
            gap: 0.35rem;
            padding: 0.35rem;
            border-radius: 16px;
            background: rgba(118, 118, 128, 0.08);
        }

        /* Alerts and status */
        [data-testid="stAlert"] {
            border: 0;
            border-radius: var(--apple-radius-small);
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.035);
        }

        [data-testid="stProgress"] > div > div {
            overflow: hidden;
            border-radius: 999px;
            background: rgba(0, 113, 227, 0.10);
        }

        [data-testid="stProgress"] > div > div > div {
            border-radius: 999px;
            background: var(--apple-blue);
            transition: width 600ms var(--apple-ease);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            border-right: 0;
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 12px 0 42px rgba(0, 0, 0, 0.035);
            backdrop-filter: saturate(180%) blur(26px);
            -webkit-backdrop-filter: saturate(180%) blur(26px);
        }

        section[data-testid="stSidebar"] > div {
            background: transparent;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            letter-spacing: -0.025em;
        }

        /* Entry and scroll animations */
        @keyframes apple-fade-up {
            from { opacity: 0; transform: translateY(24px) scale(0.985); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes apple-scroll-reveal {
            from { opacity: 0; transform: translateY(30px) scale(0.98); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }

        @media (prefers-reduced-motion: no-preference) {
            .apple-hero {
                animation: apple-fade-up 800ms var(--apple-ease) both;
            }

            .apple-context-card {
                animation: apple-fade-up 760ms var(--apple-ease) both;
            }

            .apple-context-card:nth-child(1) { animation-delay: 80ms; }
            .apple-context-card:nth-child(2) { animation-delay: 150ms; }
            .apple-context-card:nth-child(3) { animation-delay: 220ms; }

            @supports (animation-timeline: view()) {
                div[data-testid="stMetric"],
                div[data-testid="stDataFrame"],
                div[data-testid="stVegaLiteChart"],
                div[data-testid="stPlotlyChart"],
                div[data-testid="stPyplotGlobalUse"],
                div[data-testid="stImage"],
                details[data-testid="stExpander"] {
                    animation: apple-scroll-reveal both;
                    animation-timeline: view();
                    animation-range: entry 2% entry 24%;
                }
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        @media (max-width: 900px) {
            .block-container {
                padding: 2rem 1.15rem 5rem;
            }

            .apple-hero {
                grid-template-columns: 1fr;
                min-height: 390px;
                padding: 2.3rem 1.65rem;
            }

            .apple-hero::before {
                right: -36%;
                bottom: -54%;
                width: 520px;
                opacity: 0.72;
            }

            .apple-hero::after {
                right: -12%;
                bottom: -10%;
                width: 260px;
            }

            .apple-context-grid {
                grid-template-columns: 1fr;
            }

            div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
                top: 3rem;
                border-radius: 17px;
            }
        }

        @media (max-width: 640px) {
            .apple-hero {
                min-height: 420px;
                border-radius: 28px;
            }

            .apple-context-card,
            div[data-testid="stMetric"],
            div[data-testid="stDataFrame"],
            div[data-testid="stVegaLiteChart"],
            div[data-testid="stPlotlyChart"],
            div[data-testid="stImage"] {
                border-radius: 22px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_apple_hero(market_date: str, report_period: str) -> None:
    """渲染顶部 Hero 与上下文 Bento 卡片。"""
    safe_market_date = escape(str(market_date))
    safe_report_period = escape(str(report_period))
    st.markdown(
        f"""
        <section class="apple-hero">
            <div class="apple-hero-copy">
                <div class="apple-eyebrow">
                    <span class="apple-live-dot"></span>
                    Market Intelligence
                </div>
                <h1>看见市场，<br><span>保持清晰。</span></h1>
            </div>
        </section>
        <section class="apple-context-grid" aria-label="当前数据上下文">
            <article class="apple-context-card apple-context-card--accent">
                <span class="apple-context-label">Workspace</span>
                <span class="apple-context-value">中国市场数据中心</span>
                <span class="apple-context-note">多源数据 · 统一视图 · 实时分析</span>
            </article>
            <article class="apple-context-card">
                <span class="apple-context-label">Market date</span>
                <span class="apple-context-value">{safe_market_date}</span>
                <span class="apple-context-note">当前市场统计日期</span>
            </article>
            <article class="apple-context-card">
                <span class="apple-context-label">Reporting period</span>
                <span class="apple-context-value">{safe_report_period}</span>
                <span class="apple-context-note">当前财务报告周期</span>
            </article>
        </section>
        """,
        unsafe_allow_html=True,
    )
