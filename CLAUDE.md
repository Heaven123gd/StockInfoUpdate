# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A股市场数据可视化工具 — a Streamlit web app that fetches Chinese (and some US) stock-market data via [AKShare](https://akshare.akshare.xyz/) and renders interactive analysis with matplotlib. UI text and most code comments are in Chinese. Single-process app, no backend service, no database — all state lives in `st.session_state`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app (default port 8501)
streamlit run app.py

# Run as configured in the devcontainer (disables CORS/XSRF for Codespaces preview)
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

There is no test suite, linter config, or build step. To sanity-check data logic, import the function from `data_fetcher.py` in a Python REPL and call it directly — every fetch function takes plain args (dates as `YYYYMMDD` strings) and returns DataFrames, independent of Streamlit.

## Architecture

Three modules, strict layering — UI never fetches, fetchers never render:

- **`app.py`** — Streamlit entry point. Defines the sidebar (date/report-period inputs, Dow Jones file uploaders) and 7 tabs. Orchestrates loading: calls `data_fetcher` functions, stashes results in `st.session_state`, then hands them to `charts` display functions. This is the only file that touches `st.session_state`.
- **`data_fetcher.py`** — All data acquisition and numeric processing (AKShare calls, local Excel parsing, CPI inflation adjustment, OLS regression, normalization). Pure-ish: returns `(DataFrame, errors_dict)` tuples or `(df, stats, merged)`. No Streamlit imports.
- **`charts.py`** — All rendering (`st.dataframe`, `st.metric`, matplotlib figures). Functions are named `display_*`; private `_*` helpers do the plotting. Sets up the SimHei Chinese font for matplotlib at import time.
- **`config.py`** — Constants: index symbol maps, color schemes, industry-classification keywords, report-date logic, featured stock lists. Imported by both other modules.

### Tab dependency chain (important)

Tabs use **lazy loading** — each shows a "📥 加载" button and only fetches when clicked (or when the sidebar "🔄 刷新所有数据" `refresh_btn` is pressed). Several tabs depend on data loaded by an earlier tab via shared `st.session_state` keys:

- Tab 3 (利润分布) needs `st.session_state.merged_data` from **Tab 2** (净利润100强).
- Tabs 5 (概率分布) and 6 (指数对比) need `st.session_state.index_data` from **Tab 4** (指数走势).

If you add/rename a session_state key, check every tab that reads it. Common keys: `sse_data`, `szse_data`, `profit_top100`/`merged_data`, `index_data`/`index_errors`, `adjusted_data`/`regression_results`, `comparison_data`, `featured_stocks_data`, `us_stocks_data`.

### Tab 7 preload pattern

Tab 7 (主要股票数据) loads fast on first render by reading `prepared_stock_data.csv` / `prepared_us_stock_data.csv` (committed CSVs), and only hits the AKShare API when the user clicks "🔄 刷新最新数据". `featured_stocks_source` / `us_stocks_source` session keys track whether displayed data is `preload` vs `api`. `fetch_all_featured_stocks_data` parallelizes per-stock fetches with a `ThreadPoolExecutor` (max 4 workers).

### Index data: AKShare vs local Excel

Indices come from two sources, configured in `config.py`:
- `INDEX_SYMBOLS_AKSHARE` (上证/深证/深圳综指) — fetched live via `ak.stock_zh_index_daily`.
- `INDEX_SYMBOLS_LOCAL` + `DJI_SYMBOL` (the four Dow Jones指数) — read from committed `*.GI-行情统计-YYYYMMDD.xlsx` files via `find_local_index_file`. Users can override these by uploading files in the sidebar, which `parse_uploaded_index_file` parses into the same DataFrame shape; uploaded data lives in `uploaded_dji_data` / `uploaded_dj{china,sh,sz}_data` session keys and replaces the local default in Tabs 4/6.

## Conventions

- **Network resilience**: AKShare calls are flaky from cloud hosts. Wrap them with `retry_request(func, max_retries, delay)` (exponential-ish backoff). `AKSHARE_TIMEOUT=60` is set at the top of `app.py`. Always return errors as `(None/empty_df, errors_dict)` rather than raising — the UI renders `errors_dict` as warnings.
- **Chinese fonts**: matplotlib needs `fonts/SimHei.ttf` (registered in `charts.py`) to render Chinese labels in cloud deployments where system fonts are absent. Don't remove it.
- **Date formats**: report/market dates are `YYYYMMDD` strings throughout; the comparison/normalization baseline `COMPARISON_START_DATE` is `1994-01-03`.
- **Financial columns** use AKShare's nested naming, e.g. `净利润-净利润`, `净利润-同比增长` — match these exactly when filtering.
