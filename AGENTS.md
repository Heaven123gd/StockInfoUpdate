# Repository Guidelines

## Project Structure & Module Organization

This repository is a Streamlit app for A-share market visualization. `app.py` is the UI entry point and owns sidebar inputs, tab layout, and `st.session_state`. `data_fetcher.py` contains AKShare calls, local Excel/CSV parsing, and numeric processing. `charts.py` renders Streamlit tables, metrics, and matplotlib charts. `config.py` centralizes symbols, colors, date defaults, and featured stock lists.

Committed data assets live at the repository root: `prepared_stock_data.csv` and `*.GI-行情统计-YYYYMMDD.xlsx`. Keep `fonts/SimHei.ttf`; `charts.py` uses it for Chinese chart labels in cloud environments. Streamlit theme settings are in `.streamlit/config.toml`, and Codespaces/devcontainer behavior is in `.devcontainer/devcontainer.json`.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

Run with Codespaces-friendly flags:

```bash
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

Basic syntax check:

```bash
python -m py_compile app.py data_fetcher.py charts.py config.py
```

There is no build step; the app runs directly from source.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and clear `snake_case` names. Keep constants in `UPPER_CASE` in `config.py`. Preserve the existing layering: UI orchestration in `app.py`, data acquisition and transformations in `data_fetcher.py`, rendering in `charts.py`. Follow existing prefixes such as `fetch_*`, `load_*`, `parse_*`, and `display_*`. UI text and many comments are Chinese; keep new user-facing labels consistent with that style.

## Testing Guidelines

No test suite is currently configured. For logic changes, at minimum run the `py_compile` command above and manually open the affected Streamlit tab. If adding tests, prefer `pytest` under `tests/` with files named `test_*.py`. Mock AKShare/network calls and test pure DataFrame transformations separately from Streamlit rendering.

## Commit & Pull Request Guidelines

Recent history uses short, outcome-focused subjects in English or Chinese, for example `modify the pie chart` and `Tab7性能优化：并行获取A股和美股数据，添加进度提示`. Use concise imperative or descriptive commit subjects that name the changed behavior.

Pull requests should include a brief summary, manual verification steps, and screenshots for visible UI/chart changes. Note any data-source changes, new local asset files, or AKShare behavior assumptions.

## Security & Configuration Tips

Do not commit `.env`, credentials, virtual environments, cache folders, or generated exports unless they are intentional project data. AKShare can be flaky; route new network calls through the existing retry pattern and surface recoverable failures to the UI instead of crashing the app.
