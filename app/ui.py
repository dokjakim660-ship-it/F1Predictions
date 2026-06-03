"""F1 Predictions — shared UI/design-system module.

This is the Python side of the redesign spec exported from Claude Design.

  Tokens     -> .streamlit/config.toml   (native widget + dataframe theming)
  Components -> this module               (CSS injection + header/table helpers)
  Charts     -> ui.CHART palette + apply  (one consistent Altair look)

Everything here is pure Python/Streamlit — no JS framework, HF-Docker safe.
`inject()` also carries the anti-reflow CSS block that keeps the HF Spaces
iframe from entering its resize loop; it MUST run once, early, on every page.

Theme toggle
------------
The app ships dark by default (config.toml). A sidebar toggle flips a runtime
*chrome* theme: backgrounds, sidebar, headers, cards, native widgets and charts
all repaint via CSS variables. The one thing it cannot repaint is the
canvas-rendered st.dataframe / st.data_editor grid — its interior colors come
from config.toml, which CSS cannot reach. By design choice those grids stay a
dark "data panel" in both modes, which reads as intentional rather than broken.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import streamlit as st

FONT_UI = (
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif'
)
FONT_MONO = '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace'

# Tables stay dark in both themes (see module docstring) — these match the
# config.toml dark surface so the wrapper blends with the canvas grid.
TBL_SURFACE = "#0e1626"
TBL_BORDER = "#1e2c46"

# ============================================================
# PALETTES — full light/dark parity (mirror of the design tokens)
# ============================================================
_DARK = {
    "bg": "#080d18",
    "bg_grad": "radial-gradient(1200px 700px at 78% -8%, #0e1830 0%, rgba(8,13,24,0) 55%)",
    "sidebar": "#0a1120",
    "surface": "#0e1626",
    "surface_2": "#121d31",
    "surface_3": "#16233b",
    "border": "#1e2c46",
    "border_2": "#2a3a58",
    "text": "#eef3fc",
    "text_muted": "#94a3bd",
    "text_faint": "#5d6c89",
    "accent": "#3b82f6",
    "accent_2": "#60a5fa",
    "accent_bg": "rgba(59,130,246,.14)",
    "good": "#34d399",
    "bad": "#f87171",
    "warn": "#fbbf24",
    "grid": "rgba(148,163,189,.18)",
    "shadow": "0 1px 0 rgba(255,255,255,.03), 0 8px 24px -12px rgba(0,0,0,.6)",
}
_LIGHT = {
    "bg": "#f4f6fa",
    "bg_grad": "radial-gradient(1200px 700px at 80% -10%, #e9eef7 0%, rgba(244,246,250,0) 55%)",
    "sidebar": "#ffffff",
    "surface": "#ffffff",
    "surface_2": "#f6f8fb",
    "surface_3": "#eef2f8",
    "border": "#e4e9f1",
    "border_2": "#d4dbe7",
    "text": "#0c1320",
    "text_muted": "#5a6679",
    "text_faint": "#8a97ac",
    "accent": "#2563eb",
    "accent_2": "#1d4ed8",
    "accent_bg": "rgba(37,99,235,.09)",
    "good": "#0f9d63",
    "bad": "#dc2626",
    "warn": "#b7791f",
    "grid": "rgba(12,19,32,.10)",
    "shadow": "0 1px 2px rgba(16,24,40,.04), 0 12px 28px -16px rgba(16,24,40,.18)",
}
_PALETTES = {"dark": _DARK, "light": _LIGHT}

# ============================================================
# MUTABLE CHART COLOR GLOBALS — updated by inject() per theme so the page
# modules (which read ui.ACCENT / ui.GOOD / ... at render time) follow the
# active theme. Default to dark for any import-time use.
# ============================================================
ACCENT = _DARK["accent"]
ACCENT_2 = _DARK["accent_2"]
GOOD = _DARK["good"]
BAD = _DARK["bad"]
WARN = _DARK["warn"]
TEXT = _DARK["text"]
TEXT_MUTED = _DARK["text_muted"]
TEXT_FAINT = _DARK["text_faint"]
GRID = _DARK["grid"]
CHART: dict[str, Any] = {}


def _apply_globals(p: dict[str, str]) -> None:
    global ACCENT, ACCENT_2, GOOD, BAD, WARN, TEXT, TEXT_MUTED, TEXT_FAINT, GRID, CHART
    ACCENT, ACCENT_2 = p["accent"], p["accent_2"]
    GOOD, BAD, WARN = p["good"], p["bad"], p["warn"]
    TEXT, TEXT_MUTED, TEXT_FAINT = p["text"], p["text_muted"], p["text_faint"]
    GRID = p["grid"]
    CHART = {
        "accent": ACCENT, "good": GOOD, "bad": BAD, "warn": WARN, "muted": TEXT_FAINT,
        "range": [ACCENT, GOOD, WARN, BAD, "#a78bfa", "#22d3ee", "#fb923c"],
    }


# ============================================================
# TEAM COLOR SYSTEM (2026 grid) — thin identity accent only
# ============================================================
_TEAM_COLORS: list[tuple[tuple[str, ...], str]] = [
    (("mclaren",), "#FF8000"),
    (("ferrari",), "#E8002D"),
    (("mercedes",), "#27F4D2"),
    (("red bull", "redbull"), "#3671C6"),
    (("williams",), "#1868DB"),
    (("aston",), "#229971"),
    (("alpine",), "#00A1E8"),
    (("racing bull", "visa", "alphatauri", "rb f1", "scuderia rb"), "#6C7BFF"),
    (("audi", "sauber"), "#B91D33"),
    (("haas",), "#9CA3AD"),
    (("cadillac",), "#C9A24B"),
]


def team_color(name: str | None) -> str:
    """Constructor identity color, or a neutral border tone if unknown."""
    if not name:
        return TBL_BORDER
    low = str(name).lower()
    for keys, color in _TEAM_COLORS:
        if any(k in low for k in keys):
            return color
    return "#9CA3AD"


# ============================================================
# THEME STATE
# ============================================================
_THEME_KEY = "f1_theme"


def current_theme() -> str:
    """Active chrome theme — 'dark' (default) or 'light'."""
    return st.session_state.get(_THEME_KEY, "dark")


def theme_toggle() -> None:
    """Render the Dark/Light switch (call inside the sidebar). Writes the choice
    to session_state under _THEME_KEY; the next run's inject() reads it."""
    cur = current_theme()
    choice = st.segmented_control(
        "Theme",
        options=["dark", "light"],
        default=cur,
        format_func=lambda v: ("🌙 Dark" if v == "dark" else "☀️ Light"),
        key="f1_theme_choice",
        label_visibility="collapsed",
    )
    new = choice or cur
    if new != cur:
        st.session_state[_THEME_KEY] = new
        st.rerun()
    st.session_state[_THEME_KEY] = new


# ============================================================
# GLOBAL CSS — anti-reflow block + design-system polish
# ============================================================
def _css(p: dict[str, str], theme: str) -> str:
    light = theme == "light"
    # In light mode the compiled Streamlit theme is still dark (config.toml is
    # the single source for the canvas grid), so we repaint native widget text
    # and surfaces here. In dark mode config already matches — no overrides.
    light_overrides = (
        f"""
/* ---- LIGHT MODE: repaint native widgets (config theme is dark) ---- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ color: {p['text']}; }}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{ color: {p['text']}; }}
h1, h2, h3, h4, h5, h6 {{ color: {p['text']}; }}
[data-testid="stCaptionContainer"], small {{ color: {p['text_faint']} !important; }}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{ color: {p['text_muted']}; }}
/* segmented-control / radio option text */
[data-testid="stRadio"] [role="radiogroup"] label,
[data-testid="stRadio"] [role="radiogroup"] label p {{ color: {p['text']}; }}
[data-testid="stSidebar"] {{ color: {p['text']}; }}
[data-testid="stSidebarNav"] a span {{ color: {p['text_muted']}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: {p['text']}; }}
/* inputs */
input, textarea, [data-baseweb="input"], [data-baseweb="select"] > div {{
  background: {p['surface_2']} !important; color: {p['text']} !important;
  border-color: {p['border_2']} !important;
}}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {{
  background: {p['surface']} !important; color: {p['text']} !important;
}}
[role="option"] {{ color: {p['text']} !important; }}
/* slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background: {p['accent']}; }}
/* divider */
hr {{ border-color: {p['border']}; }}
"""
        if light
        else ""
    )

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
  --bg: {p['bg']}; --bg-grad: {p['bg_grad']}; --sidebar: {p['sidebar']};
  --surface: {p['surface']}; --surface-2: {p['surface_2']}; --surface-3: {p['surface_3']};
  --border: {p['border']}; --border-2: {p['border_2']};
  --text: {p['text']}; --text-muted: {p['text_muted']}; --text-faint: {p['text_faint']};
  --accent: {p['accent']}; --accent-2: {p['accent_2']}; --accent-bg: {p['accent_bg']};
  --good: {p['good']}; --bad: {p['bad']}; --warn: {p['warn']};
  --shadow: {p['shadow']};
  --font-mono: {FONT_MONO};
  --r-sm: 6px; --r-md: 10px; --r-lg: 14px;
}}

/* ============================================================
   (A) ANTI-REFLOW — keep verbatim. Stops the HF Spaces iframe
   resize loop. See streamlit_app.py history for the full rationale.
   ============================================================ */
html {{ overflow-y: scroll !important; }}
.vega-embed details, .vega-embed summary {{ display: none !important; }}
#vg-tooltip-element {{ position: fixed !important; pointer-events: none !important; z-index: 9999 !important; }}
[data-testid="stVegaLiteChart"], [data-testid="stAltairChart"] {{ contain: layout style !important; transition: none !important; }}
[data-testid="stVegaLiteChart"] *, [data-testid="stAltairChart"] * {{ transition: none !important; animation: none !important; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"] {{ contain: layout style !important; }}
[data-testid="stVegaLiteChart"] {{ height: auto !important; min-height: 1px; }}

/* ============================================================
   (B) DESIGN SYSTEM — restyle native Streamlit DOM to the spec
   ============================================================ */
.stApp {{ background: var(--bg-grad), var(--bg); }}
html, body, [data-testid="stAppViewContainer"] {{ font-family: {FONT_UI}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ max-width: 920px; padding-top: 2.4rem; padding-bottom: 4rem; }}
h1, h2, h3 {{ letter-spacing: -.02em; }}
[data-testid="stMarkdownContainer"] h3 {{ font-weight: 700; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{ background: var(--sidebar); border-right: 1px solid var(--border); }}
[data-testid="stSidebarNav"] {{ padding-top: .25rem; }}
[data-testid="stSidebarNav"] a {{ border-radius: var(--r-sm); }}
[data-testid="stSidebarNav"] a:hover {{ background: var(--surface-2); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: var(--accent-bg); box-shadow: inset 3px 0 0 var(--accent); }}

/* ---- brand header (ui.app_header) ---- */
.f1-brand {{ display:flex; align-items:center; gap:12px; margin: 0 0 6px; }}
.f1-brand .mark {{
  width:38px; height:38px; border-radius:10px; display:grid; place-items:center;
  font-size:20px; flex:none; background:linear-gradient(150deg, var(--surface-3), var(--surface-2));
  border:1px solid var(--border);
}}
.f1-brand .bn {{ font-weight:800; font-size:17px; letter-spacing:-.02em; line-height:1.1; color:var(--text); }}
.f1-brand .bs {{ font-size:11px; color:var(--text-faint); font-family:var(--font-mono); letter-spacing:.02em; }}
.f1-pill {{
  display:inline-flex; align-items:center; gap:7px; font-size:11px; font-weight:600;
  color:var(--text-muted); background:var(--surface-2); border:1px solid var(--border);
  padding:5px 11px; border-radius:999px; font-family:var(--font-mono);
}}
.f1-pill .dot {{ width:7px; height:7px; border-radius:50%; background:var(--good); box-shadow:0 0 0 3px var(--accent-bg); }}

/* ---- page header (ui.page_header) ---- */
.f1-eyebrow {{ font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--accent-2); font-family:var(--font-mono); margin-bottom:6px; }}
.f1-title {{ font-size:24px; font-weight:800; letter-spacing:-.025em; line-height:1.1; margin:0; color:var(--text); }}
.f1-desc {{ color:var(--text-muted); font-size:14px; margin:8px 0 0; max-width:66ch; }}

/* ---- section header (ui.section) ---- */
.f1-section {{ display:flex; align-items:baseline; gap:12px; margin:6px 0 2px; }}
.f1-section h4 {{ font-size:17px; font-weight:700; letter-spacing:-.02em; margin:0; color:var(--text); }}
.f1-section .sub {{ font-size:12.5px; color:var(--text-faint); }}
.f1-section .rule {{ flex:1; height:1px; background:var(--border); align-self:center; }}

/* ---- metric cards ---- */
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--r-md); padding:14px 16px 12px; position:relative; overflow:hidden; box-shadow:var(--shadow); }}
[data-testid="stMetric"]::after {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--accent); opacity:.55; }}
[data-testid="stMetricLabel"] p {{ font-size:11px !important; font-weight:600; color:var(--text-faint); text-transform:uppercase; letter-spacing:.07em; }}
[data-testid="stMetricValue"] {{ font-family:var(--font-mono); letter-spacing:-.02em; font-weight:600; color:var(--text); }}

/* ---- horizontal radios -> segmented control ---- */
[data-testid="stRadio"] [role="radiogroup"] {{ flex-direction:row; flex-wrap:wrap; gap:6px; background:var(--surface-2); border:1px solid var(--border); border-radius:var(--r-md); padding:4px; width:fit-content; }}
[data-testid="stRadio"] [role="radiogroup"] label {{ margin:0; padding:5px 12px; border-radius:7px; cursor:pointer; transition:background .12s, color .12s; }}
[data-testid="stRadio"] [role="radiogroup"] label:hover {{ background:var(--surface-3); }}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{ background:var(--surface); box-shadow:var(--shadow); }}
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{ display:none; }}

/* ---- buttons ---- */
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {{ border-radius:var(--r-sm); font-weight:600; }}

/* ---- expander as a card ---- */
[data-testid="stExpander"] details {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--r-lg); }}

/* ---- alerts ---- */
[data-testid="stAlert"] {{ border-radius:var(--r-md); border:1px solid var(--border); }}

/* ---- TABLES: always a dark data panel (both themes — see module docstring) ---- */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
  background:{TBL_SURFACE} !important; border:1px solid {TBL_BORDER} !important;
  border-radius:var(--r-lg); box-shadow:var(--shadow);
}}

/* ---- scrollbars: stable gutter ---- */
::-webkit-scrollbar {{ width:10px; height:10px; }}
::-webkit-scrollbar-thumb {{ background:var(--border-2); border-radius:999px; border:2px solid transparent; background-clip:padding-box; }}
::-webkit-scrollbar-track {{ background:transparent; }}
{light_overrides}
</style>
"""


def inject(theme: str | None = None) -> None:
    """Inject the global stylesheet (anti-reflow + design system) for the active
    theme and sync the chart-color globals. Call once, early, per page run."""
    theme = theme or current_theme()
    p = _PALETTES.get(theme, _DARK)
    _apply_globals(p)
    st.markdown(_css(p, theme), unsafe_allow_html=True)


# ============================================================
# COMPONENT HELPERS
# ============================================================
def app_header(title: str, subtitle: str, status: str | None = None) -> None:
    """Branded wordmark header for the top of the main content column."""
    pill = f'<span class="f1-pill"><span class="dot"></span>{status}</span>' if status else ""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
          <div class="f1-brand">
            <div class="mark">🏎️</div>
            <div><div class="bn">{title}</div><div class="bs">{subtitle}</div></div>
          </div>
          {pill}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, eyebrow: str | None = None, desc: str | None = None) -> None:
    """Consistent page header: accent eyebrow + bold title + muted description."""
    parts = ['<div style="margin-bottom:18px;">']
    if eyebrow:
        parts.append(f'<div class="f1-eyebrow">{eyebrow}</div>')
    parts.append(f'<h1 class="f1-title">{title}</h1>')
    if desc:
        parts.append(f'<p class="f1-desc">{desc}</p>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, sub: str | None = None) -> None:
    """Section header with a hairline rule — replaces bare st.subheader."""
    sub_html = f'<span class="sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="f1-section"><h4>{title}</h4>{sub_html}<span class="rule"></span></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TABLE HELPERS
# ============================================================
def pct(series: Any) -> Any:
    """Scale a 0..1 probability Series to 0..100 for prob_column (which renders
    the data-bar AND its % label from the same number, so it must be a percent)."""
    return (series.astype(float) * 100.0).round(1)


def prob_column(label: str, help: str | None = None) -> Any:
    """A percentage column (0..100) rendered as an inline data-bar — the design's
    table upgrade. Feed it values from ui.pct(), not raw 0..1 probabilities."""
    return st.column_config.ProgressColumn(
        label, help=help, min_value=0.0, max_value=100.0, format="%.0f%%"
    )


def team_styler(df: Any, team_col: str = "team") -> Any:
    """Return a pandas Styler that tints the team cell with its identity color.
    Tables stay dark in both themes, so team colors read consistently."""
    def _team_fg(val: Any) -> str:
        return f"color:{team_color(val)}; font-weight:600;"

    styler = df.style
    if team_col in df.columns:
        styler = styler.map(_team_fg, subset=[team_col])
    return styler


# ============================================================
# ALTAIR — one shared theme (axes/grid/font/legend/tooltip)
# ============================================================
def _altair_theme_dict() -> dict:
    return {
        "config": {
            "background": "transparent",
            "font": FONT_UI,
            "view": {"stroke": "transparent"},
            "axis": {
                "labelFont": FONT_MONO, "labelColor": TEXT_FAINT, "labelFontSize": 10,
                "titleFont": FONT_UI, "titleColor": TEXT_MUTED, "titleFontSize": 11,
                "titleFontWeight": 600, "gridColor": GRID, "gridWidth": 1,
                "domainColor": GRID, "tickColor": GRID,
            },
            "legend": {
                "labelFont": FONT_UI, "labelColor": TEXT_MUTED, "titleFont": FONT_UI,
                "titleColor": TEXT_FAINT, "labelFontSize": 11,
            },
            "range": {"category": CHART.get("range", [ACCENT])},
            "mark": {"color": ACCENT},
            "bar": {"color": ACCENT},
            "line": {"color": ACCENT, "strokeWidth": 2.4},
            "point": {"color": ACCENT, "filled": True},
        }
    }


def enable_altair_theme() -> None:
    """Register + enable the shared chart theme for the active palette. Re-runs
    each script pass so a theme toggle recolors every chart."""
    name = "f1"
    try:  # altair >= 5.5 / 6.x
        alt.theme.register(name, enable=True)(_altair_theme_dict)
    except (AttributeError, TypeError):  # older altair
        alt.themes.register(name, _altair_theme_dict)
        alt.themes.enable(name)


def altair_chart(chart: Any, **kwargs: Any) -> None:
    """st.altair_chart wrapper that lets our registered theme win over
    Streamlit's built-in chart theme and keeps container-width default."""
    kwargs.setdefault("use_container_width", True)
    kwargs.setdefault("theme", None)
    st.altair_chart(chart, **kwargs)
