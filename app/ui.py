"""F1 Predictions — shared UI/design-system module.

This is the Python side of the redesign spec exported from Claude Design.

  Tokens     -> .streamlit/config.toml   (native widget + dataframe theming)
  Components -> this module               (CSS injection + header/table helpers)
  Charts     -> ui.CHART palette + apply  (one consistent Altair look)

Everything here is pure Python/Streamlit — no JS framework, HF-Docker safe.
`inject()` also carries the anti-reflow CSS block that keeps the HF Spaces
iframe from entering its resize loop; it MUST run once, early, on every page.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import streamlit as st

# ============================================================
# DESIGN TOKENS (mirror of .streamlit/config.toml + ui CSS vars)
# ============================================================
BG = "#080d18"
SURFACE = "#0e1626"
SURFACE_2 = "#121d31"
SURFACE_3 = "#16233b"
BORDER = "#1e2c46"
TEXT = "#eef3fc"
TEXT_MUTED = "#94a3bd"
TEXT_FAINT = "#5d6c89"

ACCENT = "#3b82f6"
ACCENT_2 = "#60a5fa"
GOOD = "#34d399"
BAD = "#f87171"
WARN = "#fbbf24"
GRID = "rgba(148,163,189,.18)"

FONT_UI = (
    '-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif'
)
FONT_MONO = '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace'

# Chart palette — used in place of matplotlib's default tab10 so every page's
# Altair charts share one language: accent for the hero series, good/bad for
# semantic up/down, faint for baselines.
CHART = {
    "accent": ACCENT,
    "good": GOOD,
    "bad": BAD,
    "warn": WARN,
    "muted": TEXT_FAINT,
    # Categorical range for multi-series charts (e.g. model comparison).
    "range": ["#3b82f6", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#22d3ee", "#fb923c"],
}

# ============================================================
# TEAM COLOR SYSTEM (2026 grid) — thin identity accent only
# ============================================================
# Keyed by a substring of the constructor display name so it survives the
# small naming differences between Jolpica / FastF1 ("Red Bull Racing" vs
# "Red Bull", "Sauber" vs "Audi", "RB" vs "Racing Bulls", …).
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
        return BORDER
    low = str(name).lower()
    for keys, color in _TEAM_COLORS:
        if any(k in low for k in keys):
            return color
    return BORDER


# ============================================================
# GLOBAL CSS — anti-reflow block + design-system polish
# ============================================================
def _css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
  --accent: {ACCENT}; --accent-2: {ACCENT_2};
  --good: {GOOD}; --bad: {BAD}; --warn: {WARN};
  --surface: {SURFACE}; --surface-2: {SURFACE_2}; --surface-3: {SURFACE_3};
  --border: {BORDER}; --text: {TEXT}; --text-muted: {TEXT_MUTED}; --text-faint: {TEXT_FAINT};
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
.stApp {{
  background:
    radial-gradient(1200px 700px at 78% -8%, #0e1830 0%, rgba(8,13,24,0) 55%),
    {BG};
}}
html, body, [data-testid="stAppViewContainer"] {{ font-family: {FONT_UI}; }}

/* slim, transparent default header */
[data-testid="stHeader"] {{ background: transparent; }}

/* content column — single readable, narrow-friendly column */
[data-testid="stMainBlockContainer"] {{
  max-width: 920px; padding-top: 2.4rem; padding-bottom: 4rem;
}}

/* number-ish runs render with tabular mono via st.dataframe theme; here we
   just make headings tight + confident */
h1, h2, h3 {{ letter-spacing: -.02em; }}
[data-testid="stMarkdownContainer"] h3 {{ font-weight: 700; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
[data-testid="stSidebarNav"] {{ padding-top: .25rem; }}
[data-testid="stSidebarNav"] a {{ border-radius: var(--r-sm); }}
[data-testid="stSidebarNav"] a:hover {{ background: var(--surface-2); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
  background: rgba(59,130,246,.14);
  box-shadow: inset 3px 0 0 var(--accent);
}}

/* ---- brand header (ui.app_header) ---- */
.f1-brand {{ display:flex; align-items:center; gap:12px; margin: 0 0 6px; }}
.f1-brand .mark {{
  width:38px; height:38px; border-radius:10px; display:grid; place-items:center;
  font-size:20px; flex:none;
  background:linear-gradient(150deg, var(--surface-3), var(--surface-2));
  border:1px solid var(--border);
}}
.f1-brand .bn {{ font-weight:800; font-size:17px; letter-spacing:-.02em; line-height:1.1; }}
.f1-brand .bs {{ font-size:11px; color:var(--text-faint); font-family:var(--font-mono); letter-spacing:.02em; }}
.f1-pill {{
  display:inline-flex; align-items:center; gap:7px; font-size:11px; font-weight:600;
  color:var(--text-muted); background:var(--surface-2); border:1px solid var(--border);
  padding:5px 11px; border-radius:999px; font-family:var(--font-mono);
}}
.f1-pill .dot {{ width:7px; height:7px; border-radius:50%; background:var(--good); box-shadow:0 0 0 3px rgba(52,211,153,.13); }}

/* ---- page header (ui.page_header) ---- */
.f1-eyebrow {{
  font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent-2); font-family:var(--font-mono); margin-bottom:6px;
}}
.f1-title {{ font-size:24px; font-weight:800; letter-spacing:-.025em; line-height:1.1; margin:0; }}
.f1-desc {{ color:var(--text-muted); font-size:14px; margin:8px 0 0; max-width:66ch; }}

/* ---- section header (ui.section) ---- */
.f1-section {{ display:flex; align-items:baseline; gap:12px; margin:6px 0 2px; }}
.f1-section h4 {{ font-size:17px; font-weight:700; letter-spacing:-.02em; margin:0; }}
.f1-section .sub {{ font-size:12.5px; color:var(--text-faint); }}
.f1-section .rule {{ flex:1; height:1px; background:var(--border); align-self:center; }}

/* ---- metric cards ---- */
[data-testid="stMetric"] {{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--r-md);
  padding:14px 16px 12px; position:relative; overflow:hidden;
}}
[data-testid="stMetric"]::after {{
  content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--accent); opacity:.55;
}}
[data-testid="stMetricLabel"] p {{
  font-size:11px !important; font-weight:600; color:var(--text-faint);
  text-transform:uppercase; letter-spacing:.07em;
}}
[data-testid="stMetricValue"] {{
  font-family:var(--font-mono); letter-spacing:-.02em; font-weight:600;
}}

/* ---- horizontal radios -> segmented control ---- */
[data-testid="stRadio"] [role="radiogroup"] {{
  flex-direction:row; flex-wrap:wrap; gap:6px; background:var(--surface-2);
  border:1px solid var(--border); border-radius:var(--r-md); padding:4px; width:fit-content;
}}
[data-testid="stRadio"] [role="radiogroup"] label {{
  margin:0; padding:5px 12px; border-radius:7px; cursor:pointer;
  transition:background .12s, color .12s;
}}
[data-testid="stRadio"] [role="radiogroup"] label:hover {{ background:var(--surface-3); }}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{
  background:var(--surface); box-shadow:0 1px 2px rgba(0,0,0,.4);
}}
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{ display:none; }}  /* hide the dot */

/* ---- buttons ---- */
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {{
  border-radius:var(--r-sm); font-weight:600;
}}

/* ---- expander as a card ---- */
[data-testid="stExpander"] details {{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--r-lg);
}}

/* ---- dataframe wrapper ---- */
[data-testid="stDataFrame"] {{ border-radius:var(--r-lg); }}

/* ---- alerts (warning/info/error/success) — quieter, design-aligned ---- */
[data-testid="stAlert"] {{ border-radius:var(--r-md); border:1px solid var(--border); }}

/* ---- scrollbars: stable gutter ---- */
::-webkit-scrollbar {{ width:10px; height:10px; }}
::-webkit-scrollbar-thumb {{ background:#2a3a58; border-radius:999px; border:2px solid transparent; background-clip:padding-box; }}
::-webkit-scrollbar-track {{ background:transparent; }}
</style>
"""


def inject() -> None:
    """Inject the global stylesheet (anti-reflow + design system). Idempotent
    per page run; call once near the top of streamlit_app.py."""
    st.markdown(_css(), unsafe_allow_html=True)


# ============================================================
# COMPONENT HELPERS
# ============================================================
def app_header(title: str, subtitle: str, status: str | None = None) -> None:
    """Branded wordmark header for the top of the main content column."""
    pill = (
        f'<span class="f1-pill"><span class="dot"></span>{status}</span>' if status else ""
    )
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


def page_header(
    title: str, eyebrow: str | None = None, desc: str | None = None
) -> None:
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
    table upgrade. Feed it values from ui.pct(), not raw 0..1 probabilities:
    ProgressColumn formats the bar label from the raw number, so it must already
    be on a 0..100 scale to read as a percent."""
    return st.column_config.ProgressColumn(
        label, help=help, min_value=0.0, max_value=100.0, format="%.0f%%"
    )


def team_styler(df: Any, team_col: str = "team", subset: list[str] | None = None) -> Any:
    """Return a pandas Styler that tints the team cell with its identity color
    and right-aligns numeric columns. st.dataframe honours Styler backgrounds."""
    def _team_bg(val: Any) -> str:
        c = team_color(val)
        return f"color:{c}; font-weight:600;"

    styler = df.style
    if team_col in df.columns:
        styler = styler.map(_team_bg, subset=[team_col])
    return styler


# ============================================================
# ALTAIR — one shared theme (axes/grid/font/legend/tooltip)
# ============================================================
def _f1_altair_theme() -> dict:
    return {
        "config": {
            "background": "transparent",
            "font": FONT_UI,
            "view": {"stroke": "transparent"},
            "axis": {
                "labelFont": FONT_MONO,
                "labelColor": TEXT_FAINT,
                "labelFontSize": 10,
                "titleFont": FONT_UI,
                "titleColor": TEXT_MUTED,
                "titleFontSize": 11,
                "titleFontWeight": 600,
                "gridColor": GRID,
                "gridWidth": 1,
                "domainColor": BORDER,
                "tickColor": BORDER,
            },
            "legend": {
                "labelFont": FONT_UI,
                "labelColor": TEXT_MUTED,
                "titleFont": FONT_UI,
                "titleColor": TEXT_FAINT,
                "labelFontSize": 11,
            },
            "range": {"category": CHART["range"]},
            "mark": {"color": ACCENT},
            "bar": {"color": ACCENT},
            "line": {"color": ACCENT, "strokeWidth": 2.4},
            "point": {"color": ACCENT, "filled": True},
        }
    }


def enable_altair_theme() -> None:
    """Register + enable the shared chart theme globally. Pages should pass
    `theme=None` to st.altair_chart so this theme (not Streamlit's default)
    drives the look; the helper below does that for you."""
    name = "f1"
    try:  # altair >= 5.5 / 6.x
        alt.theme.register(name, enable=True)(_f1_altair_theme)
    except (AttributeError, TypeError):  # older altair
        alt.themes.register(name, _f1_altair_theme)
        alt.themes.enable(name)


def altair_chart(chart: Any, **kwargs: Any) -> None:
    """st.altair_chart wrapper that lets our registered theme win over
    Streamlit's built-in chart theme and keeps container-width default."""
    kwargs.setdefault("use_container_width", True)
    kwargs.setdefault("theme", None)
    st.altair_chart(chart, **kwargs)
