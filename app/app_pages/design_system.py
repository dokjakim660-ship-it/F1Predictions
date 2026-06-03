"""Design System page — the implementation spec, rendered live.

A portfolio/reference page that shows the actual tokens and components the app
is built from (app/ui.py + .streamlit/config.toml). Everything here uses the
same CSS variables the rest of the app does, so toggling the sidebar theme
repaints this page too — it doubles as a light/dark parity demo.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import ui

ui.page_header(
    "Design System",
    eyebrow="About · Reference",
    desc="The single source of truth for the redesign. Tokens map to "
    "`.streamlit/config.toml`, components to `app/ui.py`, and every chart to one "
    "shared Altair theme. Toggle the theme in the sidebar — every token has "
    "light/dark parity.",
)


# --- Colour tokens ------------------------------------------------------
def _swatch(var: str, label: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;">'
        f'<span style="width:30px;height:30px;border-radius:8px;background:var({var});'
        f'border:1px solid var(--border-2);flex:none;"></span>'
        f'<div><div style="font-size:12.5px;font-weight:600;color:var(--text);">{label}</div>'
        f'<div style="font-family:var(--font-mono);font-size:11px;color:var(--text-faint);">'
        f"var({var})</div></div></div>"
    )


ui.section("Colour tokens", sub="full light/dark parity")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        "".join(
            _swatch(v, lbl)
            for v, lbl in [
                ("--bg", "App background"),
                ("--surface", "Surface / card"),
                ("--surface-2", "Surface raised"),
                ("--surface-3", "Track / inset"),
                ("--border", "Border"),
                ("--text", "Text"),
                ("--text-muted", "Text muted"),
                ("--text-faint", "Text faint"),
            ]
        ),
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        "".join(
            _swatch(v, lbl)
            for v, lbl in [
                ("--accent", "Accent (primaryColor)"),
                ("--accent-2", "Accent hover"),
                ("--good", "Positive / win"),
                ("--bad", "Negative / DNF"),
                ("--warn", "Warning"),
            ]
        ),
        unsafe_allow_html=True,
    )


# --- Team colour system -------------------------------------------------
ui.section("Team colour system", sub="2026 grid")
_TEAMS = [
    "McLaren", "Ferrari", "Mercedes", "Red Bull", "Williams", "Aston Martin",
    "Alpine", "Racing Bulls", "Audi", "Haas", "Cadillac",
]
chips = "".join(
    f'<div style="display:flex;align-items:center;gap:10px;padding:8px 10px;'
    f'border:1px solid var(--border);border-radius:10px;background:var(--surface-2);">'
    f'<span style="width:22px;height:22px;border-radius:6px;background:{ui.team_color(t)};'
    f'flex:none;box-shadow:var(--shadow);"></span>'
    f'<div style="min-width:0;"><div style="font-size:12.5px;font-weight:600;color:var(--text);'
    f'white-space:nowrap;">{t}</div>'
    f'<div style="font-family:var(--font-mono);font-size:10.5px;color:var(--text-faint);">'
    f"{ui.team_color(t)}</div></div></div>"
    for t in _TEAMS
)
st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;">'
    f"{chips}</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Used only as a thin identity accent (table team column, swatch) — never as a "
    "fill behind text, to preserve contrast."
)


# --- Typography ---------------------------------------------------------
ui.section("Typography", sub="Helvetica UI · JetBrains Mono for numbers")
type_rows = [
    ("Display / hero", 30, 800),
    ("H1 / page title", 24, 800),
    ("H2 / section", 17, 700),
    ("Body", 14, 400),
    ("Small / caption", 12, 400),
]
rows_html = "".join(
    f'<div style="display:flex;align-items:baseline;gap:16px;padding:9px 0;'
    f'border-bottom:1px solid var(--border);">'
    f'<div style="flex:1;font-size:{fs}px;font-weight:{w};letter-spacing:-.02em;'
    f'line-height:1;color:var(--text);">Aa Grid</div>'
    f'<div style="width:150px;font-size:12px;color:var(--text-muted);">{name}</div>'
    f'<div style="font-family:var(--font-mono);font-size:11px;color:var(--text-faint);">'
    f"{fs}px / {w}</div></div>"
    for name, fs, w in type_rows
)
st.markdown(
    f'<div style="background:var(--surface);border:1px solid var(--border);'
    f'border-radius:var(--r-lg);padding:6px 16px;">{rows_html}'
    f'<div style="display:flex;gap:24px;flex-wrap:wrap;padding:14px 0 6px;">'
    f'<div><div style="font-size:11px;color:var(--text-faint);">UI font</div>'
    f'<div style="font-weight:700;color:var(--text);">Helvetica Neue / Arial</div></div>'
    f'<div><div style="font-size:11px;color:var(--text-faint);">Numeric font</div>'
    f'<div style="font-family:var(--font-mono);font-weight:600;color:var(--text);">'
    f"0123456789 · 18.6% · 0.118</div></div></div></div>",
    unsafe_allow_html=True,
)


# --- Components ---------------------------------------------------------
ui.section("Components", sub="from app/ui.py")

st.markdown("**Metric cards** — `st.metric`, restyled")
m1, m2, m3 = st.columns(3)
m1.metric("Predicted winner", "NOR", "30% win")
m2.metric("Net P&L", "+26.4u", "+18.6%")
m3.metric("Active model", "LightGBM")

st.markdown("**Model comparison** — `ui.model_comparison` (best-first, active highlighted)")
ui.model_comparison(
    [
        {"key": "lgbm", "name": "LightGBM", "score": 0.118},
        {"key": "ens", "name": "Ensemble", "score": 0.124},
        {"key": "logit", "name": "Logistic", "score": 0.139},
        {"key": "elo", "name": "Elo-baseline", "score": 0.171},
    ],
    active_key="lgbm",
    fmt=lambda s: f"{s:.3f}",
)

st.markdown("**Data-bar table** — `ui.prob_column` + `ui.team_styler`")
demo = pd.DataFrame(
    {
        "driver": ["Norris", "Verstappen", "Leclerc", "Hamilton"],
        "team": ["McLaren", "Red Bull", "Ferrari", "Ferrari"],
        "P(podium)": ui.pct(pd.Series([0.62, 0.55, 0.41, 0.22])),
    }
)
st.dataframe(
    ui.team_styler(demo),
    hide_index=True,
    width="stretch",
    column_config={"P(podium)": ui.prob_column("P(podium)")},
)

st.markdown("**Segmented controls** — horizontal `st.radio`, restyled")
st.radio(
    "Example", ["podium", "teammate", "pole"], horizontal=True,
    key="ds_demo_radio", label_visibility="collapsed",
)


# --- Chart theme --------------------------------------------------------
ui.section("Chart theme", sub="one Altair theme for every page")
trend = pd.DataFrame(
    {
        "round": ["R1", "R2", "R3", "R4", "R5", "R6"],
        "Ensemble": [0.142, 0.131, 0.128, 0.121, 0.119, 0.118],
        "Baseline": [0.198, 0.185, 0.180, 0.176, 0.172, 0.171],
    }
).melt("round", var_name="series", value_name="brier")
chart = (
    alt.Chart(trend)
    .mark_line(point=True)
    .encode(
        x=alt.X("round:N", title=None),
        y=alt.Y("brier:Q", title="Brier", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None),
        tooltip=["round", "series", alt.Tooltip("brier:Q", format=".3f")],
    )
    .properties(height=220)
)
ui.altair_chart(chart)
st.caption(
    "Muted gridlines, monospace tick labels, accent series, transparent "
    "background. Registered once and enabled globally."
)


# --- Spacing & radii ----------------------------------------------------
ui.section("Spacing & radii", sub="4pt scale")
spaces = "".join(
    f'<div style="text-align:center;"><div style="width:{v}px;height:{v}px;'
    f'background:var(--accent);border-radius:3px;margin:0 auto 6px;"></div>'
    f'<div style="font-family:var(--font-mono);font-size:10.5px;color:var(--text-faint);">'
    f"{v}px</div></div>"
    for v in [4, 8, 12, 16, 20, 24, 32, 40]
)
radii = "".join(
    f'<div style="text-align:center;"><div style="width:56px;height:40px;'
    f'background:var(--surface-3);border:1px solid var(--border-2);border-radius:{v}px;"></div>'
    f'<div style="font-family:var(--font-mono);font-size:10.5px;color:var(--text-faint);'
    f'margin-top:6px;">r-{n} · {v}px</div></div>'
    for n, v in [("sm", 6), ("md", 10), ("lg", 14)]
)
st.markdown(
    f'<div style="background:var(--surface);border:1px solid var(--border);'
    f'border-radius:var(--r-lg);padding:18px 16px;">'
    f'<div style="display:flex;flex-wrap:wrap;gap:18px;align-items:flex-end;">{spaces}</div>'
    f'<div style="display:flex;gap:14px;margin-top:22px;">{radii}</div></div>',
    unsafe_allow_html=True,
)
