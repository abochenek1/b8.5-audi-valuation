import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

DB_PATH = "cab_b85.db"

# Material Design 3 Color Tokens
M3_PRIMARY = "#1A73E8"
M3_PRIMARY_CONTAINER = "#E8F0FE"
M3_ON_PRIMARY_CONTAINER = "#174EA6"
M3_SURFACE = "#FFFFFF"
M3_SURFACE_VARIANT = "#F1F3F4"
M3_OUTLINE = "#DADCE0"
M3_TEXT = "#202124"
M3_TEXT_MUTED = "#5F6368"
M3_POSITIVE = "#1E8E3E"
M3_NEGATIVE = "#D93025"
M3_NEUTRAL = "#9AA0A6"

BODY_PALETTE = {
    "S4 Sedan": "#1A73E8",
    "S5 Coupe": "#1E8E3E",
    "S5 Cabriolet": "#E37400",
}

CATEGORICAL_SPECS = {
    "body_style": ("Body Style", "S4 Sedan"),
    "transmission": ("Transmission", "S tronic"),
    "color_group": ("Color Group", "Gray/Silver"),
    "mod_tier": ("Modification Tier", "Stock"),
    "accident_history": ("Accident History", "Clean"),
    "optic_package": ("Optic Package", "Standard"),
    "seat_package": ("Seat Package", "Leather/Alcantara"),
    "interior_inlays": ("Interior Inlays", "Brushed Aluminum"),
    "wheel_size_factory": ("Factory Wheel Size", "18-inch"),
}

DEFAULT_CATEGORICALS = [
    "body_style",
    "transmission",
    "color_group",
    "mod_tier",
    "accident_history",
    "optic_package",
    "seat_package",
    "interior_inlays",
]

OPTION_LABELS = {
    "opt_sport_diff": "Sports Differential",
    "opt_prestige": "Prestige Trim",
    "opt_bang_olufsen": "Bang & Olufsen Audio",
    "opt_adaptive_damping": "Adaptive Damping Suspension",
    "opt_dynamic_steering": "Dynamic Steering",
    "opt_driver_assist": "Driver Assistance Pkg (ACC)",
    "opt_tech_pkg": "Technology / MMI Nav+ Pkg",
    "opt_advanced_key": "Audi Advanced Key",
    "opt_two_tone_interior": "Two-Tone Interior",
    "opt_rear_sunshade": "Power Rear Sunshade",
    "opt_nogaro_edition": "Nogaro Blue Special Edition",
}

DEFAULT_OPTIONS = [
    "opt_sport_diff",
    "opt_prestige",
    "opt_bang_olufsen",
    "opt_adaptive_damping",
    "opt_dynamic_steering",
    "opt_driver_assist",
    "opt_tech_pkg",
    "opt_two_tone_interior",
    "opt_nogaro_edition",
]

st.set_page_config(
    page_title="B8.5 Audi S4/S5 Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Material Design 3 CSS Layer ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #202124;
    }

    /* Main canvas spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #DADCE0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* M3 Hero Header */
    .m3-hero {
        background: #FFFFFF;
        border: 1px solid #DADCE0;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.08), 0 1px 3px 1px rgba(60, 64, 67, 0.04);
    }
    .m3-hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #1A73E8;
        margin-bottom: 0.35rem;
    }
    .m3-hero-title {
        font-size: 1.75rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #202124;
        margin: 0;
        line-height: 1.25;
    }
    .m3-hero-subtitle {
        font-size: 0.9rem;
        color: #5F6368;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* M3 KPI Metric Cards */
    .m3-kpi-card {
        background: #FFFFFF;
        border: 1px solid #DADCE0;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.06);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .m3-kpi-card:hover {
        border-color: #BDC1C6;
        box-shadow: 0 2px 6px rgba(60, 64, 67, 0.1);
    }
    .m3-kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #5F6368;
        margin-bottom: 0.4rem;
    }
    .m3-kpi-value {
        font-size: 1.65rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #202124;
        line-height: 1.2;
    }
    .m3-kpi-sub {
        font-size: 0.76rem;
        color: #5F6368;
        margin-top: 0.3rem;
    }

    /* M3 Section Titles inside Containers */
    .m3-section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #202124;
        margin-bottom: 0.15rem;
    }
    .m3-section-caption {
        font-size: 0.82rem;
        color: #5F6368;
        margin-bottom: 1rem;
    }

    /* M3 Tonal Valuation Result Banner */
    .m3-result-banner {
        background: #E8F0FE;
        border: 1px solid #AECBFA;
        border-radius: 16px;
        padding: 1.4rem 1.75rem;
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
    }
    .m3-result-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #174EA6;
    }
    .m3-result-price {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #174EA6;
        line-height: 1.15;
    }
    .m3-result-pi {
        font-size: 0.88rem;
        color: #1967D2;
        font-weight: 500;
    }

    /* M3 Pill Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #FFFFFF;
        padding: 0.35rem;
        border-radius: 9999px;
        border: 1px solid #DADCE0;
        width: fit-content;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9999px;
        padding: 0.45rem 1.15rem;
        font-size: 0.86rem;
        font-weight: 500;
        color: #5F6368;
        border: none !important;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E8F0FE !important;
        color: #1A73E8 !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* Elevated st.container(border=True) overrides */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border: 1px solid #DADCE0 !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.35rem !important;
        box-shadow: 0 1px 2px rgba(60, 64, 67, 0.06);
    }

    /* Clean multiselect chips */
    span[data-baseweb="tag"] {
        background-color: #E8F0FE !important;
        color: #174EA6 !important;
        border-radius: 8px !important;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_m3_plotly_layout(fig: go.Figure, height: int = 440) -> go.Figure:
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=12, color=M3_TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=12, r=24, t=36, b=24),
        hoverlabel=dict(
            bgcolor="#202124",
            font_size=12,
            font_family="Inter, Roboto, sans-serif",
            font_color="#FFFFFF",
            bordercolor="#202124",
        ),
        xaxis=dict(
            gridcolor="#E8EAED",
            zerolinecolor="#BDC1C6",
            linecolor="#DADCE0",
            tickfont=dict(color=M3_TEXT_MUTED),
            title_font=dict(color=M3_TEXT_MUTED, size=12),
        ),
        yaxis=dict(
            gridcolor="#E8EAED",
            zerolinecolor="#BDC1C6",
            linecolor="#DADCE0",
            tickfont=dict(color=M3_TEXT),
            title_font=dict(color=M3_TEXT_MUTED, size=12),
        ),
    )
    return fig


def render_kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="m3-kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="m3-kpi-card">'
        f'<div class="m3-kpi-label">{label}</div>'
        f'<div class="m3-kpi-value">{value}</div>'
        f"{sub_html}"
        f"</div>"
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            r.url, r.year, r.title, r.status, r.price, r.sale_date, r.mileage,
            r.exterior_color_raw, r.interior_color_raw, r.location, r.vin,
            p.body_style, p.transmission, p.color_group, p.accident_history, p.mod_tier,
            p.optic_package, p.wheel_size_factory, p.seat_package, p.interior_inlays,
            p.opt_sport_diff, p.opt_prestige, p.opt_bang_olufsen, p.opt_adaptive_damping,
            p.opt_dynamic_steering, p.opt_driver_assist, p.opt_tech_pkg, p.opt_advanced_key,
            p.opt_two_tone_interior, p.opt_rear_sunshade, p.opt_nogaro_edition,
            p.notes
        FROM raw_auctions r
        INNER JOIN parsed_auctions p ON r.url = p.url
        WHERE r.price IS NOT NULL AND r.mileage IS NOT NULL
        """,
        conn,
    )
    conn.close()
    df["sale_date_dt"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["sale_year_frac"] = (
        df["sale_date_dt"].dt.year + (df["sale_date_dt"].dt.dayofyear - 1) / 365.25
    )
    df["mileage_10k"] = df["mileage"] / 10000.0
    df["log_price"] = np.log(df["price"])
    return df


try:
    df_all = load_data()
except Exception as exc:
    st.error(f"Could not load `{DB_PATH}`. Confirm the database exists in the project root. ({exc})")
    st.stop()

if df_all.empty:
    st.warning("No parsed auctions found in the database.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.markdown("### Sample Filters")
body_styles = sorted(df_all["body_style"].dropna().unique().tolist())
selected_bodies = st.sidebar.multiselect("Body Styles", body_styles, default=body_styles)

valid_dates = df_all["sale_date_dt"].dropna()
min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
date_range = st.sidebar.date_input(
    "Sale Date Window", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

max_mi_cap = st.sidebar.slider("Max Mileage Cap", 20000, 200000, 150000, step=5000)
min_obs = st.sidebar.slider(
    "Min Sample Count (n) per Feature",
    2,
    15,
    5,
    help="Omits sparse categories/flags from the chart to prevent inflated variance.",
)
show_rnm_table = st.sidebar.checkbox(
    "Include RNM listings in Ledger tab (always excluded from OLS)", value=True
)

st.sidebar.divider()
st.sidebar.markdown("### OLS Covariate Toggles")
selected_cats = st.sidebar.multiselect(
    "Categorical Features",
    options=list(CATEGORICAL_SPECS.keys()),
    default=DEFAULT_CATEGORICALS,
    format_func=lambda k: CATEGORICAL_SPECS[k][0],
    help="Wheel Size is off by default due to collinearity with Black Optic / Black Optic Plus.",
)
selected_opts = st.sidebar.multiselect(
    "Binary Factory Options",
    options=list(OPTION_LABELS.keys()),
    default=DEFAULT_OPTIONS,
    format_func=lambda k: OPTION_LABELS[k],
    help="Toggle Prestige or Tech/B&O off if VIF diagnostics show bundling collinearity.",
)

# --- Filter & Fit HC3 OLS Model ---
mask = df_all["body_style"].isin(selected_bodies) & (df_all["mileage"] <= max_mi_cap)
if isinstance(date_range, tuple) and len(date_range) == 2:
    mask &= (df_all["sale_date_dt"].dt.date >= date_range[0]) & (
        df_all["sale_date_dt"].dt.date <= date_range[1]
    )

df_filtered = df_all[mask].copy()
df_sold = (
    df_filtered[df_filtered["status"].isin(["Sold", "Sold After"])]
    .dropna(subset=["sale_year_frac"])
    .copy()
)

# --- Hero Header ---
date_span_str = (
    f"{df_sold['sale_date_dt'].min().strftime('%b %Y')} – "
    f"{df_sold['sale_date_dt'].max().strftime('%b %Y')}"
    if not df_sold.empty
    else "N/A"
)
st.markdown(
    f"""
    <div class="m3-hero">
        <div class="m3-hero-eyebrow">Cars &amp; Bids Market Analytics · Log-Linear HC3 OLS</div>
        <h1 class="m3-hero-title">2013–2016 Audi S4 &amp; S5 (B8.5) Hedonic Valuation</h1>
        <p class="m3-hero-subtitle">
            Regression-adjusted option, color, transmission, and modification premiums across
            <strong>{len(df_sold)}</strong> completed auctions ({date_span_str}).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if len(df_sold) < 20:
    st.warning(
        f"Only {len(df_sold)} completed sales match the current filters. Widen filters to run OLS."
    )
    st.stop()

df_model = df_sold.copy()
terms = ["mileage_10k", "sale_year_frac", "year"]
active_categoricals: dict[str, str] = {}

for col in selected_cats:
    _, default_ref = CATEGORICAL_SPECS[col]
    counts = df_model[col].value_counts()
    valid_levels = counts[counts >= min_obs].index.tolist()
    if len(valid_levels) >= 2:
        chosen_ref = default_ref if default_ref in valid_levels else counts.idxmax()
        terms.append(f"C({col}, Treatment(reference='{chosen_ref}'))")
        active_categoricals[col] = chosen_ref

active_options: list[str] = []
for opt_col in selected_opts:
    n_pos = int(df_model[opt_col].sum())
    n_neg = len(df_model) - n_pos
    if n_pos >= min_obs and n_neg >= min_obs:
        terms.append(opt_col)
        active_options.append(opt_col)

formula = "log_price ~ " + " + ".join(terms)
model = smf.ols(formula, data=df_model).fit(cov_type="HC3")
smearing_factor = float(np.mean(np.exp(model.resid)))

# --- KPI Strip ---
k1, k2, k3, k4, k5 = st.columns(5)
rnm_count = int((df_filtered["status"] == "RNM").sum())
k1.markdown(
    render_kpi_card("Completed Sales", f"{len(df_sold):,}", f"{rnm_count} RNM excluded"),
    unsafe_allow_html=True,
)
k2.markdown(
    render_kpi_card(
        "Mean Sale Price",
        f"${df_sold['price'].mean():,.0f}",
        f"Range: ${df_sold['price'].min():,.0f}–${df_sold['price'].max():,.0f}",
    ),
    unsafe_allow_html=True,
)
k3.markdown(
    render_kpi_card(
        "Median Sale Price",
        f"${df_sold['price'].median():,.0f}",
        f"IQR: ${df_sold['price'].quantile(0.25):,.0f}–${df_sold['price'].quantile(0.75):,.0f}",
    ),
    unsafe_allow_html=True,
)
k4.markdown(
    render_kpi_card(
        "Median Mileage",
        f"{df_sold['mileage'].median():,.0f} mi",
        f"Mean: {df_sold['mileage'].mean():,.0f} mi",
    ),
    unsafe_allow_html=True,
)
k5.markdown(
    render_kpi_card(
        "Model Fit (Adj. R²)",
        f"{model.rsquared_adj:.3f}",
        f"HC3 Robust SE · {int(model.df_model)} covariates",
    ),
    unsafe_allow_html=True,
)

st.write("")

# --- Parse Regression Coefficients ---
ci = model.conf_int()
rows = []
for param, beta in model.params.items():
    if param == "Intercept":
        continue
    p_val = float(model.pvalues[param])
    ci_low, ci_high = float(ci.loc[param, 0]), float(ci.loc[param, 1])

    pct = (np.exp(beta) - 1.0) * 100.0
    pct_low = (np.exp(ci_low) - 1.0) * 100.0
    pct_high = (np.exp(ci_high) - 1.0) * 100.0

    if param == "mileage_10k":
        label, group, n_count = "Mileage (per +10k miles)", "Continuous", len(df_sold)
    elif param == "sale_year_frac":
        label, group, n_count = "Sale Date (per +1 year)", "Continuous", len(df_sold)
    elif param == "year":
        label, group, n_count = "Model Year (per +1 yr newer)", "Continuous", len(df_sold)
    elif param in OPTION_LABELS:
        label, group, n_count = OPTION_LABELS[param], "Factory Option", int(df_sold[param].sum())
    elif "[T." in param:
        col_name = param.split("C(")[1].split(",")[0]
        level_name = param.split("[T.")[1].rstrip("]")
        ref_name = active_categoricals.get(col_name, "Ref")
        label = f"{level_name} (vs. {ref_name})"
        group = CATEGORICAL_SPECS.get(col_name, (col_name, ""))[0]
        n_count = int((df_sold[col_name] == level_name).sum())
    else:
        label, group, n_count = param, "Other", len(df_sold)

    if n_count < min_obs:
        continue

    if p_val < 0.05 and pct >= 0:
        sig_bucket = "Positive Premium (p < 0.05)"
    elif p_val < 0.05 and pct < 0:
        sig_bucket = "Negative Discount (p < 0.05)"
    else:
        sig_bucket = "Not Significant (p >= 0.05)"

    rows.append(
        {
            "Feature": label,
            "Category": group,
            "Premium (%)": pct,
            "CI Low (%)": pct_low,
            "CI High (%)": pct_high,
            "p-value": p_val,
            "n": n_count,
            "Bucket": sig_bucket,
        }
    )

coef_df = pd.DataFrame(rows).sort_values("Premium (%)", ascending=True)

# --- 4 Material Navigation Tabs ---
tab_val, tab_mkt, tab_est, tab_ledger = st.tabs(
    [
        "Valuation & Premiums",
        "Market Dynamics",
        "Price Estimator",
        "Auction Ledger & Diagnostics",
    ]
)

# ================= TAB 1: VALUATION & PREMIUMS =================
with tab_val:
    with st.container(border=True):
        st.markdown(
            '<div class="m3-section-title">Regression-Adjusted Price Premiums &amp; Discounts</div>'
            '<div class="m3-section-caption">'
            "Exact percentage impact (e<sup>β</sup> − 1) holding mileage, sale date, model year, and all active covariates constant. "
            "Whiskers show 95% HC3 heteroskedasticity-robust confidence intervals."
            "</div>",
            unsafe_allow_html=True,
        )

        fig_prem = go.Figure()
        bucket_styles = [
            ("Positive Premium (p < 0.05)", M3_POSITIVE),
            ("Negative Discount (p < 0.05)", M3_NEGATIVE),
            ("Not Significant (p >= 0.05)", M3_NEUTRAL),
        ]

        for bucket_name, color in bucket_styles:
            sub = coef_df[coef_df["Bucket"] == bucket_name]
            if sub.empty:
                continue
            fig_prem.add_trace(
                go.Bar(
                    y=sub["Feature"],
                    x=sub["Premium (%)"],
                    orientation="h",
                    name=bucket_name,
                    marker=dict(color=color, line=dict(width=0)),
                    text=[f"{v:+.1f}%" for v in sub["Premium (%)"]],
                    textposition="outside",
                    textfont=dict(size=11, color=M3_TEXT),
                    cliponaxis=False,
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=sub["CI High (%)"] - sub["Premium (%)"],
                        arrayminus=sub["Premium (%)"] - sub["CI Low (%)"],
                        thickness=1.3,
                        width=3,
                        color="#5F6368",
                    ),
                    customdata=np.stack(
                        [sub["n"], sub["p-value"], sub["CI Low (%)"], sub["CI High (%)"]],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Impact: %{x:+.1f}%<br>"
                        "95% CI: [%{customdata[2]:+.1f}%, %{customdata[3]:+.1f}%]<br>"
                        "p-value: %{customdata[1]:.4f} · n = %{customdata[0]}<extra></extra>"
                    ),
                )
            )

        fig_prem.add_vline(x=0, line_width=1.2, line_color="#5F6368", opacity=0.7)
        apply_m3_plotly_layout(fig_prem, height=max(500, len(coef_df) * 30))
        fig_prem.update_layout(
            xaxis_title="Regression-Adjusted Price Impact (%)",
            yaxis_title="",
            barmode="overlay",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=12, color=M3_TEXT_MUTED),
            ),
            margin=dict(l=12, r=56, t=44, b=24),
        )
        st.plotly_chart(fig_prem, use_container_width=True)

    st.write("")

    with st.container(border=True):
        st.markdown(
            '<div class="m3-section-title">Baseline Selling Prices by Body Style</div>'
            '<div class="m3-section-caption">Unadjusted summary statistics across completed sales in the active filter window.</div>',
            unsafe_allow_html=True,
        )
        summary_df = (
            df_sold.groupby("body_style")
            .agg(
                Sales=("price", "count"),
                Mean_Price=("price", "mean"),
                Median_Price=("price", "median"),
                Avg_Mileage=("mileage", "mean"),
                Manual_Share=("transmission", lambda s: (s == "Manual").mean() * 100),
            )
            .reset_index()
        )
        summary_df.columns = [
            "Body Style",
            "Completed Sales",
            "Mean Price",
            "Median Price",
            "Avg Mileage",
            "6MT Share",
        ]
        st.dataframe(
            summary_df.style.format(
                {
                    "Mean Price": "${:,.0f}",
                    "Median Price": "${:,.0f}",
                    "Avg Mileage": "{:,.0f} mi",
                    "6MT Share": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# ================= TAB 2: MARKET DYNAMICS =================
with tab_mkt:
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown(
                '<div class="m3-section-title">Sale Price vs. Mileage</div>'
                '<div class="m3-section-caption">Completed auctions segmented by body style and transmission.</div>',
                unsafe_allow_html=True,
            )
            fig_mi = px.scatter(
                df_sold,
                x="mileage",
                y="price",
                color="body_style",
                symbol="transmission",
                color_discrete_map=BODY_PALETTE,
                hover_data=["year", "color_group", "optic_package", "mod_tier", "sale_date"],
                labels={
                    "mileage": "Odometer (Miles)",
                    "price": "Sale Price ($)",
                    "body_style": "Body Style",
                    "transmission": "Trans",
                },
            )
            fig_mi.update_traces(marker=dict(size=8, opacity=0.82, line=dict(width=0.6, color="#FFFFFF")))
            apply_m3_plotly_layout(fig_mi, height=440)
            fig_mi.update_layout(
                yaxis_tickprefix="$",
                yaxis_tickformat=",",
                xaxis_tickformat=",",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_mi, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="m3-section-title">Sale Price Trajectory Over Time</div>'
                '<div class="m3-section-caption">LOWESS smoothed market trends across completed B8.5 sales.</div>',
                unsafe_allow_html=True,
            )
            fig_time = px.scatter(
                df_sold,
                x="sale_date_dt",
                y="price",
                color="body_style",
                color_discrete_map=BODY_PALETTE,
                trendline="lowess",
                hover_data=["year", "mileage", "transmission", "optic_package", "mod_tier"],
                labels={
                    "sale_date_dt": "Auction Close Date",
                    "price": "Sale Price ($)",
                    "body_style": "Body Style",
                },
            )
            fig_time.update_traces(
                selector=dict(mode="markers"),
                marker=dict(size=7.5, opacity=0.75, line=dict(width=0.5, color="#FFFFFF")),
            )
            apply_m3_plotly_layout(fig_time, height=440)
            fig_time.update_layout(
                yaxis_tickprefix="$",
                yaxis_tickformat=",",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_time, use_container_width=True)

# ================= TAB 3: PRICE ESTIMATOR =================
with tab_est:
    with st.container(border=True):
        st.markdown(
            '<div class="m3-section-title">Interactive B8.5 Auction Valuation Estimator</div>'
            '<div class="m3-section-caption">'
            "Configure a specific vehicle specification to generate a Duan-smeared fair auction value and 95% prediction interval."
            "</div>",
            unsafe_allow_html=True,
        )

        vc1, vc2, vc3, vc4 = st.columns(4)
        in_body = vc1.selectbox("Body Style", sorted(df_sold["body_style"].unique()))
        in_year = vc1.selectbox("Model Year", [2013, 2014, 2015, 2016], index=3)
        in_optic = vc1.selectbox("Optic Package", sorted(df_sold["optic_package"].unique()))

        in_mi = vc2.number_input(
            "Mileage", min_value=5000, max_value=200000, value=60000, step=2500
        )
        in_trans = vc2.selectbox("Transmission", sorted(df_sold["transmission"].unique()))
        in_seats = vc2.selectbox("Seat Package", sorted(df_sold["seat_package"].unique()))

        in_color = vc3.selectbox("Color Group", sorted(df_sold["color_group"].unique()))
        in_mod = vc3.selectbox("Modification Tier", sorted(df_sold["mod_tier"].unique()))
        in_inlays = vc3.selectbox("Interior Inlays", sorted(df_sold["interior_inlays"].unique()))

        in_acc = vc4.selectbox("Accident History", sorted(df_sold["accident_history"].unique()))
        in_wheels = vc4.selectbox(
            "Factory Wheel Size", sorted(df_sold["wheel_size_factory"].unique())
        )
        in_sale_yr = vc4.number_input(
            "Valuation Year",
            min_value=2020.0,
            max_value=2028.0,
            value=float(round(df_sold["sale_year_frac"].max(), 2)),
            step=0.25,
        )

        st.markdown("##### Factory Options")
        opt_cols = st.columns(4)
        in_opts = {}
        for idx, (opt_key, opt_label) in enumerate(OPTION_LABELS.items()):
            in_opts[opt_key] = int(
                opt_cols[idx % 4].checkbox(opt_label, value=(opt_key == "opt_sport_diff"))
            )

        pred_row = pd.DataFrame(
            [
                {
                    "body_style": in_body,
                    "year": in_year,
                    "mileage_10k": in_mi / 10000.0,
                    "transmission": in_trans,
                    "color_group": in_color,
                    "mod_tier": in_mod,
                    "accident_history": in_acc,
                    "optic_package": in_optic,
                    "wheel_size_factory": in_wheels,
                    "seat_package": in_seats,
                    "interior_inlays": in_inlays,
                    "sale_year_frac": in_sale_yr,
                    **in_opts,
                }
            ]
        )

        try:
            pred = model.get_prediction(pred_row).summary_frame(alpha=0.05)
            est_price = float(np.exp(pred["mean"].iloc[0]) * smearing_factor)
            pi_low = float(np.exp(pred["obs_ci_lower"].iloc[0]) * smearing_factor)
            pi_high = float(np.exp(pred["obs_ci_upper"].iloc[0]) * smearing_factor)
            st.markdown(
                f"""
                <div class="m3-result-banner">
                    <div>
                        <div class="m3-result-label">Estimated Fair Auction Value</div>
                        <div class="m3-result-price">${est_price:,.0f}</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="m3-result-label">95% Prediction Interval</div>
                        <div class="m3-result-pi">${pi_low:,.0f} &ndash; ${pi_high:,.0f}</div>
                        <div style="font-size: 0.78rem; color: #5F6368; margin-top: 0.25rem;">
                            {in_year} {in_body} · {in_trans} · {in_mi:,} mi · {in_color} · {in_mod}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.warning(
                f"Estimator unavailable for selected combination (sparse category omitted from model): {exc}"
            )

# ================= TAB 4: AUCTION LEDGER & DIAGNOSTICS =================
with tab_ledger:
    with st.container(border=True):
        hdr_l, hdr_r = st.columns([3, 1])
        with hdr_l:
            st.markdown(
                '<div class="m3-section-title">Auction Results Ledger</div>'
                '<div class="m3-section-caption">Complete parsed dataset matching active sidebar filters.</div>',
                unsafe_allow_html=True,
            )
        table_df = df_filtered if show_rnm_table else df_sold
        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        with hdr_r:
            st.download_button(
                "Export CSV",
                data=csv_bytes,
                file_name="b85_s4_s5_auctions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        display_cols = [
            "sale_date",
            "year",
            "body_style",
            "status",
            "price",
            "mileage",
            "transmission",
            "color_group",
            "optic_package",
            "seat_package",
            "interior_inlays",
            "mod_tier",
            "accident_history",
            "opt_sport_diff",
            "opt_prestige",
            "opt_bang_olufsen",
            "opt_adaptive_damping",
            "url",
        ]
        st.dataframe(
            table_df[display_cols].sort_values("sale_date", ascending=False),
            column_config={
                "url": st.column_config.LinkColumn("Auction", display_text="Open"),
                "price": st.column_config.NumberColumn("Price", format="$%d"),
                "mileage": st.column_config.NumberColumn("Mileage", format="%d"),
                "sale_date": st.column_config.TextColumn("Date"),
                "year": st.column_config.NumberColumn("Year", format="%d"),
                "body_style": st.column_config.TextColumn("Body"),
                "status": st.column_config.TextColumn("Status"),
                "transmission": st.column_config.TextColumn("Trans"),
                "color_group": st.column_config.TextColumn("Color"),
                "optic_package": st.column_config.TextColumn("Optic"),
                "seat_package": st.column_config.TextColumn("Seats"),
                "interior_inlays": st.column_config.TextColumn("Inlays"),
                "mod_tier": st.column_config.TextColumn("Mods"),
                "accident_history": st.column_config.TextColumn("History"),
                "opt_sport_diff": st.column_config.CheckboxColumn("Sport Diff"),
                "opt_prestige": st.column_config.CheckboxColumn("Prestige"),
                "opt_bang_olufsen": st.column_config.CheckboxColumn("B&O"),
                "opt_adaptive_damping": st.column_config.CheckboxColumn("Damping"),
            },
            use_container_width=True,
            hide_index=True,
        )

    st.write("")

    d1, d2 = st.columns([1, 1.35])
    with d1:
        with st.container(border=True):
            st.markdown(
                '<div class="m3-section-title">Multicollinearity Diagnostics (VIF)</div>'
                '<div class="m3-section-caption">Values &gt; 5 indicate moderate bundling collinearity; &gt; 10 warrants toggling a covariate off.</div>',
                unsafe_allow_html=True,
            )
            exog = model.model.exog
            exog_names = model.model.exog_names
            vif_rows = []
            for i, name in enumerate(exog_names):
                if name == "Intercept":
                    continue
                try:
                    vif_val = float(variance_inflation_factor(exog, i))
                except Exception:
                    vif_val = np.nan
                flag = "High (>10)" if vif_val > 10 else ("Moderate (>5)" if vif_val > 5 else "OK")
                vif_rows.append({"Term": name, "VIF": round(vif_val, 2), "Status": flag})
            vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
            st.dataframe(vif_df, use_container_width=True, hide_index=True, height=380)

    with d2:
        with st.container(border=True):
            st.markdown(
                '<div class="m3-section-title">Full OLS Regression Output (HC3 Robust SEs)</div>'
                '<div class="m3-section-caption">Complete statsmodels summary for the active specification.</div>',
                unsafe_allow_html=True,
            )
            st.code(model.summary().as_text(), language="text")
