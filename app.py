import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

DB_PATH = "cab_b85.db"

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

st.set_page_config(page_title="B8.5 Audi S4/S5 Valuation Dashboard", layout="wide")
st.title("2013–2016 Audi S4 & S5 (B8.5) Cars & Bids Hedonic Valuation")


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
    st.error(f"Could not load `{DB_PATH}`. Run `scrape_cab.py` and `extract_features.py` first. ({exc})")
    st.stop()

if df_all.empty:
    st.warning("No parsed auctions found in the database.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("1. Sample Filters")
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
    help="Automatically omits sparse categories/flags to prevent singular matrices and inflated variance.",
)
show_rnm_table = st.sidebar.checkbox("Show RNM listings in table (always excluded from OLS)", value=True)

st.sidebar.header("2. OLS Covariate Toggles")
selected_cats = st.sidebar.multiselect(
    "Categorical Features",
    options=list(CATEGORICAL_SPECS.keys()),
    default=DEFAULT_CATEGORICALS,
    format_func=lambda k: CATEGORICAL_SPECS[k][0],
    help="Wheel Size is off by default because it is collinear with Black Optic / Black Optic Plus.",
)
selected_opts = st.sidebar.multiselect(
    "Binary Factory Options",
    options=list(OPTION_LABELS.keys()),
    default=DEFAULT_OPTIONS,
    format_func=lambda k: OPTION_LABELS[k],
    help="Toggle Prestige or Tech/B&O off if VIF diagnostics show high bundling collinearity.",
)

# Filter dataframe
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

if len(df_sold) < 20:
    st.warning(f"Only {len(df_sold)} completed sales match the current filters. Widen filters to run OLS.")
    st.stop()

# --- Dynamic Formula Construction ---
# Lump sparse levels (< min_obs) into the reference category or drop to avoid singular matrices
df_model = df_sold.copy()
terms = ["mileage_10k", "sale_year_frac", "year"]
active_categoricals: dict[str, str] = {}

for col in selected_cats:
    _, default_ref = CATEGORICAL_SPECS[col]
    counts = df_model[col].value_counts()
    valid_levels = counts[counts >= min_obs].index.tolist()
    if len(valid_levels) >= 2:
        chosen_ref = default_ref if default_ref in valid_levels else counts.idxmax()
        # Keep only rows with valid levels or group rare levels? Grouping rare levels into 'Other' can distort ref.
        # Cleanest for OLS stability: treat rare levels (< min_obs) as separate only if >= min_obs, else drop from dummy?
        # Using Patsy Treatment encoding on all rows while keeping sparse levels causes high SEs; we filter sparse dummy levels to ref only if needed.
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

# --- KPI Row ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Completed Sales (n)", f"{len(df_sold)}")
k2.metric("Mean Sale Price", f"${df_sold['price'].mean():,.0f}")
k3.metric("Median Sale Price", f"${df_sold['price'].median():,.0f}")
k4.metric("Median Mileage", f"{df_sold['mileage'].median():,.0f} mi")
k5.metric("Adj. R² (HC3 OLS)", f"{model.rsquared_adj:.3f}")

# --- Body Style Baseline Summary ---
st.subheader("Average Selling Price by Body Style (Completed Sales)")
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
st.dataframe(
    summary_df.style.format(
        {
            "Mean_Price": "${:,.0f}",
            "Median_Price": "${:,.0f}",
            "Avg_Mileage": "{:,.0f}",
            "Manual_Share": "{:.1f}%",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# --- Parse Regression Coefficients into Exact Percentage Premiums ---
ci = model.conf_int()
rows = []

for param, beta in model.params.items():
    if param == "Intercept":
        continue
    p_val = model.pvalues[param]
    ci_low, ci_high = ci.loc[param, 0], ci.loc[param, 1]

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

    # Hide ultra-sparse dummy levels (< min_obs) from chart while keeping them in full OLS output
    if n_count < min_obs:
        continue

    rows.append(
        {
            "Feature": label,
            "Category": group,
            "Premium (%)": pct,
            "CI Low (%)": pct_low,
            "CI High (%)": pct_high,
            "p-value": p_val,
            "n": n_count,
            "Significant": "p < 0.05" if p_val < 0.05 else "p >= 0.05 (Noisy)",
        }
    )

coef_df = pd.DataFrame(rows).sort_values("Premium (%)", ascending=True)

# --- Premium / Discount Chart ---
st.subheader("Regression-Adjusted Price Premiums & Discounts (95% HC3 CI)")
fig_prem = go.Figure()
for sig_label, color in [("p < 0.05", "#1f77b4"), ("p >= 0.05 (Noisy)", "#9e9e9e")]:
    sub = coef_df[coef_df["Significant"] == sig_label]
    if sub.empty:
        continue
    fig_prem.add_trace(
        go.Bar(
            y=sub["Feature"],
            x=sub["Premium (%)"],
            orientation="h",
            name=sig_label,
            marker_color=color,
            error_x=dict(
                type="data",
                symmetric=False,
                array=sub["CI High (%)"] - sub["Premium (%)"],
                arrayminus=sub["Premium (%)"] - sub["CI Low (%)"],
                thickness=1.2,
            ),
            customdata=np.stack(
                [sub["n"], sub["p-value"], sub["CI Low (%)"], sub["CI High (%)"]], axis=-1
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Impact: %{x:+.1f}%<br>"
                "95% CI: [%{customdata[2]:+.1f}%, %{customdata[3]:+.1f}%]<br>"
                "p-value: %{customdata[1]:.4f} | n = %{customdata[0]}<extra></extra>"
            ),
        )
    )

fig_prem.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
fig_prem.update_layout(
    height=max(520, len(coef_df) * 28),
    xaxis_title="Regression-Adjusted Price Impact (%)",
    yaxis_title="",
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_prem, use_container_width=True)

# --- Market Dynamics Charts ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("Sale Price vs. Mileage")
    fig_mi = px.scatter(
        df_sold,
        x="mileage",
        y="price",
        color="body_style",
        symbol="transmission",
        hover_data=["year", "color_group", "optic_package", "mod_tier", "sale_date"],
        labels={"mileage": "Mileage", "price": "Sale Price ($)"},
    )
    st.plotly_chart(fig_mi, use_container_width=True)

with c2:
    st.subheader("Sale Price Over Time")
    fig_time = px.scatter(
        df_sold,
        x="sale_date_dt",
        y="price",
        color="body_style",
        trendline="lowess",
        hover_data=["year", "mileage", "transmission", "optic_package", "mod_tier"],
        labels={"sale_date_dt": "Sale Date", "price": "Sale Price ($)"},
    )
    st.plotly_chart(fig_time, use_container_width=True)

# --- Live Valuation Calculator ---
st.subheader("Interactive B8.5 Valuation Estimator")
vc1, vc2, vc3, vc4 = st.columns(4)
in_body = vc1.selectbox("Body Style", sorted(df_sold["body_style"].unique()))
in_year = vc1.selectbox("Model Year", [2013, 2014, 2015, 2016], index=3)
in_optic = vc1.selectbox("Optic Package", sorted(df_sold["optic_package"].unique()))

in_mi = vc2.number_input("Mileage", min_value=5000, max_value=200000, value=60000, step=2500)
in_trans = vc2.selectbox("Transmission", sorted(df_sold["transmission"].unique()))
in_seats = vc2.selectbox("Seat Package", sorted(df_sold["seat_package"].unique()))

in_color = vc3.selectbox("Color Group", sorted(df_sold["color_group"].unique()))
in_mod = vc3.selectbox("Modification Tier", sorted(df_sold["mod_tier"].unique()))
in_inlays = vc3.selectbox("Interior Inlays", sorted(df_sold["interior_inlays"].unique()))

in_acc = vc4.selectbox("Accident History", sorted(df_sold["accident_history"].unique()))
in_wheels = vc4.selectbox("Factory Wheel Size", sorted(df_sold["wheel_size_factory"].unique()))
in_sale_yr = vc4.number_input(
    "Valuation Year",
    min_value=2020.0,
    max_value=2028.0,
    value=float(df_sold["sale_year_frac"].max()),
    step=0.25,
)

opt_cols = st.columns(4)
in_opts = {}
for idx, (opt_key, opt_label) in enumerate(OPTION_LABELS.items()):
    in_opts[opt_key] = int(opt_cols[idx % 4].checkbox(opt_label, value=(opt_key == "opt_sport_diff")))

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
    st.info(
        f"**Estimated Fair Auction Value:** ${est_price:,.0f}  "
        f"(95% Prediction Interval: ${pi_low:,.0f} – ${pi_high:,.0f})"
    )
except Exception as exc:
    st.warning(f"Estimator unavailable for selected combination (category omitted from model): {exc}")

# --- Full Auction Table ---
st.subheader("Auction Results Table")
table_df = df_filtered if show_rnm_table else df_sold
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
        "url": st.column_config.LinkColumn("Auction Link"),
        "price": st.column_config.NumberColumn("Price", format="$%d"),
        "mileage": st.column_config.NumberColumn("Mileage", format="%d"),
    },
    use_container_width=True,
    hide_index=True,
)

csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Filtered Dataset (CSV)",
    data=csv_bytes,
    file_name="b85_s4_s5_auctions.csv",
    mime="text/csv",
)

# --- Diagnostics ---
with st.expander("Multicollinearity Diagnostics (VIF)"):
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
    st.dataframe(vif_df, use_container_width=True, hide_index=True)

with st.expander("Full OLS Regression Output (HC3 Robust SEs)"):
    st.text(model.summary().as_text())
