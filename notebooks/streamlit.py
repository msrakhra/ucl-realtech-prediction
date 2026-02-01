import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


DEFAULT_BASE_URL = "https://api.scansan.com"
HIST_ENDPOINT = "/v1/postcode/{area_code_postal}/valuations/historical"
CURR_ENDPOINT = "/v1/postcode/{area_code_postal}/valuations/current"


@dataclass
class FitResult:
    beta0: float
    beta1: float
    sigma2: float
    xtx_inv: np.ndarray
    n: int


def decimal_year(dt: pd.Series) -> pd.Series:
    """Convert datetime series to decimal year (e.g. 2024.5)."""
    year = dt.dt.year
    doy = dt.dt.dayofyear
    # 365.25 is fine for visualization and basic regression
    return year + (doy - 1) / 365.25


def t_critical_approx(df: int, alpha: float = 0.05) -> float:
    try:
        from scipy.stats import t  # type: ignore

        return float(t.ppf(1 - alpha / 2, df))
    except Exception:
        return 1.96


def fetch_scansan_json(base_url: str, path: str, api_key: str) -> dict:
    url = base_url.rstrip("/") + path
    headers = {"X-Auth-Token": api_key}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_historical_response(resp: dict) -> pd.DataFrame:
    data = resp.get("data", [])
    rows = []
    for prop in data:
        addr = prop.get("property_address")
        for v in prop.get("valuations", []) or []:
            rows.append(
                {
                    "property_address": addr,
                    "date": v.get("date"),
                    "valuation": v.get("valuation"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    df["valuation"] = pd.to_numeric(df["valuation"], errors="coerce")
    df = df.dropna(subset=["property_address", "date", "valuation"]).sort_values(["property_address", "date"])
    return df


def parse_current_response(resp: dict) -> pd.DataFrame:

    data = resp.get("data", [])
    rows = []
    for prop in data:
        rows.append(
            {
                "property_address": prop.get("property_address"),
                "last_sold_price": prop.get("last_sold_price"),
                "last_sold_date": prop.get("last_sold_date"),
                "lower_outlier": prop.get("lower_outlier"),
                "upper_outlier": prop.get("upper_outlier"),
                "bounded_low": (prop.get("bounded_valuation") or [None, None])[0],
                "bounded_high": (prop.get("bounded_valuation") or [None, None])[1],
            }
        )
    return pd.DataFrame(rows)


def fit_log_linear_regression(t: np.ndarray, p: np.ndarray) -> FitResult:

    if len(t) < 3:
        raise ValueError("Need at least 3 points to fit and compute variance (n-2 > 0).")

    y = np.log(p)
    X = np.column_stack([np.ones_like(t), t])
    # beta = (X'X)^-1 X'y
    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (X.T @ y)

    yhat = X @ beta
    resid = y - yhat
    n = len(t)
    sse = float(np.sum(resid**2))
    sigma2 = sse / (n - 2)

    return FitResult(beta0=float(beta[0]), beta1=float(beta[1]), sigma2=float(sigma2), xtx_inv=xtx_inv, n=n)


def predict_with_interval(
    fit: FitResult,
    t_new: np.ndarray,
    alpha: float = 0.05,
    interval: str = "prediction",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x = np.column_stack([np.ones_like(t_new), t_new])
    y_mean = fit.beta0 + fit.beta1 * t_new

    # Var(mean) = sigma^2 * x'(X'X)^-1 x
    # Var(pred) = sigma^2 * (1 + x'(X'X)^-1 x)
    quad = np.einsum("ij,jk,ik->i", x, fit.xtx_inv, x)  # x_i^T * inv * x_i

    if interval == "mean":
        var = fit.sigma2 * quad
    else:
        var = fit.sigma2 * (1.0 + quad)

    se = np.sqrt(var)
    tcrit = t_critical_approx(df=fit.n - 2, alpha=alpha)

    y_lo = y_mean - tcrit * se
    y_hi = y_mean + tcrit * se

    mid = np.exp(y_mean)
    lo = np.exp(y_lo)
    hi = np.exp(y_hi)
    return mid, lo, hi


def metrics_in_sample(t: np.ndarray, p: np.ndarray, fit: FitResult) -> dict:
    y = np.log(p)
    yhat = fit.beta0 + fit.beta1 * t
    resid = y - yhat

    sse = float(np.sum(resid**2))
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")

    # RMSE in log space
    rmse_log = float(np.sqrt(np.mean(resid**2)))

    # RMSE in price space (approx; back-transform)
    phat = np.exp(yhat)
    rmse_price = float(np.sqrt(np.mean((phat - p) ** 2)))
    mae_price = float(np.mean(np.abs(phat - p)))

    return {
        "SSE (log space)": sse,
        "R² (log space)": r2,
        "RMSE (log space)": rmse_log,
        "RMSE (£)": rmse_price,
        "MAE (£)": mae_price,
    }



st.set_page_config(page_title="Property Valuation Forecast (ScanSan)", layout="wide")
st.title("Property valuation forecast: scatter + log-linear regression + uncertainty band")

with st.sidebar:
    st.header("API")
    api_key = st.text_input("ScanSan API key (X-Auth-Token)", type="password")
    base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)
    postcode = st.text_input("Postcode / area_code_postal", value="KT9 1AY")

    st.header("Model")
    interval_type = st.selectbox(
        "Uncertainty band type",
        options=["prediction", "mean"],
        index=0,
        help=(
            "prediction = band for a future observed valuation (wider). "
            "mean = band for the trend mean only (narrower)."
        ),
    )
    alpha = st.slider("Significance level (alpha)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

    fetch_btn = st.button("Fetch data")

if not fetch_btn:
    st.info("Enter API key + postcode, then click **Fetch data**.")
    st.stop()

if not api_key.strip():
    st.error("API key is required.")
    st.stop()


try:
    hist_json = fetch_scansan_json(base_url, HIST_ENDPOINT.format(area_code_postal=postcode), api_key)
    df_hist = parse_historical_response(hist_json)

    curr_json = fetch_scansan_json(base_url, CURR_ENDPOINT.format(area_code_postal=postcode), api_key)
    df_curr = parse_current_response(curr_json)

except requests.HTTPError as e:
    st.error(f"API request failed: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error fetching/parsing data: {e}")
    st.stop()

if df_hist.empty:
    st.warning("No historical valuations returned for this postcode.")
    st.stop()

# Property selection
addresses = sorted(df_hist["property_address"].unique().tolist())
addr = st.selectbox("Select property address", options=addresses)

prop = df_hist[df_hist["property_address"] == addr].copy()
prop = prop.sort_values("date")
prop = prop.dropna(subset=["date", "valuation"])

if len(prop) < 3:
    st.warning("Not enough data points for this property (need at least 3).")
    st.stop()

# Show current summary if available
curr_row = None
if not df_curr.empty:
    match = df_curr[df_curr["property_address"] == addr]
    if len(match) > 0:
        curr_row = match.iloc[0].to_dict()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Data preview")
    st.dataframe(prop.tail(12), use_container_width=True)

with right:
    st.subheader("Current valuation summary")
    if curr_row:
        st.write(
            {
                "bounded_low": curr_row.get("bounded_low"),
                "bounded_high": curr_row.get("bounded_high"),
                "lower_outlier": curr_row.get("lower_outlier"),
                "upper_outlier": curr_row.get("upper_outlier"),
                "last_sold_price": curr_row.get("last_sold_price"),
                "last_sold_date": curr_row.get("last_sold_date"),
            }
        )
    else:
        st.write("No current summary found for this address in current valuations response.")


prop["year"] = decimal_year(prop["date"])
min_year = int(math.floor(prop["year"].min()))
max_year = int(math.ceil(prop["year"].max()))
default_start = max(min_year, max_year - 25)

colA, colB, colC = st.columns(3)
with colA:
    train_start = st.number_input("Training start year", min_value=min_year, max_value=max_year, value=default_start)
with colB:
    train_end = st.number_input("Training end year", min_value=min_year, max_value=max_year, value=max_year)
with colC:
    forecast_end = st.number_input("Forecast end year", min_value=max_year, max_value=max_year + 50, value=max_year + 15)

if train_start >= train_end:
    st.error("Training start must be < training end.")
    st.stop()

train_mask = (prop["year"] >= train_start) & (prop["year"] <= train_end)
train = prop[train_mask].copy()

if len(train) < 3:
    st.error("Selected training window has fewer than 3 points. Choose a wider window.")
    st.stop()


t_train = train["year"].to_numpy(dtype=float)
p_train = train["valuation"].to_numpy(dtype=float)

try:
    fit = fit_log_linear_regression(t_train, p_train)
except Exception as e:
    st.error(f"Model fit failed: {e}")
    st.stop()

# Prediction grid (monthly for smooth plot)
start_date = prop["date"].min()
end_date = pd.Timestamp(year=int(forecast_end), month=12, day=31)
grid_dates = pd.date_range(start=start_date, end=end_date, freq="MS")
grid_years = decimal_year(pd.Series(grid_dates))

mid, lo, hi = predict_with_interval(
    fit,
    grid_years.to_numpy(dtype=float),
    alpha=float(alpha),
    interval=interval_type,
)


m = metrics_in_sample(t_train, p_train, fit)


last_obs_date = prop["date"].max()
train_start_date = pd.Timestamp(year=int(train_start), month=1, day=1)
train_end_date = pd.Timestamp(year=int(train_end), month=12, day=31)

fig = go.Figure()

# Training window shading
fig.add_vrect(
    x0=train_start_date,
    x1=train_end_date,
    fillcolor="rgba(0,0,0,0.06)",
    line_width=0,
    layer="below",
)

# Scatter: historical valuations
fig.add_trace(
    go.Scatter(
        x=prop["date"],
        y=prop["valuation"],
        mode="markers",
        name="Historical valuations",
        marker=dict(size=6),
        hovertemplate="Date=%{x|%Y-%m-%d}<br>Valuation=£%{y:,.0f}<extra></extra>",
    )
)

# Interval band (lo->hi)
fig.add_trace(
    go.Scatter(
        x=grid_dates,
        y=hi,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=grid_dates,
        y=lo,
        mode="lines",
        fill="tonexty",
        name=f"{int((1-alpha)*100)}% {interval_type} interval",
        hoverinfo="skip",
        line=dict(width=0),
    )
)

# Forecast line (mid)
fig.add_trace(
    go.Scatter(
        x=grid_dates,
        y=mid,
        mode="lines",
        name="Trend + forecast",
        line=dict(width=3),
        hovertemplate="Date=%{x|%Y-%m-%d}<br>Predicted=£%{y:,.0f}<extra></extra>",
    )
)

# Vertical line at last observation
fig.add_vline(x=last_obs_date, line_dash="dash", line_width=2)

fig.update_layout(
    title="Scatter + log-linear regression forecast with uncertainty band",
    xaxis_title="Date",
    yaxis_title="Valuation (£)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=50, b=10),
)

st.subheader("Chart")
st.plotly_chart(fig, use_container_width=True)


st.subheader("Model summary (transparent)")
growth = math.exp(fit.beta1) - 1.0
st.write(
    {
        "Model": "log(price) = beta0 + beta1 * year + noise",
        "beta0": fit.beta0,
        "beta1": fit.beta1,
        "Implied avg annual growth (approx)": f"{growth*100:.2f}%",
        "Training points": fit.n,
        "Residual variance (sigma^2, log space)": fit.sigma2,
        "Interval type": interval_type,
        "Confidence": f"{int((1-alpha)*100)}%",
    }
)

st.subheader("Fit metrics on selected training window")
st.write(m)

st.caption(
    "Notes: R² and SSE above are computed in log(price) space. "
    "MAE/RMSE in £ are computed after back-transforming predictions."
)