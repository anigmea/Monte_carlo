import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

import streamlit as st


# -----------------------------
# Core helper functions
# -----------------------------

@st.cache_data(show_spinner=False)
def download_prices(tickers, period="1y"):
    if isinstance(tickers, str):
        tickers = [tickers]
    df = yf.download(tickers, period=period)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df / df.shift(1)).dropna()


def simulate_gbm_paths(mu, cov, S0, n_steps=200, n_sims=10_000, dt=1 / 252):
    """
    Multi-asset GBM Monte Carlo using Cholesky for correlated shocks.
    mu: (n_assets,)
    cov: (n_assets, n_assets)
    S0: (n_assets,)
    Returns array of shape (n_sims, n_steps, n_assets)
    """
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    S0 = np.asarray(S0)
    n_assets = len(S0)

    L = np.linalg.cholesky(cov)
    vol = np.sqrt(np.diag(cov))

    output = np.zeros((n_sims, n_steps, n_assets))
    for i in range(n_sims):
        S = S0.copy()
        for t in range(n_steps):
            Z = np.random.normal(size=n_assets)
            correlated_Z = L @ Z
            S = S * np.exp((mu - 0.5 * vol**2) * dt + correlated_Z * vol * np.sqrt(dt))
            output[i, t, :] = S
    return output


def portfolio_var(weights, returns, alpha=5):
    """Return negative VaR (so we can minimize)."""
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, alpha)
    return -var


def portfolio_cvar(weights, returns, alpha=5):
    """Return negative CVaR (so we can minimize)."""
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, alpha)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return -cvar


def optimize_portfolio(sim_asset_returns, risk_measure="CVaR", alpha=5):
    """
    Optimize portfolio weights using simulated asset returns.
    risk_measure: 'CVaR' or 'VaR'
    """
    n_assets = sim_asset_returns.shape[1]
    if risk_measure == "CVaR":
        obj = portfolio_cvar
    else:
        obj = portfolio_var

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]
    initial_weights = np.ones(n_assets) / n_assets

    result = minimize(obj, initial_weights, args=(sim_asset_returns,), bounds=bounds, constraints=constraints)
    return result


# -----------------------------
# Plot helpers
# -----------------------------

def plot_single_ticker_paths(prices, ticker, n_steps=200, n_sims=10_000, dt=1 / 252):
    df = prices.copy()
    df["Return"] = (df["Close"] / df["Close"].shift(1)).apply(lambda x: np.log(x))
    growth_rate = df["Return"].mean()
    std = df["Return"].std()

    output = np.zeros((n_sims, n_steps))
    S0 = df["Close"].iloc[-1]

    for i in range(n_sims):
        S = S0
        for j in range(n_steps):
            Z = np.random.normal()
            S = S * np.exp((growth_rate - 0.5 * std**2) * dt + std * np.sqrt(dt) * Z)
            output[i, j] = S

    time = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Confidence interval
    p_lower = np.percentile(output, 2.5, axis=0)
    p_upper = np.percentile(output, 97.5, axis=0)
    ax.fill_between(time, p_lower, p_upper, color="green", alpha=0.2, label="95% CI")

    # Sample paths
    for path in output[:100]:  # limit for clarity
        ax.plot(time, path, color="blue", alpha=0.05)

    mean_path = output.mean(axis=0)
    ax.plot(time, mean_path, color="red", linewidth=2, label="Mean Path")

    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker} Monte Carlo {n_steps}-Day Forecast")
    ax.legend()

    return fig, output


def plot_asset_paths(time, output, tickers):
    figs = []
    for idx, ticker in enumerate(tickers):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Sample paths
        sample_indices = np.random.choice(np.arange(output.shape[0]), size=min(50, output.shape[0]), replace=False)
        for path in sample_indices:
            ax.plot(time, output[path, :, idx], color="blue", alpha=0.05)

        mean_path = output[:, :, idx].mean(axis=0)
        ax.plot(time, mean_path, color="red", linewidth=2, label="Mean Path")

        p_lower = np.percentile(output[:, :, idx], 2.5, axis=0)
        p_upper = np.percentile(output[:, :, idx], 97.5, axis=0)
        ax.fill_between(time, p_lower, p_upper, color="green", alpha=0.2, label="95% CI")

        ax.set_title(f"{ticker} — {len(time)}-Day Monte Carlo Simulation")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.3)

        figs.append(fig)
    return figs


def plot_portfolio_distribution(portfolio_returns, alpha=5, stress_loss=None, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(portfolio_returns, bins=50, alpha=0.7, color="skyblue", label="Simulated Portfolio Returns")
    var_level = np.percentile(portfolio_returns, alpha)
    ax.axvline(var_level, color="red", linestyle="--", label=f"{alpha}% VaR")
    if stress_loss is not None:
        ax.axvline(stress_loss, color="black", linestyle="-", linewidth=2, label="Stress Scenario")
    ax.set_title(f"Portfolio Returns Distribution {title_suffix}".strip())
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Monte Carlo & CVaR Portfolio Toolkit", layout="wide")
    st.title("Monte Carlo Simulation & Portfolio Risk Toolkit")

    st.sidebar.header("Configuration")
    app_mode = st.sidebar.radio(
        "Mode",
        ["Single Ticker Simulation", "Multi-Asset Portfolio Optimization"],
    )

    period = st.sidebar.selectbox("History period", ["6mo", "1y", "2y", "5y"], index=1)
    n_steps = st.sidebar.slider("Forecast horizon (days)", min_value=50, max_value=365, value=200, step=10)
    n_sims = st.sidebar.slider("Number of Monte Carlo paths", min_value=1000, max_value=20000, value=10_000, step=1000)

    if app_mode == "Single Ticker Simulation":
        st.subheader("Single Ticker Monte Carlo Forecast")
        ticker = st.text_input("Ticker symbol", value="AAPL")

        if st.button("Run simulation"):
            with st.spinner("Downloading data and running simulation..."):
                try:
                    raw_df = yf.download(ticker, period=period)
                    if raw_df.empty:
                        st.error("No data returned for this ticker/period.")
                        return
                    df = raw_df[["Close"]].dropna()
                    fig, output = plot_single_ticker_paths(df, ticker, n_steps=n_steps, n_sims=n_sims)
                    st.pyplot(fig)
                    st.write(
                        f"**Simulated price range** over {n_steps} days: "
                        f"min={output.min():.2f}, max={output.max():.2f}"
                    )
                except Exception as e:
                    st.error(f"Simulation failed: {e}")

    else:
        st.subheader("Multi-Asset Portfolio Monte Carlo & Optimization")

        default_tickers = [
            "VTI",
            "VWO",
            "VEA",
            "BND",
            "NVDA",
            "TSLA",
            "AAPL",
            "GOOGL",
            "LMT",
            "RMBS",
            "PLTR",
            "MSFT",
            "PEP",
            "AVGO",
            "SOFI",
            "OKLO",
            "QSI",
        ]

        tickers_text = st.text_area(
            "Tickers (comma-separated)",
            value=", ".join(default_tickers),
            help="Edit this list to change the investable universe.",
        )
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

        col1, col2, col3 = st.columns(3)
        with col1:
            risk_measure = st.selectbox("Optimize for", ["CVaR", "VaR"])
        with col2:
            alpha = st.slider("Tail percentile (α)", min_value=1, max_value=10, value=5)
        with col3:
            rf_annual = st.number_input("Risk-free rate (annual)", min_value=0.0, value=0.0355, step=0.005)

        stress_text = st.text_input(
            "Stress scenario (% change per asset, comma-separated; optional)",
            value="-10, -15, -12, -2, -25, -22, -18, -16, -10, -5, -8, -12, -7, -9, -20, -18, -15",
        )

        if st.button("Run portfolio simulation & optimization"):
            with st.spinner("Downloading data and running Monte Carlo..."):
                try:
                    df = download_prices(tickers, period=period)
                    returns = compute_log_returns(df)

                    mu = returns.mean().values
                    cov = returns.cov().values
                    S0 = df.iloc[-1].values

                    output = simulate_gbm_paths(mu, cov, S0, n_steps=n_steps, n_sims=n_sims)
                    time = np.arange(1, n_steps + 1)

                    # Asset-level simulated returns
                    sim_asset_returns = output[:, -1, :] / output[:, 0, :] - 1

                    # Plot asset paths
                    with st.expander("Asset-level Monte Carlo paths", expanded=False):
                        figs = plot_asset_paths(time, output, tickers)
                        for fig in figs:
                            st.pyplot(fig)

                    # Random portfolio stats (for reference)
                    rand_weights = np.random.rand(len(tickers))
                    rand_weights /= rand_weights.sum()
                    portfolio_values = np.sum(output * rand_weights, axis=2)
                    final_returns = portfolio_values[:, -1] / portfolio_values[:, 0] - 1

                    rf_daily = rf_annual / 252
                    VaR_rand = np.percentile(final_returns, alpha)
                    CVaR_rand = final_returns[final_returns <= VaR_rand].mean()
                    sharpe_rand = (final_returns.mean() - rf_daily) / final_returns.std()

                    st.markdown("### Random Portfolio (benchmark)")
                    st.write(
                        f"**Sharpe**: {sharpe_rand:.2f}  |  "
                        f"**{alpha}% VaR**: {VaR_rand:.4f}  |  "
                        f"**{alpha}% CVaR**: {CVaR_rand:.4f}"
                    )

                    # Optimize portfolio
                    result = optimize_portfolio(sim_asset_returns, risk_measure=risk_measure, alpha=alpha)
                    opt_weights = result.x
                    port_ret = sim_asset_returns @ opt_weights

                    VaR_opt = np.percentile(port_ret, alpha)
                    CVaR_opt = port_ret[port_ret <= VaR_opt].mean()
                    sharpe_opt = (port_ret.mean() - rf_daily) / port_ret.std()

                    st.markdown("### Optimized Portfolio")
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        st.metric("Sharpe Ratio", f"{sharpe_opt:.2f}")
                    with metrics_cols[1]:
                        st.metric(f"{alpha}% VaR", f"{VaR_opt:.4f}")
                    with metrics_cols[2]:
                        st.metric(f"{alpha}% CVaR", f"{CVaR_opt:.4f}")

                    weights_df = pd.DataFrame(
                        {"Ticker": tickers, "Weight": opt_weights},
                    ).sort_values("Weight", ascending=False)
                    st.dataframe(weights_df.style.format({"Weight": "{:.3f}"}), use_container_width=True)

                    # Optional stress test
                    stress_loss = None
                    try:
                        stress_vals = [float(x) / 100 for x in stress_text.split(",")]
                        if len(stress_vals) == len(tickers):
                            stress_vec = np.array(stress_vals)
                            stress_loss = float(stress_vec @ opt_weights)
                            st.write(f"**Stress scenario loss:** {stress_loss * 100:.2f}%")
                        else:
                            st.info("Stress vector length does not match number of tickers; skipping stress test.")
                    except Exception:
                        st.info("Could not parse stress scenario; skipping.")

                    # Distribution plot
                    dist_fig = plot_portfolio_distribution(
                        port_ret,
                        alpha=alpha,
                        stress_loss=stress_loss,
                        title_suffix=f"({risk_measure}-optimized)",
                    )
                    st.pyplot(dist_fig)

                except Exception as e:
                    st.error(f"Portfolio simulation failed: {e}")


if __name__ == "__main__":
    main()


