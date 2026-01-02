## Monte Carlo & Portfolio Risk Toolkit

This project combines your previous notebooks into a single, interactive app for:

- **Single-ticker Monte Carlo simulation**
- **Multi-asset portfolio Monte Carlo simulation**
- **Portfolio optimization by VaR or CVaR**
- **Optional stress testing of the optimized portfolio**

### Setup

From the project directory:

```bash
pip install -r requirements.txt
```

### Run the UI

From the same directory:

```bash
streamlit run app.py
```

Then open the URL that Streamlit prints in your terminal (usually `http://localhost:8501`).

### Features

- **Single Ticker Simulation**
  - Enter a ticker (e.g., `AAPL`)
  - Choose lookback period, horizon, and number of paths
  - See price paths, mean path, and 95% confidence band

- **Multi-Asset Portfolio Optimization**
  - Edit the ticker universe (comma-separated)
  - Configure horizon, number of simulations, and risk metric (VaR or CVaR)
  - Run correlated Monte Carlo using GBM
  - View asset-level paths, random-portfolio stats, and optimized weights
  - Optional stress-scenario loss and distribution plot with VaR / stress markers


