# FinPilot Robo-Advisor

A student FinTech prototype for COM3115, built with Python and Streamlit.

FinPilot explores historical asset performance, groups assets using K-Means,
and calculates portfolio allocations based on a user-selected risk preference.

## Features

- Historical price and return analysis for 12 assets
- Annualized volatility and asset performance comparisons
- K-Means clustering with standardized return and volatility features
- Silhouette score for clustering evaluation
- Investment amount and risk preference inputs
- Mean-variance portfolio optimization
- Long-only portfolios with a 30% maximum weight per asset
- Markowitz efficient frontier
- Chronological 70/30 training and testing split
- Test-period comparison against an equal-weight portfolio

## Technology

Python 3.12, Streamlit, Pandas, NumPy, scikit-learn, SciPy, and yfinance.

## Project Files

- app.py: dashboard, analysis, optimization, and evaluation
- download_data.py: downloads and validates historical prices
- data/prices.csv: saved adjusted closing prices
- requirements.txt: Python package versions
- README.md: project overview and setup instructions

## Setup on Windows

Open a terminal in the project folder.

Create a Python environment:

    py -3.12 -m venv .venv

Install the required packages:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Start the application:

    .\.venv\Scripts\python.exe -m streamlit run app.py

Open the Local URL shown in the terminal, usually:

    http://localhost:8501

Keep the terminal running while using the application.

## Data

The dashboard reads data/prices.csv and does not download prices on each run.
Keep the supplied CSV file to reproduce the demonstration results.

To refresh the data, run:

    .\.venv\Scripts\python.exe download_data.py

This requires internet access. A successful download replaces the CSV.
Refreshing the data can change the analysis and portfolio results.

Source: Yahoo Finance via yfinance.
The project uses historical adjusted daily closing prices, not live quotes.

## Dashboard Tabs

1. Market Overview: individual assets and performance comparisons
2. AI Clustering: asset groups, standardized features, and evaluation
3. Portfolio Builder: allocations, estimated returns, risk, and efficient frontier
4. Performance Test: training-period weights and test-period results

## Methods and Assumptions

K-Means uses standardized cumulative return and annualized volatility.
Cluster labels describe relative historical volatility within this asset sample.

Portfolio optimization minimizes:

    risk-aversion coefficient × annualized variance − estimated annualized return

The demonstration coefficients are 20 for Conservative, 5 for Balanced,
and 1 for Growth. They are illustrative settings, not a validated
investor suitability assessment.

Portfolio weights sum to 100%. Short selling is not allowed.
Each asset has a maximum allocation of 30%.

Annualized mean returns and covariance use 252 trading days per year.

The Portfolio Builder uses the full dataset.
The Performance Test estimates new weights using only the first 70%
of daily returns and evaluates them on the remaining 30%.

The test assumes daily rebalancing to fixed target weights.
Transaction costs, taxes, and slippage are excluded.

## Limitations

- Historical estimates do not guarantee future returns.
- Clustering evaluates asset similarity, not future investment performance.
- One small asset sample and one test period provide limited evidence.
- The project does not execute trades or connect to brokerage accounts.
- This is an educational prototype, not personalized financial advice.