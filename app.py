import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import minimize

st.set_page_config(page_title="FinPilot Robo-Advisor", layout="wide")

st.title("FinPilot Robo-Advisor")
st.write("Explore historical asset performance, risk, and clustering.")
st.sidebar.header("Investment Settings")

investment_amount = st.sidebar.number_input(
    "Investment Amount (USD)",
    min_value=100.0,
    max_value=10000000.0,
    value=10000.0,
    step=100.0
)

st.sidebar.caption(
    "This amount will be used to calculate the allocation in USD."
)
risk_preference = st.sidebar.selectbox(
    "Risk Preference",
    ["Conservative", "Balanced", "Growth"]
)

st.sidebar.caption(
    "Conservative prioritizes lower volatility. "
    "Balanced and Growth place increasing emphasis on estimated returns."
)
asset_codes = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Alphabet (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "JPMorgan Chase (JPM)": "JPM",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Exxon Mobil (XOM)": "XOM",
    "Gold ETF (GLD)": "GLD",
    "Long-Term Treasury ETF (TLT)": "TLT",
    "Short-Term Treasury ETF (SHY)": "SHY"
}

prices = pd.read_csv(
    "data/prices.csv",
    index_col=0,
    parse_dates=True
)

st.caption(
    "Source: Yahoo Finance via yfinance. Loaded from a local CSV file."
)

st.caption(
    f"Data period: {prices.index.min():%Y-%m-%d} to "
    f"{prices.index.max():%Y-%m-%d} | "
    f"Trading days: {len(prices)} | Assets: {len(prices.columns)}"
)

market_tab, ai_tab, portfolio_tab, test_tab = st.tabs([
    "Market Overview", "AI Clustering", "Portfolio Builder", "Performance Test"
])

with market_tab:
    asset = st.selectbox("Select an asset", list(asset_codes.keys()))
    ticker = asset_codes[asset]

    st.write("Selected asset:", asset)

    with st.expander("View price data"):
        st.subheader(f"{ticker} Historical Adjusted Closing Prices")
        st.dataframe(prices[[ticker]].head())

    st.subheader(f"{ticker} Price Trend")
    st.line_chart(prices[ticker])
    st.caption("Y-axis: Adjusted closing price (USD). X-axis: Date.")

    start_price = prices[ticker].iloc[0]
    end_price = prices[ticker].iloc[-1]
    total_return = (end_price / start_price - 1) * 100

    st.metric("Cumulative Return", f"{total_return:.2f}%")

    daily_returns = prices[ticker].pct_change(fill_method=None).dropna()

    with st.expander("View daily returns"):
        st.subheader("Daily Returns (First 5 Observations)")
        st.dataframe(
            (daily_returns.head() * 100).to_frame("Daily Return (%)")
        )

    annual_volatility = daily_returns.std() * (252 ** 0.5)
    st.metric("Annualized Volatility", f"{annual_volatility:.2%}")

    st.subheader("Asset Performance Comparison (Base = 100)")

    comparison_prices = prices.dropna()
    normalized_prices = (
        comparison_prices / comparison_prices.iloc[0] * 100
    )

    st.line_chart(normalized_prices)

    all_returns = comparison_prices.pct_change(fill_method=None).dropna()

    summary = pd.DataFrame({
        "Cumulative Return (%)": (
            comparison_prices.iloc[-1] / comparison_prices.iloc[0] - 1
        ) * 100,
        "Annualized Volatility (%)": (
            all_returns.std() * (252 ** 0.5) * 100
        )
    })

    st.subheader("Asset Return and Risk Comparison")
    st.dataframe(summary.round(2))

with ai_tab:
    features = summary.copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    with st.expander("View standardized features"):
        st.subheader("Clustering Preparation: Standardized Features")
        st.dataframe(
            pd.DataFrame(
                scaled_features,
                index=features.index,
                columns=[
                    "Standardized Cumulative Return",
                    "Standardized Annualized Volatility"
                ]
            ).round(2)
        )

        st.caption(
            "These values are standardized scores, not percentages. "
            "Negative values indicate observations below the feature mean."
        )

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    cluster_result = summary.copy()
    cluster_result["Cluster ID"] = cluster_labels


    cluster_summary = cluster_result.groupby("Cluster ID")[
        ["Cumulative Return (%)", "Annualized Volatility (%)"]
    ].mean()

    cluster_summary.columns = [
        "Mean Cumulative Return (%)",
        "Mean Annualized Volatility (%)"
    ]

    cluster_summary["Number of Assets"] = (
        cluster_result.groupby("Cluster ID").size()
    )

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary.round(2))

    cluster_order = cluster_summary[
        "Mean Annualized Volatility (%)"
    ].sort_values().index

    risk_labels = {
        cluster_order[0]: "Lower Volatility",
        cluster_order[1]: "Medium Volatility",
        cluster_order[2]: "Higher Volatility"
    }

    cluster_result["Volatility Group"] = (
        cluster_result["Cluster ID"].map(risk_labels)
    )

    st.subheader("Assets by Historical Volatility Group")
    st.dataframe(cluster_result.round(2))

    st.caption(
        "Groups are ranked by their mean historical volatility within this sample. "
        "K-Means uses both return and volatility; these labels are not credit ratings "
        "or predictions of future risk."
    )

    plot_data = cluster_result.reset_index()
    plot_data = plot_data.rename(
        columns={plot_data.columns[0]: "Asset"}
    )

    st.subheader("Asset Return and Volatility Clusters")

    st.scatter_chart(
        plot_data,
        x="Annualized Volatility (%)",
        y="Cumulative Return (%)",
        color="Volatility Group"
    )


    from sklearn.metrics import silhouette_score

    silhouette = silhouette_score(scaled_features, cluster_labels)

    st.subheader("Clustering Evaluation")
    st.metric("Silhouette Score", f"{silhouette:.3f}")
    st.caption(
        "Scores range from -1 to 1. Higher values indicate more compact "
        "and better-separated clusters in the selected feature space. "
        "This score does not measure investment performance."
    )

with portfolio_tab:
    annual_covariance = all_returns.cov() * 252
    annual_returns = all_returns.mean() * 252

    with st.expander("View return estimates and covariance"):
        st.subheader("Historical Annualized Return Estimates")
        st.dataframe(
            (annual_returns * 100)
            .to_frame("Annualized Mean Return (%)")
            .round(2)
        )

        st.caption(
            "Estimated as mean daily return multiplied by 252. "
            "These historical estimates are not guaranteed future returns."
        )

        st.subheader("Annualized Return Covariance Matrix")
        st.dataframe(annual_covariance.round(4))

        st.caption(
            "Calculated from daily returns using 252 trading days per year. "
            "Values are not percentages."
        )

    asset_count = len(all_returns.columns)
    equal_weights = np.ones(asset_count) / asset_count

    portfolio_variance = (
        equal_weights @ annual_covariance.to_numpy() @ equal_weights
    )
    portfolio_volatility = np.sqrt(portfolio_variance)

    with st.expander("View equal-weight benchmark"):
        st.subheader("Equal-Weight Portfolio Benchmark")
        st.write(f"Number of assets: {asset_count}")
        st.write(f"Weight per asset: {1 / asset_count:.2%}")

        st.metric(
            "Estimated Annualized Portfolio Volatility",
            f"{portfolio_volatility:.2%}"
        )

    def calculate_portfolio_volatility(weights):
        covariance = annual_covariance.to_numpy()
        variance = weights @ covariance @ weights
        return np.sqrt(variance)


    check_volatility = calculate_portfolio_volatility(equal_weights)



    risk_aversion = {
        "Conservative": 20.0,
        "Balanced": 5.0,
        "Growth": 1.0
    }


    def portfolio_objective(weights):
        estimated_return = weights @ annual_returns.to_numpy()
        variance = weights @ annual_covariance.to_numpy() @ weights
        penalty = risk_aversion[risk_preference] * variance
        return penalty - estimated_return


    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    bounds = [(0, 0.30)] * asset_count

    result = minimize(
        portfolio_objective,
        x0=equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    st.subheader(f"{risk_preference} Portfolio (30% Asset Cap)")

    if result.success:
        optimized_volatility = calculate_portfolio_volatility(result.x)
        optimized_return = result.x @ annual_returns.to_numpy()

        st.metric(
            "Estimated Annualized Return",
            f"{optimized_return:.2%}"
        )

        st.caption(
            "Return estimated from historical mean daily returns. "
            "This is not a guaranteed future return."
        )

        st.metric(
            "Estimated Annualized Volatility",
            f"{optimized_volatility:.2%}"
        )

        optimized_weights = pd.DataFrame({
            "Asset": all_returns.columns,
            "Weight (%)": result.x * 100,
            "Allocation (USD)": result.x * investment_amount
        })

        optimized_weights = optimized_weights.sort_values(
            "Weight (%)",
            ascending=False
        )

        st.subheader("Optimized Portfolio Allocation")
        st.dataframe(optimized_weights.round(2))
        st.write(f"Total weight: {result.x.sum():.2%}")

        st.subheader("Portfolio Allocation Chart")

        allocation_chart = optimized_weights.set_index("Asset")
        st.bar_chart(allocation_chart["Weight (%)"])

        st.caption(
            "Portfolio weights sum to 100%. "
            "Each asset is limited to a maximum weight of 30%."
        )

        risk_comparison = pd.DataFrame({
            "Portfolio": [
                "Equal Weight",
                f"{risk_preference} (30% Cap)"
            ],
            "Estimated Annualized Return (%)": [
                (equal_weights @ annual_returns.to_numpy()) * 100,
                optimized_return * 100
            ],
            "Estimated Annualized Volatility (%)": [
                portfolio_volatility * 100,
                optimized_volatility * 100
            ]
        })

        st.subheader("Portfolio Return and Risk Comparison")
        st.dataframe(risk_comparison.round(2), hide_index=True)

        st.caption(
            "Both estimates use the same historical data. "
            "This is an in-sample comparison, not an out-of-sample backtest."
        )
    else:
        st.error(f"Optimization failed: {result.message}")
    def negative_portfolio_return(weights):
        return -(weights @ annual_returns.to_numpy())


    max_return_result = minimize(
        negative_portfolio_return,
        x0=equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    with st.expander("View frontier endpoints"):
        st.subheader("Efficient Frontier Preparation")

        if max_return_result.success:
            max_return = max_return_result.x @ annual_returns.to_numpy()

            st.metric(
                "Maximum Estimated Annualized Return (30% Asset Cap)",
                f"{max_return:.2%}"
            )
        else:
            st.error(f"Calculation failed: {max_return_result.message}")
        min_risk_result = minimize(
            calculate_portfolio_volatility,
            x0=equal_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if min_risk_result.success:
            frontier_start_return = (
                min_risk_result.x @ annual_returns.to_numpy()
            )
            frontier_start_volatility = calculate_portfolio_volatility(
                min_risk_result.x
            )

            st.subheader("Efficient Frontier Starting Point")
            st.metric(
                "Estimated Annualized Return",
                f"{frontier_start_return:.2%}"
            )
            st.metric(
                "Estimated Annualized Volatility",
                f"{frontier_start_volatility:.2%}"
            )
        else:
            st.error(f"Calculation failed: {min_risk_result.message}")

    if min_risk_result.success and max_return_result.success:
        target_returns = np.linspace(
            frontier_start_return, max_return, 20
        )
        frontier_rows = []

        for target in target_returns:
            frontier_constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {
                    "type": "eq",
                    "fun": lambda w, target_value=target: (
                                                                  w @ annual_returns.to_numpy()
                                                          ) - target_value
                }
            ]

            frontier_result = minimize(
                calculate_portfolio_volatility,
                x0=equal_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=frontier_constraints,
                options={"maxiter": 1000, "ftol": 1e-9}
            )

            if frontier_result.success:
                frontier_rows.append({
                    "Annualized Volatility (%)":
                        calculate_portfolio_volatility(
                            frontier_result.x
                        ) * 100,
                    "Estimated Annualized Return (%)":
                        (frontier_result.x @ annual_returns.to_numpy()) * 100
                })

        frontier_df = pd.DataFrame(frontier_rows)

        with st.expander("View calculated frontier portfolios"):
            st.write(f"Successfully calculated: {len(frontier_df)} / 20")
            st.dataframe(frontier_df.round(2))
    if min_risk_result.success and max_return_result.success:
        if not frontier_df.empty:
            st.subheader("Markowitz Efficient Frontier (30% Asset Cap)")

            st.line_chart(
                frontier_df,
                x="Annualized Volatility (%)",
                y="Estimated Annualized Return (%)"
            )

            st.caption(
                "Each point represents the minimum-volatility portfolio "
                "for a given target return under the same constraints. "
                "The curve connects 20 calculated portfolios using historical estimates."
            )

with test_tab:
    split_point = int(len(all_returns) * 0.7)

    train_returns = all_returns.iloc[:split_point]
    test_returns = all_returns.iloc[split_point:]

    st.subheader("Out-of-Sample Test: Data Split")

    st.write(
        f"Training: {train_returns.index.min():%Y-%m-%d} to "
        f"{train_returns.index.max():%Y-%m-%d} "
        f"({len(train_returns)} trading days)"
    )

    st.write(
        f"Testing: {test_returns.index.min():%Y-%m-%d} to "
        f"{test_returns.index.max():%Y-%m-%d} "
        f"({len(test_returns)} trading days)"
    )

    st.caption(
        "The first 70% of daily returns will be used to estimate portfolio weights. "
        "The remaining 30% will be reserved for evaluation."
    )
    train_annual_returns = train_returns.mean() * 252
    train_covariance = train_returns.cov() * 252


    def training_objective(weights):
        estimated_return = weights @ train_annual_returns.to_numpy()
        variance = weights @ train_covariance.to_numpy() @ weights
        return risk_aversion[risk_preference] * variance - estimated_return


    train_result = minimize(
        training_objective,
        x0=equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    with st.expander("View training-period weights"):
        st.subheader("Portfolio Weights Estimated from Training Data")

        if train_result.success:
            training_weights = pd.DataFrame({
                "Asset": train_returns.columns,
                "Weight (%)": train_result.x * 100
            })

            st.dataframe(
                training_weights.sort_values(
                    "Weight (%)", ascending=False
                ).round(2),
                hide_index=True
            )
            st.write(f"Selected preference: {risk_preference}")
            st.write(f"Total weight: {train_result.x.sum():.2%}")
        else:
            st.error(f"Training optimization failed: {train_result.message}")

    if train_result.success:
        test_portfolio_returns = pd.DataFrame({
            "Optimized": test_returns @ train_result.x,
            "Equal Weight": test_returns @ equal_weights
        })

        test_growth = (1 + test_portfolio_returns).cumprod()

        st.subheader("Out-of-Sample Portfolio Performance")
        st.line_chart(test_growth * 100)

        st.caption(
            "Initial portfolio value before the test period: 100. "
            "Weights are estimated using training data only and held "
            "at constant target proportions through daily rebalancing. "
            "Transaction costs and taxes are excluded."
        )
    if train_result.success:
        running_peak = test_growth.cummax().clip(lower=1.0)
        drawdown = test_growth / running_peak - 1

        test_summary = pd.DataFrame({
            "Test Period Return (%)":
                (test_growth.iloc[-1] - 1) * 100,
            "Annualized Volatility (%)":
                test_portfolio_returns.std() * np.sqrt(252) * 100,
            "Maximum Drawdown (%)":
                -drawdown.min() * 100
        })

        st.subheader("Out-of-Sample Test Results")
        st.dataframe(test_summary.round(2))

        st.caption(
            "Return covers the test period only; volatility is annualized. "
            "Maximum drawdown is shown as a positive peak-to-trough loss. "
            "Results exclude transaction costs and taxes."
        )

