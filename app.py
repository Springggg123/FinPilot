# Import the tools used for the dashboard and calculations.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import minimize

# Use a wide layout so charts and tables have more space.
st.set_page_config(page_title="FinPilot Robo-Advisor", layout="wide")

st.title("FinPilot Robo-Advisor")
st.write("Explore historical asset performance, risk, and clustering.")
st.sidebar.header("Investment Settings")

# Get the investment amount from the user.
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
# Let the user choose a risk preference.
risk_preference = st.sidebar.selectbox(
    "Risk Preference",
    ["Conservative", "Balanced", "Growth"]
)

st.sidebar.caption(
    "Conservative prioritizes lower volatility. "
    "Balanced and Growth place increasing emphasis on estimated returns."
)
# Match each display name to its asset code.
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

# Read saved prices and use the first column as dates.
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

# Create four tabs. All tab code still runs when the app reruns.
market_tab, ai_tab, portfolio_tab, test_tab = st.tabs([
    "Market Overview", "AI Clustering", "Portfolio Builder", "Performance Test"
])

# 1. Show historical prices, returns, and volatility.
with market_tab:
    asset = st.selectbox("Select an asset", list(asset_codes.keys()))
    # Get the code for the selected asset.
    ticker = asset_codes[asset]

    st.write("Selected asset:", asset)

    # Keep detailed data inside a section the user can open.
    with st.expander("View price data"):
        st.subheader(f"{ticker} Historical Adjusted Closing Prices")
        st.dataframe(prices[[ticker]].head())

    st.subheader(f"{ticker} Price Trend")
    st.line_chart(prices[ticker])
    st.caption("Y-axis: Adjusted closing price (USD). X-axis: Date.")

    # Compare the last price with the first price to find total return.
    start_price = prices[ticker].iloc[0]
    end_price = prices[ticker].iloc[-1]
    total_return = (end_price / start_price - 1) * 100

    st.metric("Cumulative Return", f"{total_return:.2f}%")

    # Calculate daily percentage changes as decimals and remove empty values.
    daily_returns = prices[ticker].pct_change(fill_method=None).dropna()

    with st.expander("View daily returns"):
        st.subheader("Daily Returns (First 5 Observations)")
        st.dataframe(
            (daily_returns.head() * 100).to_frame("Daily Return (%)")
        )

    # Convert daily volatility to an annual estimate using 252 trading days.
    annual_volatility = daily_returns.std() * (252 ** 0.5)
    st.metric("Annualized Volatility", f"{annual_volatility:.2%}")

    st.subheader("Asset Performance Comparison (Base = 100)")

    # Keep only dates with prices for every asset.
    comparison_prices = prices.dropna()
    # Set every starting price to 100 for a fair performance comparison.
    normalized_prices = (
        comparison_prices / comparison_prices.iloc[0] * 100
    )

    st.line_chart(normalized_prices)

    # Calculate daily returns for all assets.
    all_returns = comparison_prices.pct_change(fill_method=None).dropna()

    # Build a table of historical return and volatility for each asset.
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

# 2. Group assets by their historical return and volatility.
with ai_tab:
    # Use both columns in the summary as clustering features.
    features = summary.copy()

    # Put the two features on comparable scales before clustering.
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

    # Use three groups, a fixed random seed, and ten starting attempts.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    # Fit K-Means and get a group number for each asset.
    cluster_labels = kmeans.fit_predict(scaled_features)

    cluster_result = summary.copy()
    cluster_result["Cluster ID"] = cluster_labels


    # Calculate the average return and volatility in each group.
    cluster_summary = cluster_result.groupby("Cluster ID")[
        ["Cumulative Return (%)", "Annualized Volatility (%)"]
    ].mean()

    cluster_summary.columns = [
        "Mean Cumulative Return (%)",
        "Mean Annualized Volatility (%)"
    ]

    # Count how many assets belong to each group.
    cluster_summary["Number of Assets"] = (
        cluster_result.groupby("Cluster ID").size()
    )

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary.round(2))

    # Rank groups by average volatility; group numbers have no risk order.
    cluster_order = cluster_summary[
        "Mean Annualized Volatility (%)"
    ].sort_values().index

    # Give each group a relative volatility label.
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

    # Turn the asset index into a column for plotting.
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

    # Measure cluster separation. This is not prediction accuracy.
    silhouette = silhouette_score(scaled_features, cluster_labels)

    st.subheader("Clustering Evaluation")
    st.metric("Silhouette Score", f"{silhouette:.3f}")
    st.caption(
        "Scores range from -1 to 1. Higher values indicate more compact "
        "and better-separated clusters in the selected feature space. "
        "This score does not measure investment performance."
    )

# 3. Build portfolios using the full historical dataset.
with portfolio_tab:
    # Estimate annual covariance to measure how asset returns move together.
    annual_covariance = all_returns.cov() * 252
    # Estimate annual arithmetic returns from average daily returns.
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
    # Give every asset the same weight for the benchmark.
    equal_weights = np.ones(asset_count) / asset_count

    # Combine weights and covariance to calculate portfolio variance.
    portfolio_variance = (
        equal_weights @ annual_covariance.to_numpy() @ equal_weights
    )
    # Take the square root of variance to get volatility.
    portfolio_volatility = np.sqrt(portfolio_variance)

    with st.expander("View equal-weight benchmark"):
        st.subheader("Equal-Weight Portfolio Benchmark")
        st.write(f"Number of assets: {asset_count}")
        st.write(f"Weight per asset: {1 / asset_count:.2%}")

        st.metric(
            "Estimated Annualized Portfolio Volatility",
            f"{portfolio_volatility:.2%}"
        )

    # Return annual portfolio volatility for a given set of weights.
    def calculate_portfolio_volatility(weights):
        covariance = annual_covariance.to_numpy()
        variance = weights @ covariance @ weights
        return np.sqrt(variance)


    # Check the risk function with the equal-weight portfolio.
    check_volatility = calculate_portfolio_volatility(equal_weights)



    # Demo settings: a larger coefficient puts more weight on reducing risk.
    risk_aversion = {
        "Conservative": 20.0,
        "Balanced": 5.0,
        "Growth": 1.0
    }


    # Minimize the risk penalty minus estimated return.
    def portfolio_objective(weights):
        estimated_return = weights @ annual_returns.to_numpy()
        variance = weights @ annual_covariance.to_numpy() @ weights
        penalty = risk_aversion[risk_preference] * variance
        return penalty - estimated_return


    # Require all portfolio weights to add up to 1 (100%).
    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    # Do not allow short selling; limit each asset to 30%.
    bounds = [(0, 0.30)] * asset_count

    # Search for weights that best match the selected preference.
    result = minimize(
        portfolio_objective,
        x0=equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    st.subheader(f"{risk_preference} Portfolio (30% Asset Cap)")

    # Show the allocation only if the optimizer succeeds.
    if result.success:
        # Calculate volatility separately; the objective value is not volatility.
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

        # Convert weights to percentages and investment amounts in dollars.
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

        # Compare estimates using the same full dataset, not a separate test period.
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
    # Minimizing negative return is the same as maximizing return.
    def negative_portfolio_return(weights):
        return -(weights @ annual_returns.to_numpy())


    # Find the highest estimated return allowed by the constraints.
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
        # Find the lowest-volatility portfolio for the frontier starting point.
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
        # Choose 20 return targets between the two frontier endpoints.
        target_returns = np.linspace(
            frontier_start_return, max_return, 20
        )
        frontier_rows = []

        # Find the lowest volatility for each target return.
        for target in target_returns:
            # Keep weights at 100% and require the current target return.
            frontier_constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {
                    "type": "eq",
                    # Save this loop target as the function default.
                    "fun": lambda w, target_value=target: (
                                                                  w @ annual_returns.to_numpy()
                                                          ) - target_value
                }
            ]

            # Optimize risk while meeting the target return and asset limits.
            frontier_result = minimize(
                calculate_portfolio_volatility,
                x0=equal_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=frontier_constraints,
                options={"maxiter": 1000, "ftol": 1e-9}
            )

            # Keep only successful frontier calculations.
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

# 4. Evaluate weights on a later period that was not used to fit them.
with test_tab:
    # Split returns in time order: first 70% for training, last 30% for testing.
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
    # Estimate returns and covariance using training data only.
    train_annual_returns = train_returns.mean() * 252
    train_covariance = train_returns.cov() * 252


    # Use the same preference rule, but only training-period estimates.
    def training_objective(weights):
        estimated_return = weights @ train_annual_returns.to_numpy()
        variance = weights @ train_covariance.to_numpy() @ weights
        return risk_aversion[risk_preference] * variance - estimated_return


    # Fit new weights without using test-period returns.
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
            # List the weights fitted on the training period.
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
        # Apply weights to test returns, assuming daily rebalancing to fixed weights.
        test_portfolio_returns = pd.DataFrame({
            "Optimized": test_returns @ train_result.x,
            "Equal Weight": test_returns @ equal_weights
        })

        # Compound daily returns to calculate growth from an initial value of 1.
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
        # Track the highest value so far, including the initial value of 1.
        running_peak = test_growth.cummax().clip(lower=1.0)
        # Measure the fall from each previous peak.
        drawdown = test_growth / running_peak - 1

        # Compare test-period return, annualized volatility, and maximum loss from a peak.
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

