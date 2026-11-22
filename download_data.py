import yfinance as yf

data = yf.download(
    [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "JPM", "JNJ", "XOM", "GLD",
        "TLT", "SHY"
    ],
    period="1y",
    auto_adjust=True
)

if data.empty:
    print("No data downloaded. The existing CSV file has been kept.")
else:
    close_prices = data["Close"]

    if close_prices.shape[1] != 12 or close_prices.isna().any().any():
        print("Missing assets or prices detected. The CSV file was not updated.")
        print(close_prices.count())
    else:
        close_prices.to_csv("data/prices.csv")
        print("Adjusted closing prices for all 12 assets have been saved.")
        print(close_prices.head())