# Risk Dashboard

Risk Dashboard is a small, command-line program that gets a stock ticker from the S&P500 index and outputs a group of 
peer tickers that have similar risk metrics.

The program calculates the annual returns, volatility and beta of the given ticker and all other stocks in the same 
sector. Using a KMeans clustering model, the sector space is clustered by these risk parameters. A group of the 
closest stocks in the same cluster as the input stock is outputted with all metrics.

Risk Dashboard uses daily data over the past year, and refreshes cached data if older than last market close.

Risk Dashboard was developed just for fun to explore some basic architectural concepts and popular Python packages for 
AI. No practical use is expected.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/username/risk-comparer.git
cd risk-comparer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the program from the CLI:
```shell
python3 main.py
```

Input a stock ticker. The program expects a ticker that is listed 
on the S&P500. It is case-insensitive and accepts both '.' and '-' formatting for stock classes.

## Configuration

Settings are managed via `config.py` and environment variables (.env).
Edit `config.py` to change constants like file paths and clustering parameters.

## Comments

### Current APIs
Currently, stock data is acquired using the `yfinance` package, such as daily prices and company name. A list of all 
stocks and sectors based on the S&P500 index and is scraped from the Wikipedia page 
[here](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).


Since this requires some normalization to stay consistent over both datasets, and is dependent on the HTML layout of 
the Wiki page, it is highly recommended to integrate external APIs such as Polygon.io and Alpha Vantage 
(this was not done due to costs). 

Some API constants are written in `config.py` for potential future integration.

### Future expansion
- Currently, the number of clusters in the sector space is determined using the 
[elbow-method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)). This is not entirely accurate and can be 
refined by applying the [silhouette method](https://en.wikipedia.org/wiki/Silhouette_(clustering)) in addition.

- For a more interesting interface, it would be cool to make this program into a web app. 
An opportunity to try out a library like Streamlit?

- Allow the user to request the number of outputted peer tickers.