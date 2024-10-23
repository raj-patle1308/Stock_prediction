
---

# MachineLearningStocks in Python: A Starter Project and Guide

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/surelyourejoking/MachineLearningStocks/blob/master/LICENSE.txt)

*EDIT as of Feb 2021: MachineLearningStocks is no longer actively maintained*

MachineLearningStocks is designed to be an **intuitive** and **highly extensible** template project applying machine learning to make stock predictions. My hope is that this project will help you understand the overall workflow of using machine learning to predict stock movements and also appreciate some of its subtleties. After following this guide and playing around with the project, you should definitely **make your own improvements**. If you're struggling to think of what to do, I've included a long list of possibilities at the end of this README.

Concretely, we will be cleaning and preparing a dataset of historical stock prices and fundamentals using `pandas`, after which we will apply a `scikit-learn` classifier to discover the relationship between stock fundamentals (e.g., PE ratio, debt/equity, float, etc.) and the subsequent annual price change (compared with an index). We then conduct a simple backtest before generating predictions on current data.

While I would not live trade based on the predictions from this exact code, I believe you can use this project as a starting point for a profitable trading system – I have actually used code based on this project to live trade, achieving decent results (around 20% returns on backtest and 10-15% on live trading).

This project holds personal significance for me as it was my first proper Python project and my first encounter with ML. At the start, my code was rife with bad practices and inefficiency; I've since tried to amend most of this, but some minor issues may remain (feel free to raise an issue or fork and submit a PR). Both the project and I as a programmer have evolved significantly, but there is always room for improvement.

*As a disclaimer, this is a purely educational project. Be aware that backtested performance may often be deceptive – trade at your own risk!*

*MachineLearningStocks predicts which stocks will outperform. However, it does not suggest how best to combine them into a portfolio. I have released [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt), a portfolio optimization library that uses classical efficient frontier techniques (with modern improvements) to generate risk-efficient portfolios. Generating optimal allocations from the predicted outperformers might be a great way to improve risk-adjusted returns.*

This guide has been cross-posted on my academic blog, [reasonabledeviations.com](https://reasonabledeviations.com/).

## Contents

- [Overview](#overview)
  - [Edit History](#edit-history)
- [Quickstart](#quickstart)
- [Preliminaries](#preliminaries)
- [Historical Data](#historical-data)
  - [Historical Stock Fundamentals](#historical-stock-fundamentals)
  - [Historical Price Data](#historical-price-data)
- [Creating the Training Dataset](#creating-the-training-dataset)
  - [Preprocessing Historical Price Data](#preprocessing-historical-price-data)
  - [Features](#features)
    - [Valuation Measures](#valuation-measures)
    - [Financials](#financials)
    - [Trading Information](#trading-information)
  - [Parsing](#parsing)
- [Backtesting](#backtesting)
- [Current Fundamental Data](#current-fundamental-data)
- [Stock Prediction](#stock-prediction)
- [Unit Testing](#unit-testing)
- [Where to Go from Here](#where-to-go-from-here)
  - [Data Acquisition](#data-acquisition)
  - [Data Preprocessing](#data-preprocessing)
  - [Machine Learning](#machine-learning)
- [Contributing](#contributing)

## Overview

The overall workflow to use machine learning for stock prediction is as follows:

1. Acquire historical fundamental data – these are the *features* or *predictors*.
2. Acquire historical stock price data – this is the dependent variable, or label (what we are trying to predict).
3. Preprocess the data.
4. Use a machine learning model to learn from the data.
5. Backtest the performance of the machine learning model.
6. Acquire current fundamental data.
7. Generate predictions from current fundamental data.

This is a general overview, but in principle, this is all you need to build a fundamentals-based ML stock predictor.

### Edit History

#### EDIT as of 24/5/18
This project uses `pandas-datareader` to download historical price data from Yahoo Finance. However, this has become inconsistent due to changes in Yahoo's UI. As a temporary solution, I have uploaded `stock_prices.csv` and `sp500_index.csv` to keep the project functional.

#### EDIT as of October 2019
I have decided to upload additional CSV files: `keystats.csv` (the output of `parsing_keystats.py`) and `forward_sample.csv` (the output of `current_data.py`).

## Quickstart

If you want to jump right in, clone this project and download and unzip the [data file](https://pythonprogramming.net/data-acquisition-machine-learning/) into the same directory. Then, open a terminal and navigate to the project's directory, for example:

```bash
cd Users/User/Desktop/MachineLearningStocks
```

Run the following commands:

```bash
pip install -r requirements.txt
python download_historical_prices.py
python parsing_keystats.py
python backtesting.py
python current_data.py
pytest -v
python stock_prediction.py
```

Otherwise, follow the step-by-step guide below.

## Preliminaries

This project uses Python 3.6 and common data science libraries `pandas` and `scikit-learn`. If you're using Python 3.x less than 3.6, you may encounter syntax errors due to the use of f-strings. It's recommended to upgrade to 3.6 for better syntax. You can install all the requirements by running:

```bash
pip install -r requirements.txt
```

Make sure to clone this project and unzip it. This folder will become your working directory, so ensure your terminal instance is in this directory.

## Historical Data

Data acquisition and preprocessing are often the most challenging parts of machine learning projects, but it's a necessary process.

We need three datasets for this project:

1. Historical stock fundamentals
2. Historical stock prices
3. Historical S&P500 prices

The S&P500 index prices are needed as a benchmark: a 5% stock growth is insignificant if the S&P500 grew 10% in the same period. Therefore, all stock returns must be compared to those of the index.

### Historical Stock Fundamentals

Finding historical fundamental data is quite difficult (especially for free). While sites like [Quandl](https://www.quandl.com/) offer datasets, they often charge steep fees. You can parse this data for free from [Yahoo Finance](https://finance.yahoo.com/), specifically from the file called `intraQuarter.zip`, which contains HTML files that hold stock fundamentals for all stocks in the S&P500 from 2003 to 2013. However, we need to parse it into a usable CSV file.

### Historical Price Data

Initially, I used `pandas-datareader` to download stock price data, but due to changes in Yahoo Finance's UI, it stopped working. Now, I recommend using [Quandl](https://www.quandl.com/), which has free stock price data and a Python API. You can also manually download the S&P500 data from Yahoo Finance, place it in the project directory, and rename it `sp500_index.csv`.

To download historical price data, run the following command:

```bash
python download_historical_prices.py
```

## Creating the Training Dataset

The ultimate goal for the training data is to have a 'snapshot' of a stock's fundamentals at a particular time and the corresponding subsequent annual performance of that stock.

For example, if our snapshot consists of the fundamental data for AAPL on 28/1/2005, we also need to know the percentage price change of AAPL between 28/1/05 and 28/1/06. Our algorithm will learn how the fundamentals impact the annual change in stock price, particularly in relation to the S&P500 index.

### Preprocessing Historical Price Data

When `pandas-datareader` downloads stock price data, it does not include rows for weekends and public holidays. For instance, if our snapshot includes fundamental data for 28/1/05, the corresponding price change might not be available if it falls on a non-trading day.

To handle this, we 'fill forward' the missing data, assuming the stock price on a non-trading day is equal to the last available trading day's price.

### Features

Below is a list of some interesting variables available on Yahoo Finance:

#### Valuation Measures

- Market Cap
- Enterprise Value
- Trailing P/E
- Forward P/E
- PEG Ratio
- Price/Sales
- Price/Book
- Enterprise Value/Revenue
- Enterprise Value/EBITDA

#### Financials

- Profit Margin
- Operating Margin
- Return on Assets
- Return on Equity
- Revenue
- Revenue Per Share
- Quarterly Revenue Growth
- Gross Profit
- EBITDA
- Net Income Available to Common
- Diluted EPS


- Total Cash
- Total Debt
- Total Debt/Equity
- Current Ratio
- Book Value Per Share

#### Trading Information

- Beta
- 52-Week Change
- 52-Week High
- 52-Week Low
- 50-Day Moving Average
- 200-Day Moving Average
- Average Volume
- Shares Outstanding

### Parsing

To obtain the fundamental dataset, run the following command:

```bash
python parsing_keystats.py
```

This will create a CSV file called `keystats.csv`, which can be opened with any text editor or spreadsheet program.

## Backtesting

To backtest our algorithm, we must first define our entry and exit points. An entry point is when our model predicts a stock will outperform the S&P500 index over the next year (we can define outperforming as returning greater than 5% over the S&P500). An exit point could be when the model forecasts the stock will underperform over the next year.

To conduct the backtest, run:

```bash
python backtesting.py
```

## Current Fundamental Data

To obtain current fundamental data for stocks, run:

```bash
python current_data.py
```

This will output a CSV file called `forward_sample.csv`, which contains the current fundamentals of the S&P500 stocks.

## Stock Prediction

To generate predictions, run:

```bash
python stock_prediction.py
```

This will produce a prediction file based on the current stock fundamentals.

## Unit Testing

It's essential to test your code to ensure its reliability. You can run the tests in the `tests/` folder with:

```bash
pytest -v
```

This command will execute the tests and display detailed output of which tests passed and which failed.

## Where to Go from Here

After familiarizing yourself with this project, consider enhancing it or customizing it to your needs. Here are some suggestions:

### Data Acquisition

- **Expand Data Sources:** Look into other sources like Alpha Vantage, IEX Cloud, or Quandl for more robust datasets.
- **Real-time Data:** Implement functionality to fetch real-time stock data for live predictions.

### Data Preprocessing

- **Feature Engineering:** Add or modify features based on your understanding of stock movements.
- **Handle Missing Data:** Improve your strategy for dealing with missing values.

### Machine Learning

- **Experiment with Different Models:** Try various algorithms (like Random Forest, XGBoost, etc.) to see how they perform.
- **Hyperparameter Tuning:** Use techniques like Grid Search or Random Search for better model optimization.
- **Cross-Validation:** Implement cross-validation to ensure your model generalizes well.

### Backtesting

- **Improve Backtesting Logic:** Refine the backtesting mechanism to account for transaction costs, slippage, and better performance metrics.
- **Visualize Backtesting Results:** Add visualizations to understand backtesting results better.

### Stock Prediction

- **Portfolio Optimization:** Integrate the [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) library to create optimized portfolios.
- **Notifications:** Set up a notification system to alert you when stocks are predicted to outperform.

---

Feel free to modify any sections based on your preferences or project specifics!
