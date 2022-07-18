"""
Coinbase Institute
MONTHLY INSIGHTS REPORT
Date: May 2022
Author: Cesare Fracassi
Twitter: @CesareFracassi
Email: cesare.fracassi@coinbase.com
"""

# Setup
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import statsmodels.api as sm
import glob
from datetime import datetime
!pip install pandas_datareader
import pandas_datareader as web
!pip install yfinance
import yfinance as yf

#%% UPLOAD SP500 DATA

# To Download stock market data (S&P 500)

# Getting the S&P 500 data from yahoo finance
sp500 = yf.Ticker("^GSPC").history(period="10y", auto_adjust=True)

# Computing the daily returns
sp500 = sp500.sort_values(by="Date", ascending=False)
sp500["ret_sp500"] = (sp500["Close"] - sp500["Close"].shift(-1)) / sp500["Close"].shift(
    -1
)

sp500 = sp500.dropna()

# Converting the time from eastern time to UTC
sp500 = sp500.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
sp500["Date_utc"] = pd.to_datetime(sp500.index, format="%Y-%m-%d %H:%M:%S")
sp500 = sp500.reset_index()

# Changing the date to 4pm ET, 9pm UTC
sp500["Date_utc"] = sp500["Date_utc"] + pd.DateOffset(hours=16)
sp500 = sp500.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)


# Getting the N100 data from yahoo finance
n100 = yf.Ticker("^NDX").history(period="10y", auto_adjust=True)

# Second, I compute daily returns
n100 = n100.sort_values(by="Date", ascending=False)
n100["ret_n100"] = (n100["Close"] - n100["Close"].shift(-1)) / n100["Close"].shift(-1)
n100 = n100.dropna()

# Converting the time from eastern time to UTC
n100 = n100.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
n100["Date_utc"] = pd.to_datetime(n100.index, format="%Y-%m-%d %H:%M:%S")
n100 = n100.reset_index()

# Changing the date to 4pm ET, 9pm UTC
n100["Date_utc"] = n100["Date_utc"] + pd.DateOffset(hours=16)
n100.rename(columns={"Close": "Close_n100", "Volume": "Volume_n100"}, inplace=True)

n100 = n100.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)


n100_list = pd.read_csv("nasdaq100.csv")
n100_tickers = n100_list["Symbol"].sort_values().tolist()

n100_prices = pd.DataFrame()
for item in n100_tickers:
    # print(item)
    temp = yf.Ticker(item).history(period="10y", auto_adjust=True)
    temp["id"] = item
    try:
        temp["marketcap"] = web.get_quote_yahoo(item)["marketCap"][0]
    except:
        print("Error with: ", item)
    n100_prices = n100_prices.append(temp)


n100_storage = n100_prices

# Computing daily returns
n100_prices.reset_index(inplace=True)
n100_prices = n100_prices.sort_values(by=["id", "Date"], ascending=[True, False])

n100_prices.loc[n100_prices["id"] == n100_prices["id"].shift(-1), "ok"] = True
n100_prices.loc[n100_prices["id"] != n100_prices["id"].shift(-1), "ok"] = False

n100_prices.loc[n100_prices["ok"] == True, "ret_stock"] = (
    n100_prices["Close"] - n100_prices["Close"].shift(-1)
) / n100_prices["Close"].shift(-1)

n100_prices = n100_prices.dropna()

# Third, I convert the time from eastern time to UTC

n100_prices.set_index("Date", drop=True, inplace=True)
n100_prices = n100_prices.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert(
    "UTC"
)

n100_prices["Date_utc"] = pd.to_datetime(n100_prices.index, format="%Y-%m-%d %H:%M:%S")

n100_prices = n100_prices.reset_index()

# Fourth, I change the date to 4pm ET, 9pm UTC
n100_prices["Date_utc"] = n100_prices["Date_utc"] + pd.DateOffset(hours=16)
n100_prices.rename(
    columns={"Close": "Close_stock", "Volume": "Volume_stock"}, inplace=True
)

n100_prices = n100_prices.drop(
    ["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1
)

# keep data until end of april
n100_prices = n100_prices.loc[n100_prices["Date_utc"] < "May 1, 2022"]

# Compute Volatility of each stock

n100_prices = n100_prices.sort_values(["id", "Date_utc"], ascending=[True, False])

n100_prices.set_index(["Date_utc"], inplace=True)
n100_prices.sort_index(inplace=True)
n100_roll_vol1y = (
    n100_prices.groupby("id")["ret_stock"].rolling(242).std().reset_index()
)

n100_roll_vol1y.rename(columns={"ret_stock": "vol_n100"}, inplace=True)

#%% UPLOAD GOLD DATA

# Getting the GOLD ETF data from yahoo finance
gold = yf.Ticker("GLD").history(period="10y", auto_adjust=True)

# Computing daily returns
gold = gold.sort_values(by="Date", ascending=False)
gold["ret_gold"] = (gold["Close"] - gold["Close"].shift(-1)) / gold["Close"].shift(-1)

gold = gold.dropna()

# Converting the time from eastern time to UTC
gold = gold.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
gold["Date_utc"] = pd.to_datetime(gold.index, format="%Y-%m-%d %H:%M:%S")
gold = gold.reset_index()

# Changing the date to 4pm ET, 9pm UTC
gold["Date_utc"] = gold["Date_utc"] + pd.DateOffset(hours=16)
gold.rename(columns={"Close": "Close_gold", "Volume": "Volume_gold"}, inplace=True)

gold = gold.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)

#%% UPLOAD SILVER DATA


# Getting the SILVER ETF data from yahoo finance
silver = yf.Ticker("SI=F").history(period="10y", auto_adjust=True)

# Computing the daily returns
silver = silver.sort_values(by="Date", ascending=False)
silver["ret_silver"] = (silver["Close"] - silver["Close"].shift(-1)) / silver[
    "Close"
].shift(-1)

silver = silver.dropna()

# Converting the time from eastern time to UTC
silver = silver.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")

silver["Date_utc"] = pd.to_datetime(silver.index, format="%Y-%m-%d %H:%M:%S")
silver = silver.reset_index()

# Changing the date to 4pm ET, 9pm UTC
silver["Date_utc"] = silver["Date_utc"] + pd.DateOffset(hours=16)
silver.rename(
    columns={"Close": "Close_silver", "Volume": "Volume_silver"}, inplace=True
)

silver = silver.drop(
    ["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1
)

#%% UPLOAD OIL DATA


# Getting the OIL ETF data from yahoo finance
oil = yf.Ticker("CL=F").history(period="10y", auto_adjust=True)

# Computing the daily returns
oil = oil.sort_values(by="Date", ascending=False)
oil["ret_oil"] = (oil["Close"] - oil["Close"].shift(-1)) / oil["Close"].shift(-1)

oil = oil.dropna()

# Converting the time from eastern time to UTC
oil = oil.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
oil["Date_utc"] = pd.to_datetime(oil.index, format="%Y-%m-%d %H:%M:%S")
oil = oil.reset_index()

# Changing the date to 4pm ET, 9pm UTC
oil["Date_utc"] = oil["Date_utc"] + pd.DateOffset(hours=16)
oil.rename(columns={"Close": "Close_oil", "Volume": "Volume_oil"}, inplace=True)

oil = oil.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)

# DROP 2020-04-20 20:00:00+00:00 because of negative price

oil = oil.loc[oil["Close_oil"] > 0]
#%% UPLOAD NATURAL GAS DATA


# Getting the GAS ETF data from yahoo finance
gas = yf.Ticker("NG=F").history(period="10y", auto_adjust=True)

# Computing the daily returns
gas = gas.sort_values(by="Date", ascending=False)
gas["ret_gas"] = (gas["Close"] - gas["Close"].shift(-1)) / gas["Close"].shift(-1)

gas = gas.dropna()

# Converting the time from eastern time to UTC
gas = gas.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
gas["Date_utc"] = pd.to_datetime(gas.index, format="%Y-%m-%d %H:%M:%S")
gas = gas.reset_index()

# Changing the date to 4pm ET, 9pm UTC
gas["Date_utc"] = gas["Date_utc"] + pd.DateOffset(hours=16)
gas.rename(columns={"Close": "Close_gas", "Volume": "Volume_gas"}, inplace=True)

gas = gas.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)

# DROP 2020-04-20 20:00:00+00:00 because of negative price

gas = gas.loc[gas["Close_gas"] > 0]

#%% UPLOAD TESLA DATA


# Getting the TESLA ETF data from yahoo finance
tesla = yf.Ticker("TSLA").history(period="10y", auto_adjust=True)

# Computing the daily returns
tesla = tesla.sort_values(by="Date", ascending=False)
tesla["ret_tesla"] = (tesla["Close"] - tesla["Close"].shift(-1)) / tesla["Close"].shift(
    -1
)

tesla = tesla.dropna()

# Converting the time from eastern time to UTC
tesla = tesla.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")

tesla["Date_utc"] = pd.to_datetime(tesla.index, format="%Y-%m-%d %H:%M:%S")
tesla = tesla.reset_index()

# Changing the date to 4pm ET, 9pm UTC
tesla["Date_utc"] = tesla["Date_utc"] + pd.DateOffset(hours=16)
tesla.rename(columns={"Close": "Close_tesla", "Volume": "Volume_tesla"}, inplace=True)

tesla = tesla.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)
#%% UPLOAD MODERNA DATA


# Getting the MODERNA ETF data from yahoo finance
mrna = yf.Ticker("MRNA").history(period="10y", auto_adjust=True)

# Computing the daily returns
mrna = mrna.sort_values(by="Date", ascending=False)
mrna["ret_mrna"] = (mrna["Close"] - mrna["Close"].shift(-1)) / mrna["Close"].shift(-1)

mrna = mrna.dropna()

# Converting the time from eastern time to UTC
mrna = mrna.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
mrna["Date_utc"] = pd.to_datetime(mrna.index, format="%Y-%m-%d %H:%M:%S")
mrna = mrna.reset_index()

# Changing the date to 4pm ET, 9pm UTC
mrna["Date_utc"] = mrna["Date_utc"] + pd.DateOffset(hours=16)
mrna.rename(columns={"Close": "Close_mrna", "Volume": "Volume_mrna"}, inplace=True)

mrna = mrna.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)
#%% UPLOAD DOCU DATA


# Getting the DOCU ETF data from yahoo finance
docu = yf.Ticker("DOCU").history(period="10y", auto_adjust=True)

# Computing the daily returns
docu = docu.sort_values(by="Date", ascending=False)
docu["ret_docu"] = (docu["Close"] - docu["Close"].shift(-1)) / docu["Close"].shift(-1)

docu = docu.dropna()

# Converting the time from eastern time to UTC
docu = docu.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
docu["Date_utc"] = pd.to_datetime(docu.index, format="%Y-%m-%d %H:%M:%S")
docu = docu.reset_index()

# Changing the date to 4pm ET, 9pm UTC
docu["Date_utc"] = docu["Date_utc"] + pd.DateOffset(hours=16)
docu.rename(columns={"Close": "Close_docu", "Volume": "Volume_docu"}, inplace=True)

docu = docu.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)

#%% UPLOAD LUCID DATA


# Getting the LUCID ETF data from yahoo finance
lcid = yf.Ticker("LCID").history(period="10y", auto_adjust=True)

# Computing the daily returns
lcid = lcid.sort_values(by="Date", ascending=False)
lcid["ret_lcid"] = (lcid["Close"] - lcid["Close"].shift(-1)) / lcid["Close"].shift(-1)

lcid = lcid.dropna()

# Converting the time from eastern time to UTC
lcid = lcid.tz_localize(tz="US/Eastern", ambiguous="infer").tz_convert("UTC")
lcid["Date_utc"] = pd.to_datetime(lcid.index, format="%Y-%m-%d %H:%M:%S")
lcid = lcid.reset_index()

# Changing the date to 4pm ET, 9pm UTC
lcid["Date_utc"] = lcid["Date_utc"] + pd.DateOffset(hours=16)
lcid.rename(columns={"Close": "Close_lcid", "Volume": "Volume_lcid"}, inplace=True)

lcid = lcid.drop(["Date", "Open", "High", "Low", "Dividends", "Stock Splits"], axis=1)

from os.path import isfile, join
import glob
from datetime import datetime

# Get list of all crypto prices csv files
mypath = r"Crypto Data"
filelist = glob.glob(os.path.join(mypath, "*.csv"))

crypto_prices = pd.DataFrame()

for f in filelist:
    temp = pd.read_csv(f)
    fname = f.split("_")[0][12:]
    temp["id"] = fname
    crypto_prices = crypto_prices.append(temp)

# Changing the start date of the period to the end to be consistent with the returns

crypto_prices["Date_utc"] = pd.to_datetime(
    crypto_prices["date"], format="%Y-%m-%d %H:%M:%S", utc=True
) + pd.DateOffset(hours=1)

# Matching the trading end times from the sp500 with the cyrpto prices.

crypto_sp_prices = pd.merge(crypto_prices, sp500, on="Date_utc", how="right")
crypto_sp_prices = pd.merge(crypto_sp_prices, n100, on="Date_utc", how="left")
crypto_sp_prices = pd.merge(crypto_sp_prices, gold, on="Date_utc", how="left")
crypto_sp_prices = pd.merge(crypto_sp_prices, oil, on="Date_utc", how="left")
crypto_sp_prices = pd.merge(crypto_sp_prices, silver, on="Date_utc", how="left")
crypto_sp_prices = pd.merge(crypto_sp_prices, tesla, on="Date_utc", how="left")
crypto_sp_prices = pd.merge(crypto_sp_prices, gas, on="Date_utc", how="left")
crypto_sp_prices = pd.merge(crypto_sp_prices, mrna, on="Date_utc", how="left")
crypto_sp_prices = pd.merge(crypto_sp_prices, docu, on="Date_utc", how="left")
crypto_sp_prices = pd.merge(crypto_sp_prices, lcid, on="Date_utc", how="left")

crypto_sp_prices = crypto_sp_prices[
    [
        "Date_utc",
        "id",
        "close",
        "Close",
        "volume",
        "Volume",
        "ret_sp500",
        "Close_n100",
        "Volume_n100",
        "ret_n100",
        "Close_gold",
        "Volume_gold",
        "ret_gold",
        "Close_silver",
        "Volume_silver",
        "ret_silver",
        "ret_oil",
        "Close_oil",
        "Volume_oil",
        "ret_lcid",
        "Close_lcid",
        "Volume_lcid",
        "ret_mrna",
        "Close_mrna",
        "Volume_mrna",
        "ret_docu",
        "Close_docu",
        "Volume_docu",
        "ret_gas",
        "Close_gas",
        "Volume_gas",
        "Close_tesla",
        "Volume_tesla",
        "ret_tesla",
    ]
]
sp500.info()
crypto_prices.info()
crypto_sp_prices.info()


crypto_sp_prices = crypto_sp_prices.sort_values(
    by=["id", "Date_utc"], ascending=[True, False]
)

lag_ok_crypto = crypto_sp_prices["id"] == crypto_sp_prices["id"].shift(-1)
crypto_sp_prices.loc[lag_ok_crypto, "ret_crypto"] = (
    crypto_sp_prices["close"] - crypto_sp_prices["close"].shift(-1)
) / crypto_sp_prices["close"].shift(-1)

crypto_sp_prices = crypto_sp_prices.rename(
    columns={
        "volume": "volume_crypto",
        "close": "close_crypto",
        "Volume": "volume_sp500",
        "Close": "close_sp500",
    }
).reset_index(drop=True)

crypto_sp_prices["volume_crypto_usd"] = (
    crypto_sp_prices["volume_crypto"] * crypto_sp_prices["close_crypto"]
)


# Bring in crypto market caps for non-stablecoin cryptos
crypto_marketcap = pd.read_csv("crypto_marketcap.csv")
crypto_marketcap["id"] = crypto_marketcap["Crypto"].str.lower()
stablecoin_list = "|".join(["usdt", "usdc", "ust", "busd", "dai", "usdp", "usdn"])

crypto_nonST_marketcap = crypto_marketcap.loc[
    ~crypto_marketcap["id"].str.contains(stablecoin_list)
]

# Compute Volatility of crypto

crypto_sp_prices = crypto_sp_prices.sort_values(
    ["id", "Date_utc"], ascending=[True, False]
)

crypto_sp_prices.set_index(["Date_utc"], inplace=True)
crypto_sp_prices.sort_index(inplace=True)
roll_vol1y = (
    crypto_sp_prices.groupby("id")["ret_crypto"].rolling(242).std().reset_index()
)

roll_vol1y.rename(columns={"ret_crypto": "vol"}, inplace=True)

# BTC Volatility 1y
# Figure 1

fig, ax = plt.subplots()
roll_vol1y.loc[roll_vol1y["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="BTC-USD 1yr Rolling Daily Volatility",
    ylabel="Volatility",
    xlabel="Date",
    label="BTC",
)

roll_vol1y["Date_int"] = roll_vol1y["Date_utc"].astype(int)
model = sm.formula.ols(
    formula="vol ~ Date_int", data=roll_vol1y.loc[roll_vol1y["id"] == "btc"]
)

res = model.fit()
roll_vol1y.loc[roll_vol1y["id"] == "btc"].assign(fit=res.fittedvalues).plot(
    x="Date_utc", y="fit", ax=ax
)

fig.get_figure().savefig("Figure 1.png")

roll_vol1y.loc[roll_vol1y["id"] == "btc"][["vol", "Date_utc"]].to_csv("volBTC.csv")

# Last year volatility for all cryptos

rank_vol1y = (
    roll_vol1y.loc[roll_vol1y["Date_utc"] == "2022-04-29 20:00:00+00:00"]
    .groupby("id")
    .tail(1)
    .dropna()
    .sort_values(["vol"], ascending=False)
)

# Bring in volume
volume_1y = (
    crypto_sp_prices.groupby("id")["volume_crypto_usd"]
    .rolling(242)
    .mean()
    .reset_index()
)

rank_vol1y = pd.merge(
    rank_vol1y,
    crypto_sp_prices[["volume_crypto_usd", "id"]].reset_index(),
    how="left",
    left_on=["id", "Date_utc"],
    right_on=["id", "Date_utc"],
)

rank_vol1y["id2"] = rank_vol1y["id"].str.rstrip("-USD")

rank_vol1y.sort_values(by="vol", ascending=True, inplace=True)
rank_vol1y["volume_log"] = np.log(rank_vol1y["volume_crypto_usd"])

rank_vol1y.iloc[3:][["volume_log", "vol"]].to_csv("volume_vol.csv")

# Figure 2

plt.clf()
rank_vol1y.sort_values("vol", inplace=True, ascending=False)
fig_rank_vol1y_bar = (
    rank_vol1y[-13:-3]
    .plot.barh(
        x="id2",
        y="vol",
        figsize=(24, 18),
        title="Daily Volatility (1y) ",
        ylabel=False,
        xlabel=False,
        legend=False,
    )
    .get_figure()
)


plt.title("Daily Volatility (1y)", fontsize=40)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
fig_rank_vol1y_bar.savefig("Figure 2.png")

rank_vol1y[-13:-3][["Date_utc", "vol", "id2"]].to_csv("rank_vol.csv")

# Figure 3

fig, ax = plt.subplots()
fig = sns.regplot(
    rank_vol1y.iloc[3:]["volume_log"],
    rank_vol1y.iloc[3:]["vol"],
    order=1,
    scatter_kws={"color": "blue", "s": 8},
    line_kws={"color": "red"},
)

ax.set(xlabel="Volume (log)", ylabel="Daily Volatility (1y)")
fig.get_figure().savefig("Figure 3.png")

# OIL Volatility VS BTC 1y

roll_vol1y_oil = (
    crypto_sp_prices[["ret_oil", "id"]]
    .dropna()
    .groupby("id")["ret_oil"]
    .rolling(242)
    .std()
    .reset_index()
)

roll_vol1y_oil.rename(columns={"ret_oil": "vol"}, inplace=True)

# Silver Volatility VS BTC 1y

roll_vol1y_silver = (
    crypto_sp_prices.groupby("id")["ret_silver"].rolling(242).std().reset_index()
)

roll_vol1y_silver.rename(columns={"ret_silver": "vol"}, inplace=True)


# GOLD Volatility VS BTC 1y

roll_vol1y_gold = (
    crypto_sp_prices.groupby("id")["ret_gold"].rolling(242).std().reset_index()
)

roll_vol1y_gold.rename(columns={"ret_gold": "vol"}, inplace=True)

# GAS Volatility VS BTC 1y

roll_vol1y_gas = (
    crypto_sp_prices[["ret_gas", "id"]]
    .dropna()
    .groupby("id")["ret_gas"]
    .rolling(242)
    .std()
    .reset_index()
)

roll_vol1y_gas.rename(columns={"ret_gas": "vol"}, inplace=True)

# COMPARE CRYPTO WITH COMMODITIES
# Figure 4

cryptobtceth = rank_vol1y.loc[
    (rank_vol1y["id"] == "btc") | (rank_vol1y["id"] == "eth")
][["id", "vol"]]

rank_vol1y_gas = (
    roll_vol1y_gas.loc[
        (roll_vol1y_gas["Date_utc"] == "2022-04-29 20:00:00+00:00")
        & (roll_vol1y_gas["id"] == "btc")
    ]
    .groupby("id")
    .tail(1)
    .dropna()
    .sort_values(["vol"], ascending=False)
)

rank_vol1y_gas["id"] = "Natural Gas"

rank_vol1y_oil = (
    roll_vol1y_oil.loc[
        (roll_vol1y_oil["Date_utc"] == "2022-04-29 20:00:00+00:00")
        & (roll_vol1y_oil["id"] == "btc")
    ]
    .groupby("id")
    .tail(1)
    .dropna()
    .sort_values(["vol"], ascending=False)
)

rank_vol1y_oil["id"] = "Oil"

rank_vol1y_gold = (
    roll_vol1y_gold.loc[
        (roll_vol1y_gold["Date_utc"] == "2022-04-29 20:00:00+00:00")
        & (roll_vol1y_gold["id"] == "btc")
    ]
    .groupby("id")
    .tail(1)
    .dropna()
    .sort_values(["vol"], ascending=False)
)

rank_vol1y_gold["id"] = "Gold"

rank_vol1y_silver = (
    roll_vol1y_silver.loc[
        (roll_vol1y_silver["Date_utc"] == "2022-04-29 20:00:00+00:00")
        & (roll_vol1y_silver["id"] == "btc")
    ]
    .groupby("id")
    .tail(1)
    .dropna()
    .sort_values(["vol"], ascending=False)
)

rank_vol1y_silver["id"] = "Silver"

cryptocomm = pd.concat(
    [
        cryptobtceth,
        rank_vol1y_gas[["id", "vol"]],
        rank_vol1y_oil[["id", "vol"]],
        rank_vol1y_gold[["id", "vol"]],
        rank_vol1y_silver[["id", "vol"]],
    ]
).sort_values("vol", ascending=True)

plt.clf()
fig_cryptocomm_bar = cryptocomm.plot.bar(
    x="id", y="vol", figsize=(24, 18), legend=False, ylabel=False, xlabel=False
).get_figure()

plt.title("Daily Volatility (1y)", fontsize=30)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
fig_cryptocomm_bar.savefig("Figure 4.png")
cryptocomm.to_csv("cryptocomm.csv")

# SP500 Volatility VS BTC 1y

roll_vol1y_sp500 = (
    crypto_sp_prices[["ret_sp500", "id"]]
    .dropna()
    .groupby("id")["ret_sp500"]
    .rolling(242)
    .std()
    .reset_index()
)

roll_vol1y_sp500.rename(columns={"ret_sp500": "vol"}, inplace=True)

# NASDAQ100 Volatility VS BTC 1y

roll_vol1y_n100 = (
    crypto_sp_prices[["ret_n100", "id"]]
    .dropna()
    .groupby("id")["ret_n100"]
    .rolling(242)
    .std()
    .reset_index()
)

roll_vol1y_n100.rename(columns={"ret_n100": "vol"}, inplace=True)

# COMPARE CRYPTO WITH INDICES
# Figure 5

cryptobtceth = rank_vol1y.loc[
    (rank_vol1y["id"] == "btc") | (rank_vol1y["id"] == "eth")
][["id", "vol"]]

rank_vol1y_sp500 = (
    roll_vol1y_sp500.loc[
        (roll_vol1y_sp500["Date_utc"] == "2022-04-29 20:00:00+00:00")
        & (roll_vol1y_sp500["id"] == "btc")
    ]
    .groupby("id")
    .tail(1)
    .dropna()
    .sort_values(["vol"], ascending=False)
)

rank_vol1y_sp500["id"] = "S&P 500"

rank_vol1y_n100 = (
    roll_vol1y_n100.loc[
        (roll_vol1y_n100["Date_utc"] == "2022-04-29 20:00:00+00:00")
        & (roll_vol1y_n100["id"] == "btc")
    ]
    .groupby("id")
    .tail(1)
    .dropna()
    .sort_values(["vol"], ascending=False)
)

rank_vol1y_n100["id"] = "Nasdaq 100"

cryptoindex = pd.concat(
    [cryptobtceth, rank_vol1y_sp500[["id", "vol"]], rank_vol1y_n100[["id", "vol"]]]
).sort_values("vol", ascending=True)

plt.clf()
fig_cryptoindex_bar = cryptoindex.plot.bar(
    x="id", y="vol", figsize=(24, 18), legend=False, ylabel=False, xlabel=False
).get_figure()

plt.title("Daily Volatility (1y)", fontsize=30)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
fig_cryptoindex_bar.savefig("Figure 5.png")
cryptoindex.to_csv("cryptoindex.csv")

# COMPARE CRYPTO WITH ALL SP500 CONSTITUENCIES

# MARKETCAP AS OF April 29
BTC_ETH_marketcap = pd.DataFrame(
    [["BTC", 734589762490], ["ETH", 339508249529]], columns=["TICKER", "Market Cap"]
)

# Bring in volatility 1y

BTC_ETH_vol = rank_vol1y[["vol", "id2"]].loc[
    (rank_vol1y["id2"] == "eth") | (rank_vol1y["id2"] == "btc")
]

BTC_ETH_vol["id2"] = BTC_ETH_vol["id2"].str.upper()
BTC_ETH_vol_marketcap = BTC_ETH_vol.merge(
    BTC_ETH_marketcap, left_on="id2", right_on="TICKER"
)

BTC_ETH_vol_marketcap.set_index("id2", inplace=True)

# Figure 6
# Chart volatility-marketcap
fig, ax = plt.subplots()
ax.clear()
# first, I create a small dataset with volatility and market cap
vol_marketcap_n100 = n100_roll_vol1y.loc[
    n100_roll_vol1y["Date_utc"].dt.date == pd.to_datetime("2022-04-29").date()
]

marketcap_n100 = (
    n100_prices[["id", "marketcap"]]
    .loc[n100_prices.index == "2022-04-29 20:00:00+00:00"]
    .set_index("id")
)

vol_marketcap_n100 = vol_marketcap_n100.merge(
    marketcap_n100.reset_index(), left_on="id", right_on="id"
)

vol_marketcap_n100 = vol_marketcap_n100.dropna(how="any").sort_values("marketcap")

vol_marketcap_n100.plot.scatter(
    x="marketcap",
    y="vol_n100",
    c="DarkBlue",
    ylabel="Volatility",
    xlabel="Market Cap",
    ax=ax,
)

vol_marketcap_n100.set_index("id", inplace=True)
for k, v in vol_marketcap_n100.tail(9).iterrows():
    print(k)
    print(v)
    ax.annotate(k, (v["marketcap"] - 1e11, v["vol_n100"] + 0.001), fontsize=8)


for k, v in vol_marketcap_n100.sort_values("vol_n100").tail(6).iterrows():
    print(k)
    print(v)
    ax.annotate(k, (v["marketcap"] - 1e11, v["vol_n100"] + 0.001), fontsize=8)

BTC_ETH_vol_marketcap.plot.scatter(
    x="Market Cap",
    y="vol",
    c="DarkRed",
    ylabel="Volatility",
    xlabel="Market Cap",
    ax=ax,
)

for k, v in BTC_ETH_vol_marketcap.tail(2).iterrows():
    print(k)
    print(v)
    ax.annotate(k, (v["Market Cap"] - 1e11, v["vol"] + 0.001), fontsize=8)

fig.get_figure().savefig("Figure 6.png")

pd.concat(
    [
        vol_marketcap_n100[["vol_n100", "marketcap"]],
        BTC_ETH_vol_marketcap.reset_index()[["vol", "Market Cap", "TICKER"]]
        .rename(columns={"vol": "vol_n100", "Market Cap": "marketcap", "TICKER": "id"})
        .set_index("id"),
    ]
).to_csv("nasdaq100_crypto.csv")

# Figure 7 (a)
# DOCU Volatility 1y

fig, ax = plt.subplots()
roll_vol1y.loc[roll_vol1y["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="BTC-USD Daily Volatility (1Y)",
    ylabel="Volatility",
    xlabel="Date",
    label="BTC",
)

roll_vol1y_docu = (
    crypto_sp_prices[["ret_docu", "id"]]
    .dropna()
    .groupby("id")["ret_docu"]
    .rolling(242)
    .std()
    .reset_index()
)

roll_vol1y_docu.rename(columns={"ret_docu": "vol"}, inplace=True)
roll_vol1y_docu.loc[roll_vol1y_docu["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="DOCU Daily Volatility (1y)",
    ylabel="volatility",
    xlabel="Date",
    label="docu",
)

fig.get_figure().savefig("Figure 7 (a).png")
roll_vol1y_docu.loc[roll_vol1y_docu["id"] == "btc"][["vol", "Date_utc"]].to_csv(
    "voldocu.csv"
)

# Figure 7 (b)
# MODERNA Volatility 1y

fig, ax = plt.subplots()
roll_vol1y.loc[roll_vol1y["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="BTC-USD Daily Volatility (1Y)",
    ylabel="Volatility",
    xlabel="Date",
    label="BTC",
)

roll_vol1y_mrna = (
    crypto_sp_prices[["ret_mrna", "id"]]
    .dropna()
    .groupby("id")["ret_mrna"]
    .rolling(242)
    .std()
    .reset_index()
)

roll_vol1y_mrna.rename(columns={"ret_mrna": "vol"}, inplace=True)
roll_vol1y_mrna.loc[roll_vol1y_mrna["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="MODERNA Daily Volatility (1y)",
    ylabel="volatility",
    xlabel="Date",
    label="mrna",
)

fig.get_figure().savefig("Figure 7 (b).png")
roll_vol1y_mrna.loc[roll_vol1y_mrna["id"] == "btc"][["vol", "Date_utc"]].to_csv(
    "volmrna.csv"
)

# Figure 7 (c)
# TESLA Volatility VS BTC 1y


fig, ax = plt.subplots()
roll_vol1y.loc[roll_vol1y["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="BTC-USD Daily Volatility (1Y)",
    ylabel="Volatility",
    xlabel="Date",
    label="BTC",
)

roll_vol1y_tesla = (
    crypto_sp_prices.groupby("id")["ret_tesla"].rolling(242).std().reset_index()
)

roll_vol1y_tesla.rename(columns={"ret_tesla": "vol"}, inplace=True)
roll_vol1y_tesla.loc[roll_vol1y_tesla["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="TESLA Daily Volatility (1y)",
    ylabel="volatility",
    xlabel="Date",
    label="TSLA",
)

fig.get_figure().savefig("Figure 7 (c).png.png")
roll_vol1y_tesla.loc[roll_vol1y_tesla["id"] == "btc"][["vol", "Date_utc"]].to_csv(
    "voltesla.csv"
)

# Figure 7 (d)
# LUCID Volatility 1y

fig, ax = plt.subplots()
roll_vol1y.loc[roll_vol1y["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="BTC-USD Daily Volatility (1Y)",
    ylabel="Volatility",
    xlabel="Date",
    label="BTC",
)

roll_vol1y_lcid = (
    crypto_sp_prices[["ret_lcid", "id"]]
    .dropna()
    .groupby("id")["ret_lcid"]
    .rolling(242)
    .std()
    .reset_index()
)

roll_vol1y_lcid.rename(columns={"ret_lcid": "vol"}, inplace=True)
roll_vol1y_lcid.loc[roll_vol1y_lcid["id"] == "btc"].plot.line(
    x="Date_utc",
    y="vol",
    ax=ax,
    title="LUCID Daily Volatility (1y)",
    ylabel="volatility",
    xlabel="Date",
    label="lcid",
)

fig.get_figure().savefig("Figure 7 (d).png")
roll_vol1y.loc[roll_vol1y["id"] == "btc"][["vol", "Date_utc"]].to_csv("volBTC.csv")

roll_vol1y_lcid.loc[roll_vol1y_lcid["id"] == "btc"][["vol", "Date_utc"]].to_csv(
    "volLucid.csv"
)

#%% Correlations crypto with SP500, N100
# Figure 8

# Compoute Monthly Correlation crypto-sp500
crypto_sp_prices = crypto_sp_prices.sort_values(
    ["id", "Date_utc"], ascending=[True, True]
)

corr = crypto_sp_prices.groupby("id")[["ret_sp500", "ret_crypto"]].rolling(242).corr()

corr = (
    pd.DataFrame(corr.groupby(level=[0, 1]).last()["ret_sp500"])
    .reset_index()
    .rename(columns={"ret_sp500": "corr"})
)

corr["Date_int"] = corr["Date_utc"].astype(int)

# Plot correlation Ethereum sp500
ax.clear()
ax = corr.loc[corr["id"] == "eth"].plot(
    x="Date_utc",
    y="corr",
    xlabel="Date",
    label="BTC",
    ylabel="Correlation",
    legend=False,
)

ax.axhline(y=0, xmin=-1, xmax=1, color="r", linestyle="--", lw=2)
ax.get_figure().savefig("Figure 8.png")

# Figure 9
# Last year correlation for all cryptos
rank_corr = (
    corr.loc[corr["Date_utc"] == "2022-04-29 20:00:00+00:00"]
    .groupby("id")
    .tail(1)
    .dropna()
    .sort_values(["corr"], ascending=False)
)

rank_corr = pd.merge(
    rank_corr,
    crypto_sp_prices[["volume_crypto_usd", "id"]].reset_index(),
    how="left",
    left_on=["id", "Date_utc"],
    right_on=["id", "Date_utc"],
)

rank_corr["id2"] = rank_corr["id"].str.rstrip("-USD")
fig, ax = plt.subplots()
rank_corr.sort_values(by="corr", ascending=True, inplace=True)
rank_corr["volume_log"] = np.log(rank_corr["volume_crypto_usd"])
fig = sns.regplot(
    rank_corr.iloc[3:]["volume_log"],
    rank_corr.iloc[3:]["corr"],
    order=1,
    scatter_kws={"color": "blue", "s": 8},
    line_kws={"color": "red"},
    ax=ax,
)

ax.set(xlabel="Volume (log)", ylabel="Correlation")

rank_corr.set_index("id", inplace=True)
for k, v in rank_corr.tail(2).iterrows():
    print(k)
    print(v)
    ax.annotate(k, (v["volume_log"] - 1e11, v["corr"] + 0.001))

for line in range(0, rank_corr.tail(2).shape[0]):
    plt.text(
        rank_corr.tail(2)["volume_log"][line] - 0.2,
        rank_corr.tail(2)["corr"][line],
        rank_corr.tail(2)["id2"][line],
        horizontalalignment="right",
        size="medium",
        color="black",
    )


fig.get_figure().savefig("Figure 9.png")

rank_corr.iloc[3:][["volume_log", "corr"]].to_csv("corr_volume.csv")

# Figure 10
# Correlation between nasdaq 100 constituency and the market.


# Chart correlation-marketcap

# first, I create a dataset with correlation and market cap
n100_prices.rename(columns={"ret_stock": "ret_n100"}, inplace=True)
n100_prices.sort_values(["id", "Date_utc"], ascending=[True, True], inplace=True)

temp_corr_n100_sp = n100_prices.merge(sp500, on="Date_utc", how="inner")
temp_corr_n100_sp = temp_corr_n100_sp.set_index("Date_utc")
corr_n100_sp = (
    temp_corr_n100_sp.groupby("id")[["ret_sp500", "ret_n100"]].rolling(242).corr()
)

corr_n100_sp = (
    pd.DataFrame(corr_n100_sp.groupby(level=[0, 1]).last()["ret_sp500"])
    .reset_index()
    .rename(columns={"ret_sp500": "corr"})
)

marketcap_n100 = (
    n100_prices[["id", "marketcap"]]
    .loc[n100_prices.index == "2022-04-29 20:00:00+00:00"]
    .set_index("id")
)

corr_marketcap_n100_sp = corr_n100_sp.loc[
    corr_n100_sp["Date_utc"] == "2022-04-29 20:00:00+00:00"
].merge(marketcap_n100.reset_index(), left_on="id", right_on="id")

corr_marketcap_n100_sp = corr_marketcap_n100_sp.dropna(how="any").sort_values(
    "marketcap"
)

fig, ax = plt.subplots()
ax.clear()
corr_marketcap_n100_sp.plot.scatter(
    x="marketcap",
    y="corr",
    c="DarkBlue",
    ylabel="Correlation with Nasdaq 100",
    xlabel="Market Cap",
    ax=ax,
)

corr_marketcap_n100_sp.describe()


# Bring in corr 1y

BTC_ETH_corr = rank_corr[["corr", "id2"]].loc[
    (rank_corr["id2"] == "eth") | (rank_corr["id2"] == "btc")
]

BTC_ETH_corr["id2"] = BTC_ETH_corr["id2"].str.upper()
BTC_ETH_corr_marketcap = BTC_ETH_corr.merge(
    BTC_ETH_marketcap, left_on="id2", right_on="TICKER"
)

BTC_ETH_corr_marketcap.set_index("id2", inplace=True)

BTC_ETH_corr_marketcap.plot.scatter(
    x="Market Cap",
    y="corr",
    c="DarkRed",
    ylabel="Correlation with Nasdaq 100",
    xlabel="Market Cap",
    ax=ax,
)

for k, v in BTC_ETH_corr_marketcap.tail(2).iterrows():
    print(k)
    print(v)
    ax.annotate(k, (v["Market Cap"] + 0.01, v["corr"] + 0.02), fontsize=8)

fig.get_figure().savefig("Figure 10.png")

pd.concat(
    [
        corr_marketcap_n100_sp[["marketcap", "corr", "id"]],
        BTC_ETH_corr_marketcap.tail(2).rename(
            columns={"TICKER": "id", "Market Cap": "marketcap"}
        ),
    ]
).to_csv("nasdaq100_corr_volume.csv")

# Figure 11
# Compute Beta

beta_crypto = rank_corr.merge(rank_vol1y, on="id")
beta_crypto["beta"] = (
    beta_crypto["corr"]
    * beta_crypto["vol"]
    / roll_vol1y_sp500.loc[roll_vol1y_sp500["Date_utc"] == "2022-04-29 20:00:00+00:00"][
        "vol"
    ].max()
)

beta_crypto["id"] = beta_crypto["id"].str.upper()
beta_crypto = beta_crypto.merge(
    BTC_ETH_marketcap, left_on="id", right_on="TICKER", how="outer"
)

beta_n100 = corr_marketcap_n100_sp.merge(vol_marketcap_n100, on="id")
beta_n100["beta"] = (
    beta_n100["corr"]
    * beta_n100["vol_n100"]
    / roll_vol1y_sp500.loc[roll_vol1y_sp500["Date_utc"] == "2022-04-29 20:00:00+00:00"][
        "vol"
    ].max()
)

fig, ax = plt.subplots()
ax.clear()
beta_n100.plot.scatter(
    x="marketcap_x", y="beta", c="DarkBlue", ylabel="Beta", xlabel="Market Cap", ax=ax
)

beta_n100 = beta_n100.sort_values("beta").reset_index()
beta_n100.describe()

beta_crypto.plot.scatter(
    x="Market Cap", y="beta", c="Red", ylabel="Beta", xlabel="Market Cap", ax=ax
)

beta_crypto.describe()


for k, v in beta_crypto.set_index("id").tail(2).iterrows():
    print(k)
    print(v)
    ax.annotate(k, (v["Market Cap"] - 0.1, v["beta"] - 0.1), fontsize=8)

fig.get_figure().savefig("Figure 11.png")
