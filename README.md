# Cryptocurrency Derivatives Pricing and Delta-Neutral Volatility Trading

## A thesis submitted in fulfillment of the requirements for the degree of Master of Science

### Author: Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) 
### Supervisor: Dr. Peter H. Gruber
### Co-Supervisor: Dr. Paul Schneider


## Project description
In this thesis is explored and deepened the booming market for cryptocurrency derivatives. 
The [Deribit](https://www.deribit.com) Exchange is selected as market of reference.
Data are collected every 5-minutes starting from February 2021 until September 2021 for all the options listed on the Exchange for Bitcoin and Ethereum. 
The volatility surface, and the Black-Scholes’ Greeks are reverse engineered from the prices. 
The time series of the implied volatility is modelled as a linear regression depending on the option' Skew and on the option's time-to-maturity to study the at-the-money implied volatility dynamics, the at-the-money implied volatility term-structure and the implied volatility skew.
Finally, it is back-tested a pure volatility trading strategy that involves delta-neutral long and short straddles with the usage of options and of the underlying following the median-reverting properties of the implied volatility to statistically profit from the volatility movements.



Folder structure:
~~~~
thesis_bottacini_codes/
    Server/
    GetServerData/
    SpotAnalysis/
    TimeSeriesAnalysis/
    DeltaHedging/
    README.md
~~~~

## Table of contents
1. [Server](#server)
2. [Get Remote Server data](#get-remote-server-data)
3. [Spot Derivatives Analysis](#spot-derivatives-analysis)
4. [Time-Series Derivatives Analysis](#time-series-derivatives-analysis)
5. [Delta-Neutral Volatility Trading](#delta-neutral-volatility-trading)

## Server
The first step is to set up the remote Ubuntu server to download and store the financial data needed.
In these scripts are described both the codes and the intuition behind them.

* This is the [working directory](/Server)
* This is the [README.md](/Server/README.md)

## Get Remote Server data
The second step is to get option data from the remote server and to store them locally to perform the time analysis.
In these scripts are described both the codes and the intuition behind them.

* This is the [working directory](/GetServerData)
* This is the [README.md](/GetServerData/README.md)


## Spot Derivatives Analysis
The third step is to analyse current option data and visualize the results.
In these scripts are described both the codes and the intuition behind them.

* This is the [working directory](/SpotAnalysis)
* This is the [README.md](/SpotAnalysis/README.md)

##  Time-Series Derivatives Analysis
The fourth step is to analyze the crypto derivatives time-series and to prepare the data for the trading-strategy back-test.
In these scripts are described both the codes and the intuition behind them.

* This is the [working directory](/TimeSeriesAnalysis)
* This is the [README.md](/TimeSeriesAnalysis/README.md)

## Delta-Neutral Volatility Trading
The fifth and final step is to back-test a Delta-Neutral Volatility Trading strategy to statistically profit from the implied volatility deviations from the implied volatility long-run value.
In these scripts are described both the codes and the intuition behind them.

* This is the [working directory](/DeltaHedging)
* This is the [README.md](/DeltaHedging/README.md)
