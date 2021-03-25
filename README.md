# Cryptocurrency Derivatives Pricing

#### Last Update March 24, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####


## Project description

This project is to download and analyze cryptocurrency option data available on [Deribit](https://www.deribit.com) via a public API.
Data are collected on an Ubuntu remote server with the implementation of `Python3`, `Shell` and `SQLite` and are then analyzed locally with `Python3`.

Folder structure:
~~~~
thesis_bottacini_codes/
    Server/
    SpotImpliedVolatilitySurfaceAnalysis/
    GetServerData/
    README.md
~~~~

## Table of contents
1. [Server](#server)
2. [Spot Implied Volatility surface analysis](#spot-implied-volatility-surface-analysis)
3. [Get Remote Server data](#get-remote-server-data)


## Server
The first step is to set up the remote Ubuntu server to download and store the financial data needed.
In these scripts are described both the codes and the intuition behind them.

* This is the [working directory](../Server)
* This is the [README.md](../Server/README.md)

## Spot Implied Volatility surface analysis
The second step is to analyse current option data and visualize the implied volatility structure. 
In these scripts are described both the codes and the intuition behind them.

* This is the [working directory](../thesis_bottacini_codes/SpotImpliedVolatilitySurfaceAnalysis)
* This is the [README.md](../thesis_bottacini_codes/SpotImpliedVolatilitySurfaceAnalysis/README.md)

## Get Remote Server data
The third step is to get option data from the remote server and to store them locally to perform the time series analysis.
In these scripts are described both the codes and the intuition behind them.

* This is the [working directory](../thesis_bottacini_codes/GetServerData)
* This is the [README.md](../thesis_bottacini_codes/GetServerData/README.md)
