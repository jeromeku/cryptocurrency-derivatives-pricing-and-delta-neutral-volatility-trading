""" Self functions to download current data to analyze the current Implied Volatility Surface """

"""
Author: Matteo Bottacini, matteo.bottacini@usi.ch
Last update: March 19, 2021
"""

# import modules
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import shutil


# define implied volatility surface plot
def plot_iv_surf(x, y, z, coin, x2=None, y2=None, z2=None, label=''):
    # file path
    file_path = "./" + coin + "-IV-surface.png"

    # plot setup
    fig = plt.figure(3, figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_title(coin + ' Implied Volatility Surface \n' + dt.date.today().strftime("%B %d, %Y"))
    ax.set_zlabel('Implied Volatility')
    plt.xlabel('Moneyness')
    plt.ylabel('Days To Expiration')
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: '{:.0%}'.format(z)))

    # scatter observations
    if z2 is not None:
        ax.scatter3D(x2, y2, z2, c='r', s=100, label=label)
        # if coin == 'BTC':
        #   color = '#F2A900'
        # else:
        #   color = '#3c3c3d'

    # plot 3D surface
    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.8, cmap='RdYlGn_r')

    # add legend
    ax.legend()

    # save fig
    plt.savefig(file_path, dpi=80)
    plt.close()


# define ATM adjusted Implied Volatility Plot
def plot_atm_adj_iv(df_adj_iv_btc, df_adj_iv_eth):
    # file path
    file_path = "./atm_adj_iv.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.text(s='ATM Adjusted Implied Volatility', x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    fig.text(s=dt.date.today().strftime("%B %d, %Y"), x=0.5, y=0.90, fontsize=14, ha='center', va='center')
    fig.text(0.06, 0.5, 'Implied Volatility', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'Days to Expiration', ha='center', va='center')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.plot(df_adj_iv_btc.loc[:, 1], label='BTC', color='#F2A900')
    ax2.plot(df_adj_iv_eth.loc[:, 1], label='ETH', color='#3c3c3d')
    ax1.legend(loc='upper right', frameon=False)
    ax2.legend(loc='upper right', frameon=False)
    # save fig
    plt.savefig(file_path, dpi=80)
    plt.close()


# define Observation weight by moneyness and open interest plot
def moneyness_openinterest(df_btc, df_eth):
    # file path
    file_path = "./moneyness_openinterest.png"

    df_btc_ = df_btc.iloc[1:].sort_values(['t', 'strike']).query('t>0')
    df_eth_ = df_eth.iloc[1:].sort_values(['t', 'strike']).query('t>0')

    # weights
    np.seterr(divide='ignore')
    btc_weights = 1 / (1 + ((df_btc_['m'] ** 2))) + (
        (np.log(df_btc_['open_interest']).replace(-np.inf, 0) / np.log(df_btc_['open_interest']).replace(-np.inf,
                                                                                                         0).sum()))
    eth_weights = 1 / (1 + ((df_eth_['m'] ** 2))) + (
        (np.log(df_eth_['open_interest']).replace(-np.inf, 0) / np.log(df_eth_['open_interest']).replace(-np.inf,
                                                                                                         0).sum()))
    np.seterr(divide='warn')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.text(s='Observation Weight By Moneyness And Open Interest', x=0.5, y=0.95, fontsize=20, ha='center',
             va='center')
    fig.text(s=dt.date.today().strftime("%B %d, %Y"), x=0.5, y=0.90, fontsize=14, ha='center', va='center')
    fig.text(0.06, 0.5, 'Moneyness', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'Open Interest', ha='center', va='center')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.scatter(df_btc_['m'], btc_weights, label='BTC', color='#F2A900')
    ax2.scatter(df_eth_['m'], eth_weights, label='ETH', color='#3c3c3d')
    ax1.legend(loc='upper right', frameon=True)
    ax2.legend(loc='upper right', frameon=True)

    # save fig
    plt.savefig(file_path, dpi=160)
    plt.close()


# define fitted implied volatility by moneyness plot
def fitted_iv_moneyness(btc_iv_df_fit, eth_iv_df_fit):
    # file path
    file_path = "./fitted_iv_moneyness.png"

    # setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.text(s='Fitted Implied Volatility by Moneyness', x=0.5, y=0.95, fontsize=20, ha='center',
             va='center')
    fig.text(s=dt.date.today().strftime("%B %d, %Y"), x=0.5, y=0.90, fontsize=14, ha='center', va='center')
    fig.text(0.06, 0.5, 'Implied Volatility', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'Days to Expiration', ha='center', va='center')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.plot(btc_iv_df_fit.loc[:, [.95, 1, 1.05]])
    ax2.plot(eth_iv_df_fit.loc[:, [.95, 1, 1.05]])
    ax1.legend(['0.95', '1', '1.05'], loc='upper right', frameon=True)
    ax2.legend(['0.95', '1', '1.05'], loc='upper right', frameon=True)

    # save fig
    plt.savefig(file_path, dpi=160)
    plt.close()


# Get a list of all active options from the Deribit API.
def get_all_active_options(coin):
    """

    :param coin: 'BTC' or 'ETH'
    :return: list of all active options from the Deribit API
    """
    import urllib.request, json
    url = "https://test.deribit.com/api/v2/public/get_instruments?currency=" + coin + "&kind=option&expired=false"
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    data = pd.DataFrame(data['result']).set_index('instrument_name')
    data['creation_date'] = pd.to_datetime(data['creation_timestamp'], unit='ms')
    data['expiration_date'] = pd.to_datetime(data['expiration_timestamp'], unit='ms')
    print(f'{data.shape[0]} active options.')
    return data


# Filter options based on data available from 'get_instruments'
def filter_options(price, active_options):
    """

    :param price: current coin price
    :param active_options: list of active options with get_all_active_options()
    :return: list of active options after filter
    """

    # Get Put/Call information
    pc = active_options.index.str.strip().str[-1]

    # Set "moneyness"
    active_options['m'] = np.log(active_options['strike'] / price)
    active_options.loc[pc == 'P', 'm'] = -active_options['m']
    # Set days until expiration
    active_options['t'] = (active_options['expiration_date'] - pd.Timestamp.today()).dt.days

    # Only include options that are less than 30% from the current price and have less than 91 days until expiration
    active_options = active_options.query('m>0 & m<.3 & t<91')

    print(f'{active_options.shape[0]} active options after filter.')
    return active_options


# Get Tick data for a given instrument from the Deribit API
def get_tick_data(instrument_name):
    import urllib.request, json
    url = f"https://test.deribit.com/api/v2/public/ticker?instrument_name={instrument_name}"
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    data = pd.json_normalize(data['result'])
    data.index = [instrument_name]
    return data


# Loop through all filtered options to get the current 'ticker' datas
def get_all_option_data(coin):
    option_data = get_tick_data(coin + '-PERPETUAL')
    options = filter_options(option_data['last_price'][0], get_all_active_options(coin=coin))
    for o in options.index:
        option_data = option_data.append(get_tick_data(o))
    return option_data


# save WLS model summary
def save_wls_model_summary(model, coin):
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./WLS-' + coin + '.png')
    plt.close()


# move files function
def move_files(cwd, dwd, endswith):
    """

    :param cwd: current workiong directory
    :param dwd: destination working directory
    :param endswith: file ends with
    :return: move files
    """
    source_path = os.path.abspath(os.getcwd())
    source_files = os.listdir(source_path)
    destination_path = source_path.replace("/SpotImpliedVolatilitySurfaceAnalysis/" + cwd,
                                           "/SpotImpliedVolatilitySurfaceAnalysis/" + dwd)
    for file in source_files:
        if file.endswith(endswith):
            shutil.move(os.path.join(source_path, file), os.path.join(destination_path, file))
