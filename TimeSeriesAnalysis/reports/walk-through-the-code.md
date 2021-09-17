# Walk through the code

#### Last Update September 16, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

## Project description
In this report I explain how to analyze the Time-Series of Cryptocurrency Derivatives Market.

The function described are in [`../TimeSeriesAnalysis/src/utils.py`](../src/utils.py).

## Table of contents

1. [Create the Environment: directories and sub-directories](#create-the-environment-directories-and-sub-directories)
2. [Load data and pre-processing](#load-data-and-pre-processing)
3. [Underlying analytics](#underlying-analytics)   
4. [Implied Volatility historical distribution](#implied-volatility-historical-distribution) 
5. [Implied Volatility Modelling](#implied-volatility-modelling)
6. [Model calibration](#model-calibration)


## Create the Environment: directories and sub-directories
The first step is to create all the directories and sub-directories needed to perform the analysis. We split them into: `level 1`, `level 2` and `level 3`.
With `level 1` being the main sub-directories, `level 2` a sub-sub-directory, etc.

* level 1: `../reports/BTC`, `../reports/ETH`.
* level 2: `../reports/coin/?_min_calibration`.
* level 3:  `../?_min_calibration/data`, `../?_min_calibration/images`, `../?_min_calibration/tables`

In [`../src/utils.py`](../src/utils.py) the function `create_env()` performs these tasks:

```python
# Create Environment
def create_env(local_folder, min_time_interval, max_time_interval, step):

    # import modules
    import os

    # source path
    source_path = os.path.abspath(os.getcwd())

    # level_0: ../reports
    destination_path = source_path.replace(local_folder, 'reports')
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    # level_1: folders for each coin ../reports/coin
    coins = ['BTC', 'ETH']

    # level_2: folders for model calibration ../reports/coin/?_min_calibration
    intervals = list(range(min_time_interval, max_time_interval, step))
    folders = [str(interval) + '_min_calibration' for interval in intervals]

    # level_3: sub-folders ../?_min_calibration/data ../?_min_calibration/images ../?_min_calibration/tables
    sub_folders = ['data', 'images', 'tables']

    # create dirs for each coin
    for coin in coins:

        # level_1: ../reports/coin
        level_1 = destination_path + '/' + coin
        if not os.path.exists(level_1):
            os.mkdir(level_1)

        # level_2: ../reports/coin/?_min_calibration
        for folder in folders:
            level_2 = level_1 + '/' + folder
            if not os.path.exists(level_2):
                os.mkdir(level_2)

            # level_3: ../reports/coin/?_min_calibration/sub_folder
            for sub_folder in sub_folders:
                level_3 = level_2 + '/' + sub_folder
                if not os.path.exists(level_3):
                    os.mkdir(level_3)

    return print('Environment created!')
```

## Load data and pre-processing
The second step is to load the data from [`/GetServerData/data`](/GetServerData/data) to the working environment and to process the data.
The function is `load_data()` and it does the following:

1. read the feather data-set.
2. pull `strike`, `datetime`, `time-to-maturity`

```python
# load coin data from GetServerData/data/
def load_data(coin, cwd):

    """

    :param cwd: current working directory
    :param coin: 'btc' or 'eth'
    :return: pandas.DataFrame with all data collected
    """

    # import modules
    import pyarrow.feather as feather
    import pandas as pd
    import datetime as dt
    import os

    # file path
    source_path = os.path.abspath(os.getcwd())
    source_path = source_path.replace(cwd, 'GetServerData/data/' + coin + '_option_data.ftr')

    # load data from GetServerData/data/...
    df = feather.read_feather(source_path)
    print(coin + ' data imported')

    # map index
    index = df['instrument_name'].map(lambda x: x.split('-'))

    # pull strike
    strike = [int(element[2]) for element in index]
    df['strike'] = strike

    # pull maturity
    maturity = [element[1] for element in index]
    maturity = pd.to_datetime(maturity) + pd.DateOffset(hours=10)

    # pull date and time -- 5min round
    df['timestamp'] = pd.DatetimeIndex(
        df['timestamp'].apply(lambda d: dt.datetime.fromtimestamp(int(d) / 1000).strftime('%Y-%m-%d %H:%M:%S')))
    df['timestamp'] = df['timestamp'].round('5min')

    # pull time to maturity
    t = maturity - df['timestamp']
    t = t / pd.to_timedelta(1, unit='D')
    df['t'] = t / 365

    print('additional metrics added')

    return df
```

## Underlying analytics
The third step is to analyze the underlying dynamics. The function is `underlying_analytics()` and it does the following:

1. Extract the time-series of the index prices from the data-sets.
2. Plot the time-series of the underlying prices and store the results.
3. Evaluate 5-min and daily-log returns.
4. Plot a histogram of returns and store the results.
5. Evaluate cumulative returns, realized volatility and 30 days rolling correlation.
6. Store and plot the results of the cumulative returns, realized volatility and 30 days rolling correlation.

```python
# Index price analytics: price, returns, ...
def underlying_analytics(local_folder, btc_data, eth_data):

    # import modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # subset df
    btc_df = btc_data[btc_data['instrument_name'].str.contains("-C")]
    eth_df = eth_data[eth_data['instrument_name'].str.contains('-C')]

    # index price
    coins = ['BTC', 'ETH']
    btc_index_price = btc_df[['timestamp', 'index_price']].sort_values('timestamp')
    eth_index_price = eth_df[['timestamp', 'index_price']].sort_values('timestamp')

    btc_index_price = btc_index_price.groupby('timestamp').mean()
    eth_index_price = eth_index_price.groupby('timestamp').mean()
    index_price = pd.concat([btc_index_price, eth_index_price], axis=1)
    index_price.columns = coins

    folder_path = os.path.abspath(os.getcwd())
    folder_path = folder_path.replace(local_folder, 'reports/images/')

    # plot underlying
    plt.rcParams['font.family'] = 'serif'
    for coin in coins:

        # file path
        file_path = folder_path + coin + 'usd_price.pdf'
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        fig.text(s=coin + ' -USD Price \n 5 minutes interval',
                 x=0.5, y=0.95, fontsize=20, ha='center', va='center')
        if coin == 'ETH':
            ax.plot(index_price[coin], color='C1')
        else:
            ax.plot(index_price[coin], color='C0')
        fig.text(0.06, 0.5, coin + ' -USD', ha='center', va='center', rotation='vertical')
        ax.margins(x=0)
        ax.margins(y=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(file_path, dpi=160)  # save fig
        plt.close()

    # log returns
    log_ret = np.log(index_price / index_price.shift(1))
    daily_log_ret = log_ret.resample('D').sum()

    daily_log_ret.skew()
    daily_log_ret.kurt()

    # histogram
    def histogram_plot(log_ret, interval):
        file_path_plot = folder_path + interval + '_returns.pdf'
        plt.rcParams['font.family'] = 'serif'  # set font family: serif
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        fig.text(s='Returns distribution' +
                   ' \n ' + interval + ' interval',
                 x=0.5, y=0.95, fontsize=20, ha='center', va='center')
        for coin in coins:
            plt.hist(log_ret[coin], alpha=0.5, label=coin)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
        ax.legend(coins, bbox_to_anchor=(.5, 0.03), loc="lower center",
                  bbox_transform=fig.transFigure, ncol=len(coins), frameon=False)
        plt.savefig(file_path_plot, dpi=160)  # save fig
        plt.close()

    histogram_plot(log_ret=log_ret, interval='5 minutes')
    histogram_plot(log_ret=daily_log_ret, interval='1 day')

    cum_ret = (1 + log_ret).cumprod()
    realized_vol = np.sqrt((log_ret ** 2).resample('D').sum())
    rolling_correlation = log_ret[coins[0]].rolling(288 * 30).corr(log_ret[coins[1]])

    file_path_plot = folder_path + '_cum_ret.pdf'
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    fig.text(s='BTC-USD vs ETH-USD',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    for i in [0, 1, 2]:
        ax[i].margins(x=0)
        ax[i].margins(y=0)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].plot(cum_ret)
    ax[0].hlines(xmin=cum_ret.index[0], xmax=cum_ret.index[-1], y=1, linestyle='--', color='black', lw=.1)
    ax[1].plot(rolling_correlation, color='black')
    ax[2].plot(realized_vol)
    ax[2].spines['bottom'].set_visible(True)
    fig.legend(coins, bbox_to_anchor=(.5, 0.03), loc="lower center",
               bbox_transform=fig.transFigure, ncol=len(coins), frameon=False)
    fig.text(0.06, 0.22, 'Daily Realized Volatility', ha='center', va='center', rotation='vertical')
    fig.text(0.06, 0.48, '30 days Rolling Correlation', ha='center', va='center', rotation='vertical')
    fig.text(0.06, 0.77, 'Cumulative returns', ha='center', va='center', rotation='vertical')
    plt.savefig(file_path_plot, dpi=160)  # save fig
    plt.close()

    # summary statistics
    file_path = folder_path + 'summary.csv'
    summary_1 = log_ret.describe()
    summary_2 = daily_log_ret.describe()
    summary = pd.concat([summary_1, summary_2], axis=1)
    summary.columns = ['BTC_5min', 'ETH_5min', 'BTC_1day', 'ETH_1day']
    summary.to_csv(file_path)

    return index_price
```

## Implied Volatility historical distribution
The forth step is to analyze the historical distribution of the market implied volatility. The function `iv_distribution()` and it does the following:

1. Subset the market implied volatility for all the options in the data-set.
2. Plot a histogram of the implied volatility.
3. Evaluate and store the descriptive statistics of the distribution.

```python
# implied volatility distribution
def iv_distribution(coin_df, coin, local_folder):

    # import modules
    import os
    import matplotlib.pyplot as plt

    # source_path
    source_path = os.path.abspath(os.getcwd())
    file_path = source_path.replace(local_folder, 'reports/' + coin + '_iv_distribution.pdf')

    # data
    iv = coin_df['mark_iv']

    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=coin + 'USD Implied Volatility distribution',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    if coin == 'ETH':
        plt.hist(iv, color='C1')
    else:
        plt.hist(iv, color='C0')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.savefig(file_path, dpi=160)  # save fig
    plt.close()

    # summary statistics
    iv_summary = iv.describe()
    iv_summary.T['skew'] = iv.skew()
    iv_summary.T['kurt'] = iv.kurt()

    file_path = file_path.replace('_iv_distribution.pdf', '_iv_summary.csv')
    iv_summary.to_csv(file_path)

    return iv
```

## Implied Volatility modelling
The fifth step is to calibrate a model to estimate the Implied Volatility for every option at every point in time given its skew and its time-to-maturity.
The function is `iv_linear_interpolation()` and it does the following:

1. Subset the Call Options from the original dataset solely.
2. Manage the size of the new `pandas.DataFrame` keeping only the variable of interest.
3. Evaluate the skew of each option.
4. Sort the dataframe by `timestamp`.
5. Set x1, x2, x3 and y as skew, skew squared, time-to-maturity and market implied volatility respectively.
6. States the different maturities to be trained.
7. Train the model every 5-minutes with the given data depending on the window selected (e.g. 5min, 10min, 15min, ..., 1440min)
8. Store the parameters and coefficient result at every iteration.
9. Store the estimated implied volatility result.
10. Evaluate the ATM implied volatility, the term-structure and the skew.
11. Plot all the result.
12. Store the results and the summary statistics.

Note: running this function is quite long, especially training the model.
```python
# iVol linear interpolation
def iv_linear_interpolation(coin_df, step, atm_skew, itm_skew, otm_skew, local_folder, coin):
    """

    :param coin:
    :param local_folder:
    :param coin_df: pandas.DataFrame resulting from load_data()
    :param step: time interval to calibrate the model, multiples of 5 (e.g. each 5 mins, each 10 mins, ...)
    :param atm_skew: skew of atm options to find IV, skew = S/K (i.e. 1)
    :param itm_skew: skew of itm options to find IV, skew = S/K (i.e. 1.75)
    :param otm_skew: skew of otm options to find IV, skew = S/K (i.e. 0.25)
    :return:
    """

    # import modules
    import os
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from tqdm import tqdm
    pd.options.mode.chained_assignment = None  # default='warn'

    # subset call options
    df_calls = coin_df[coin_df['instrument_name'].str.contains('-C')]

    # columns: timestamp, mark_iv, underlying_price, strike, t
    df_calls = df_calls[['timestamp', 'mark_iv', 'underlying_price', 'strike', 't']]

    # create column: skew  = (S/K)
    df_calls['skew'] = df_calls['underlying_price'] / df_calls['strike']

    # sort df_calls
    df_calls = df_calls.sort_values('timestamp')
    df_calls.index = np.arange(0, len(df_calls))

    # 5 minutes dates
    dates = df_calls['timestamp'].unique()
    dates.sort()

    # interpolated iVol df
    df_atm_iv = pd.DataFrame(index=dates, columns=['1w', '2w', '1m', '3m', '6m'])
    df_itm_iv = pd.DataFrame(index=dates, columns=['1w', '2w', '1m', '3m', '6m'])
    df_otm_iv = pd.DataFrame(index=dates, columns=['1w', '2w', '1m', '3m', '6m'])

    # model parameters df
    df_parameters = pd.DataFrame(index=dates, columns=['b0', 'b1', 'b2', 'b3'])

    # standardize step
    step = int(step/5 - 1)

    # loop over each date
    pbar = tqdm(total=len(dates)-step-1)
    for d in range(len(dates)-step-1):

        # start date, end_date
        date = dates[d]
        start_date = date
        end_date = dates[d + step]

        # subset call_df from start_date to end_date
        index_start = df_calls[df_calls['timestamp'] == start_date].index.min()
        index_end = df_calls[df_calls['timestamp'] == end_date].index.max()
        df = df_calls.loc[index_start:index_end]

        # linear model: training
        # iVol = b0 + b1 * skew + b2 * skew ** 2 + b3 * t
        y = df['mark_iv']
        x1 = df['skew']
        x2 = x1 ** 2
        x3 = df['t']
        X_train = pd.concat([x1, x2, x3], axis=1)

        # Create linear regression object
        regr = LinearRegression()

        # Train the model using the training sets
        regr.fit(X_train, y)
        coefficients = [float(regr.intercept_), regr.coef_[0], regr.coef_[1], regr.coef_[2]]

        # prediction: iVol
        # time_to_maturity = 1w, 2w, 1m, 3m, 6m
        maturities = [1/52, 2/52, 1/12, 3/12, 6/12]
        atm_iv = []
        itm_iv = []
        otm_iv = []

        # loop over each time to maturity
        for maturity in maturities:

            # X to predict
            X_atm = pd.DataFrame([atm_skew, atm_skew ** 2, maturity]).T
            X_itm = pd.DataFrame([itm_skew, itm_skew ** 2, maturity]).T
            X_otm = pd.DataFrame([otm_skew, otm_skew ** 2, maturity]).T

            # interpolated iVol
            atm = float(regr.predict(X_atm))
            itm = float(regr.predict(X_itm))
            otm = float(regr.predict(X_otm))

            # append results
            atm_iv.append(atm)
            itm_iv.append(itm)
            otm_iv.append(otm)

        # store results
        df_atm_iv.loc[end_date] = atm_iv
        df_itm_iv.loc[end_date] = itm_iv
        df_otm_iv.loc[end_date] = otm_iv
        df_parameters.loc[end_date] = coefficients
        pbar.update(1)

    # close progress bar
    pbar.close()

    # drop NaN between a step and another
    df_atm_iv.dropna(inplace=True)
    df_itm_iv.dropna(inplace=True)
    df_otm_iv.dropna(inplace=True)
    df_parameters.dropna(inplace=True)

    # float check
    df_atm_iv = df_atm_iv.astype(float)
    df_itm_iv = df_itm_iv.astype(float)
    df_otm_iv = df_otm_iv.astype(float)
    df_parameters = df_parameters.astype(float)

    # skewness: OTM iVol - ITM iVol
    df_skew = pd.DataFrame(df_otm_iv['1w'] - df_itm_iv['1w'])
    df_skew.columns = ['skew']

    # folder name
    folder_name = str(step * 5 + 5) + '_min_calibration'

    # plots
    atm_iv_plot(df_atm_iv=df_atm_iv, coin=coin, step=step, local_folder=local_folder)
    skew_plot(df_skew=df_skew, coin=coin, step=step, local_folder=local_folder)
    iv_parameters_plot(df_parameters=df_parameters, coin=coin, step=step, local_folder=local_folder)

    # summary statistics
    summary_atm = df_atm_iv.describe(include='all').reset_index()
    summary_skew = df_skew.describe(include='all').reset_index()
    summary_parameters = df_parameters.describe(include='all').reset_index()

    # source path
    source_path = os.path.abspath(os.getcwd())

    # local_path: ../reports/coin/?_min_calibration/tables/coin_df_iv.csv
    atm_path = source_path.replace(local_folder,
                                   'reports/' + coin + '/' + folder_name +
                                   '/tables/' + coin + '_atm_iv.parquet')
    skew_path = atm_path.replace('_atm_iv.parquet', '_skew.parquet')
    parameters_path = atm_path.replace('_atm_iv.parquet', '_parameters.parquet')

    # write .csv
    summary_atm.to_parquet(atm_path)
    summary_skew.to_parquet(skew_path)
    summary_parameters.to_parquet(parameters_path)

    # reset index to store data as parquet
    df_atm_iv.reset_index(inplace=True)
    df_skew.reset_index(inplace=True)
    df_parameters.reset_index(inplace=True)

    # paths
    atm_path = atm_path.replace('/tables/' + coin + '_atm_iv.parquet', '/data/' + coin + '_atm_iv.parquet')
    skew_path = atm_path.replace('_atm_iv.parquet', '_skew.parquet')
    parameters_path = atm_path.replace('_atm_iv.parquet', '_parameters.parquet')

    # write .parquet
    df_atm_iv.to_parquet(atm_path)
    df_skew.to_parquet(skew_path)
    df_parameters.to_parquet(parameters_path)

    return print(folder_name + ': done!')
```

Considering that the function that plots the ATM implied volatility dynamics is `atm_iv_plot()`:

```python
# ATM iVol plot
def atm_iv_plot(df_atm_iv, coin, step, local_folder):

    # import modules
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # source_path
    source_path = os.path.abspath(os.getcwd())

    # plot ATM iVol dynamics
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=coin + ' ATM Implied Volatility with linear interpolation',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    fig.text(s='Model calibrated each ' + str(step * 5 + 5) + ' minutes',
             x=0.5, y=0.90, fontsize=18, ha='center', va='center')
    fig.text(0.06, 0.5, 'Estimate [%]', ha='center', va='center', rotation='vertical')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(df_atm_iv)
    ax.legend(list(df_atm_iv.columns), bbox_to_anchor=(.5, 0.03),
              loc="lower center", bbox_transform=fig.transFigure, ncol=len(df_atm_iv.columns), frameon=False)
    plt.margins(x=0)

    # save plot
    file_path = source_path.replace(local_folder,
                                    'reports/' + coin + '/' + str(step * 5 + 5) + '_min_calibration/images/atm_iv.pdf')
    plt.savefig(file_path, dpi=160)
    plt.close()

    # ATM Term Structure: 1m - 3m
    if coin == 'BTC':
        color = 'C0'
    else:
        color = 'C1'
    df_ts = pd.DataFrame(df_atm_iv['1m'] - df_atm_iv['3m'])
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=coin + ' ATM Implied Volatility Term Structure: 1m - 3m',
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    fig.text(s='Model calibrated each ' + str(step * 5 + 5) + ' minutes',
             x=0.5, y=0.90, fontsize=18, ha='center', va='center')
    fig.text(0.06, 0.5, '1m iVol - 3m iVol [%]', ha='center', va='center', rotation='vertical')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(df_ts, color=color)
    ax.hlines(y=0, linestyles='dashed', colors='black', xmin=df_ts.index.min(), xmax=df_ts.index.max(), linewidth=0.5)
    plt.margins(x=0)

    # save plot
    file_path = source_path.replace(local_folder,
                                    'reports/' + coin + '/' + str(step * 5 + 5) +
                                    '_min_calibration/images/atm_iv_term_structure.pdf')
    plt.savefig(file_path, dpi=160)  # save fig
    plt.close()
```

The function that plots the Volatiltiy Skew dynamics is `skew_plot()`:
```python
# iVol skew plot
def skew_plot(df_skew, coin, step, local_folder):

    # import modules
    import os
    import matplotlib.pyplot as plt

    # source path
    source_path = os.path.abspath(os.getcwd())

    # plot Skew dynamics
    if coin == 'BTC':
        color = 'C0'
    else:
        color = 'C1'
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=coin + ' Skew with linear interpolation', x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    fig.text(s='Model calibrated each ' + str(step * 5 + 5) + ' minutes',
             x=0.5, y=0.90, fontsize=18, ha='center', va='center')
    fig.text(0.06, 0.5, 'Skew [%]', ha='center', va='center', rotation='vertical')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(df_skew, color=color)
    ax.hlines(y=0, linestyles='dashed', colors='black',
              xmin=df_skew.index.min(), xmax=df_skew.index.max(), linewidth=0.5)
    plt.margins(x=0)

    # save plot
    file_path = source_path.replace(local_folder,
                                    'reports/' + coin + '/' + str(step * 5 + 5) +
                                    '_min_calibration/images/skew.pdf')
    plt.savefig(file_path, dpi=160)  # save fig
    plt.close()
```

And the function that plots the parameters of the linear model dynamics is `iv_parameters_plot()`:
```python
# iVol parameters plot
def iv_parameters_plot(df_parameters, coin, step, local_folder):

    # import modules
    import os
    import matplotlib.pyplot as plt

    # source_path
    source_path = os.path.abspath(os.getcwd())

    # plot iVol parameters dynamics
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=coin + ' parameters linear interpolation', x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    fig.text(s='Model calibrated each ' + str(step * 5 + 5) + ' minutes',
             x=0.5, y=0.90, fontsize=18, ha='center', va='center')
    fig.text(0.06, 0.5, 'Estimate', ha='center', va='center', rotation='vertical')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(df_parameters)
    ax.legend(['β0', 'β1', 'β2', 'β3'], bbox_to_anchor=(.5, 0.03),
              loc="lower center", bbox_transform=fig.transFigure, ncol=4, frameon=False)
    plt.margins(x=0)

    # save plot
    file_path = source_path.replace(local_folder,
                                    'reports/' + coin + '/' + str(step * 5 + 5) +
                                    '_min_calibration/images/parameters.pdf')
    plt.savefig(file_path, dpi=160)
    plt.close()
```

## Model calibration
The last step is to calibrate the model for all the intervals starting from the `5 minutes` up to `1440 minutes` and then store the results in their directories of interests.
The function is `model_calibration()` and it does the following:

1. Loop the function `iv_linear_interpolation()` over different intervals.

Important Note: the function takes A LOT OF TIME.

```python
# Model calibration
def model_calibration(coin_df, min_time_interval, max_time_interval, step, atm_skew, itm_skew, otm_skew, local_folder, coin):

    # intervals
    intervals = list(range(min_time_interval, max_time_interval, step))

    # calibrate the model for each interval
    for interval in intervals:
        iv_linear_interpolation(coin_df=coin_df, step=interval,
                                atm_skew=atm_skew, itm_skew=itm_skew, otm_skew=otm_skew,
                                local_folder=local_folder, coin=coin)

    return print('Calibration Done!')
```

