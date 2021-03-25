# src: walk through the code

#### Last Update March 23, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

## Project description
In this report I explain how to pull current option data from Deribit’s Rest api, cleaning that data and using it to generate a 3D plot of the entire volatility surface.

The codes described are [`../src/utils.py`](../src/utils.py) and [`../src/preprocessing.py`](../src/preprocessing.py).

## Table of contents

1. [Get Current Option Data from Deribit's API](#get-current-option-data-from-deribits-api)
2. [Pre-processing: add useful metrics](#pre-processing-add-useful-metrics)
3. [Implied Volatility: cubic spline interpolation](#implied-volatility-cubic-spline-interpolation) 
4. [The Implied Volatility surface](#the-implied-volatility-surface)
5. [At-the-Money Adjusted Implied Volatility](#at-the-money-adjusted-implied-volatility)
6. [Simple parametrized surface model](#simple-parametrized-surface-model)
7. [Move plots to the images folder](#move-plots-to-the-images-folder)
8. [Supported versions](#supported-versions)
9. [References](#references)

## Get Current Option Data From Deribit’s API
Deribit’s api provides access to option data traded on the exchange. 
Acquiring the data is a four-step process:
1. Get a list of all active options.
2. Remove in-the-money options, options with a strike that is more than 30% away from the current price and options that have more than 90 days until expiration.
3. Loop through the list of options to retrieve the relevant data.

the code is available at è [`../src/utils.py`](../src/utils.py):
```python
# import modules
import pandas as pd
import numpy as np

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
```

## Pre-processing: add useful metrics
The second step is to add different metrics to the data: 
1. days to expiration. 
2. strike price.
3. [moneyness](https://www.investopedia.com/terms/m/moneyness.asp).

In [`../src/preprocessing.py`](../src/preprocessing.py) the function `data_preprocessing()` performs these tasks:
```python
# import modules
import pandas as pd
import numpy as np
from SpotImpliedVolatilitySurfaceAnalysis.src.utils import get_all_option_data

# data pre-processing
def data_preprocessing(coin):
    """

    :param coin: 'BTC' or 'ETH'
    :return: pandas.DataFrame with relevant financial data
    """
    # disable false positive warning, default='None'
    pd.options.mode.chained_assignment = None

    # get data
    df = get_all_option_data(coin=coin)

    # add additional metrics to data
    df['t'] = np.nan;
    df['strike'] = np.nan

    # indexing index
    index = df[1:].index.map(lambda x: x.split('-'))

    # calculate days until expiration
    days = [element[1] for element in index]
    days = (pd.to_datetime(days) - pd.Timestamp.today()).days

    # add days to expiration
    df.t[1:] = np.array(days)

    # Pull strike from instrument name
    strike = [int(element[2]) for element in index]

    # add strike
    df.strike[1:] = strike

    # calculate moneyness
    df['m'] = np.log(df['last_price'][0] / df['strike'])

    return df
```

## Implied Volatility: cubic spline interpolation
The next step is to estimate the complete implied volatility surface.
The first method is the [cubic spline interpolation](https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation) with the observed data points.

These are the steps:
1. Sort data by days to expiration and strike price.
2. X-axis: Scale the Strike price as the ratio between the current price and the strike.
3. Y-axis: Days to expiration
4. Z-axis: Implied volatility
5. Make 3-D coordinate arrays for vectorized evaluations of 3-D scalar/vector fields over 3-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.
6. Cubic Spline Interpolation to find the Z-axis
7. Consider only Options whose moneyness is between 0.95 and 1.05

In [`../src/preprocessing.py`](../src/preprocessing.py) the function `cubic_spline_interpolation()` performs these tasks:
```python
# import modules
import numpy as np
from scipy import interpolate
import pandas as pd

# cubic spline interpolation for implied volatility
def cubic_spline_interpolation(df):
    """

    :param df: data_preprocessing() df
    :return: xyz pandas.DataFrame for Implied Volatility surface, X, Y, Z coordinantes
    """

    # get only options with time to expiration > 0
    df_ = df.iloc[1:].sort_values(['t', 'strike']).query('t>0')

    # x, y, z
    x = (df['last_price'][0] / df_['strike'])
    y = df_['t']
    z = df_['mark_iv'] / 100

    # X, Y
    X, Y = np.meshgrid(np.linspace(.95, 1.05, 99), np.linspace(1, np.max(y), 100))

    # Z: spline cubic interpolation
    Z = interpolate.griddata(np.array([x, y]).T, np.array(z), (X, Y), method='cubic')

    # xyz DataFrame for Implied Volatility surface
    xyz = pd.DataFrame({'x': x, 'y': y, 'z': z})

    # get only options with moneyness between .95 and 1.05
    xyz = xyz.query('x>0.95 & x<1.05')

    return xyz, X, Y, Z, x, y
```

## The Implied Volatility surface
Now it is possible to visualize the Implied Volatility surface. 

Then the plot is stored in [`../deliverables`](../deliverables/)

In [`../src/utils.py`](../src/utils.py) the function `plot_iv_surf()` performs the task:


```python
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import datetime as dt

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

    # plot 3D surface
    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.8, cmap='RdYlGn_r')

    # add legend
    ax.legend()

    # save fig
    plt.savefig(file_path, dpi=80)
    plt.close()
```

## At-the-Money Adjusted Implied Volatility
Since not all the observations have current data I try to estimate the current Implied Volatility in this way:

<div align="center"> &sigma;<sub>IV</sub> = &sigma;<sub>IV</sub> + &Delta;/&nu; * dP </div>

In [`../src/preprocessing.py`](../src/preprocessing.py) the function `atm_adjusted_IV_data()` performs the task:

```python
# import modules
from scipy import interpolate
import numpy as np
import pandas as pd

# ATM adjusted Implied Volatility data
def atm_adjusted_IV_data(df, spline_df, X, Y, x, y):

    price_diff = df['mark_price'][0] - df['underlying_price']
    df['iv_adj'] = df['mark_iv'] + (df['greeks.delta'] * price_diff) / df['greeks.vega']

    # get only options with time to expiration > 0
    df_ = df.iloc[1:].sort_values(['t', 'strike']).query('t>0')

    # spline cubic interpolation Z
    Z = interpolate.griddata(np.array([x, y]).T, np.array(df_['iv_adj'] / 100), (X, Y), method='cubic')

    # final df with adjusted IV
    iv_df_adj = pd.DataFrame(Z, index=np.linspace(10, np.max(spline_df.y), 100), columns=np.linspace(.95, 1.05, 99))

    return iv_df_adj
```

Then it is possible to visualize the ATM Adjusted Implied Volatility comparing BTC and ETH.

In [`../src/utils.py`](../src/utils.py) the function `plot_atm_adj_iv()` performs the task:

```python
import matplotlib.pyplot as plt
import datetime as dt

# define ATM adjusted Implied Volatility Plot
def plot_atm_adj_iv(df_adj_iv_btc, df_adj_iv_eth):
    
    # file path
    file_path = "./atm_adj_iv.png"
    
    # setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.text(s='ATM Adjusted Implied Volatility', x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    fig.text(s=dt.date.today().strftime("%B %d, %Y"), x=0.5, y=0.90, fontsize=14, ha='center', va='center')
    fig.text(0.06, 0.5, 'Implied Volatility', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'Days to Expiration', ha='center', va='center')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # plot
    ax1.plot(df_adj_iv_btc.loc[:, 1], label='BTC', color='#F2A900')
    ax2.plot(df_adj_iv_eth.loc[:, 1], label='ETH', color='#3c3c3d')
    
    # legend
    ax1.legend(loc='upper right', frameon=False)
    ax2.legend(loc='upper right', frameon=False)
    
    # save fig
    plt.savefig(file_path, dpi=80)
    plt.close()
```

## Simple parametrized surface model
Now the implied volatility surface is estimated as a linear regression with the features designed to model the skew and time curvature.

The first step is to weight the IV observations by strike and open interest.
The farther away an option is from the current price, the lower the weight. Moreover, options with a higher open interest receive a higher weight.

In [`../src/preprocessing.py`](../src/preprocessing.py) the function `weighted_lr_data()` performs the task:

```python
# import modules
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Weighted linear regression
def weighted_lr_data(df):

    price_diff = df['mark_price'][0] - df['underlying_price']
    df['iv_adj'] = df['mark_iv'] + (df['greeks.delta'] * price_diff) / df['greeks.vega']

    # get only options with time to expiration > 0
    df_ = df.iloc[1:].sort_values(['t', 'strike']).query('t>0')

    # weights by moneyness and open interest
    np.seterr(divide='ignore')
    weights = 1 / (1 + ((df_['m'] ** 2))) + ((np.log(df_['open_interest']).replace(-np.inf, 0) / np.log(df_['open_interest']).replace(-np.inf, 0).sum()))

    df_reg = df.iloc[1:].sort_values(['t', 'strike'])

    t = np.sqrt(df_['t'] / 365)
    m = np.log(df['last_price'][0] / df_['strike']) / t
    X = pd.DataFrame({'M': m, 'M2': m ** 2, 'M3': m ** 3, 't': t, 'tM': t * m, 't2': t ** 2})
    y = (df_['iv_adj'] / 100)
    X = sm.add_constant(X)
    np.seterr(divide='warn')

    return y, X, weights
```

It is possible to compare the Observation weight by Moneyness and Open Interest of BTC and ETH visualizing the two plots.

In [`../src/utils.py`](../src/utils.py) the function `moneyness_openinterest()` performs the task:

```python
# import modules
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

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

    # plot setup
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
    
    # scatter plot observations
    ax1.scatter(df_btc_['m'], btc_weights, label='BTC', color='#F2A900')
    ax2.scatter(df_eth_['m'], eth_weights, label='ETH', color='#3c3c3d')
    
    # add legend
    ax1.legend(loc='upper right', frameon=True)
    ax2.legend(loc='upper right', frameon=True)

    # save fig
    plt.savefig(file_path, dpi=160)
    plt.close()
```

Then, to find parametrized implied volatility surface the Weighted Least Squares (WLS) regression results are used as input for the function `simple_parametrized_surface_model()`.

```python
import numpy as np
from scipy import interpolate
import pandas as pd

# Simple parametrized surface model for IV surface
def simple_parametrized_surface_model(df, model, X):

    df_ = df.iloc[1:].sort_values(['t', 'strike']).query('t>0')
    x = df['last_price'][0] / df_['strike']
    y = df_['t']
    z = model.predict(X)
    X, Y = np.meshgrid(np.linspace(.95, 1.05, 99), np.linspace(1, np.max(y), 100))
    Z = interpolate.griddata(np.array([x, y]).T, np.array(z), (X, Y), method='linear')
    xyz = pd.DataFrame({'x': x, 'y': y, 'z': (df_['mark_iv'] / 100)})
    xyz = xyz.query('x>0.95 & x<1.05')

    return X, Y, Z, xyz
```

Then, the simple parametrized surface model is plotted using the `plot_iv_surf()` function.



## Move plots to the images folder
The last step consists in moving all the generated plots form  [`../deliverables`](../deliverables/) to [`../reports/images/`](../reports/images).

This is a three-step process:
1. identify the source path.
2. identify the destination path
3. move all the `.png` file from the `source_path` to `destination_path`.

In [`../src/utils.py`](../src/utils.py) the function `move_files()` performs the task:

```python
# import modules
import os, shutil

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
```

## Supported versions
This setup script has been tested against Python 3.8.5 and Deribit API v2.0.1.

## References
[Deribit API v2](https://docs.deribit.com/?python#deribit-api-v2-0-1)

[Interpolation with scipy.interpolate](https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html)

[WLS regression with statsmodels.api.linear_regression.WLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html)
