# Walk through the code

#### Last Update September 14, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

## Project description
In this report I explain how to analyze the Cryptocurrency Spot Derivatives Market.

The function described are in [`../SpotAnalysis/src/utils.py`](../src/utils.py).

## Table of contents

1. [Create the Environment: directories and sub-directories](#create-the-environment-directories-and-sub-directories)
2. [Download Tick Option Data](#download-tick-option-data)
3. [Data Pre-processing](#data-preprocessing)   
4. [Implied Volatility Smile](#implied-volatility-smile) 
5. [Implied Volatility Surface and ATM Term-Structure](#implied-volatility-surface-and-atm-term-structure)
6. [Greeks: Surface and ATM term-structure](#greeks-surface-and-atm-term-structure)
6. [Implied Volatility-Delta Surface](#implied-volatility-delta-surface)

## Create the Environment: directories and sub-directories
The first step is to create all the directories and sub-directories needed to perform the analysis.

* the main subdirectories to make are: `../reports/images/BTC` and `../reports/images/ETH`.
* Then, `../reports/images/coin/greeks` and `../reports/images/coin/volatility`.
* Inside `.../greeks`: `../reports/images/coin/greeks/atm_term_structure` and `../reports/images/coin/greeks/surface`.
* Inside `.../volatility`: `cubic_interpoaltion`, `linear_interpolation`, `nearest_interpolation`

In [`../src/utils.py`](../src/utils.py) the function `create_env()` performs these tasks:

```python
# Create Environment
def create_env(local_folder):

    # import modules
    import os

    # source path
    source_path = os.path.abspath(os.getcwd())

    # main folders: coins
    coins = ['BTC', 'ETH']

    # sub folders for each coin
    for coin in coins:

        # create ../reports/images/coin
        destination_path = source_path.replace(local_folder, 'reports/images/' + coin)
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        # create sub-directory: ../reports/images/coin/volatility
        sub_directory = destination_path + '/volatility'
        if not os.path.exists(sub_directory):
            os.mkdir(sub_directory)

        # create sub-directory: ../reports/images/coin/volatility/method_interpolation
        interpolations = ['nearest', 'linear', 'cubic']
        for interpolation in interpolations:
            sub_directory = destination_path + '/volatility/' + interpolation + '_interpolation'
            if not os.path.exists(sub_directory):
                os.mkdir(sub_directory)

        # create sub-directory: ../reports/images/coin/greeks
        sub_directory = destination_path + '/greeks'
        if not os.path.exists(sub_directory):
            os.mkdir(sub_directory)

        # create sub-directory: ../reports/images/coin/greeks/surface
        sub_directory = sub_directory + '/surface'
        if not os.path.exists(sub_directory):
            os.mkdir(sub_directory)

        # create sub-directory: ../reports/images/coin/greeks/atm_term_structure
        sub_directory = sub_directory.replace('/surface', '/atm_term_structure')
        if not os.path.exists(sub_directory):
            os.mkdir(sub_directory)

    print('Environment created!')

    return print('----------------------------------------------------------------------')
```

So that the folder structure becomes the following:

~~~~
    ../SpotAnalysis/
        README.md
            deliverables/
                run-spot-analysis.py
        src/
            utils.py
        reports/
            spot-derivatives-analysis.md
            walk-thorugh-the-code.md
            images/
                BTC/
                    greeks/
                        atm_term_structure/
                        surface/
                    volatility/
                        cubic_interpolation/
                        linear_interpolation/
                        nearest_interoplation/
                ETH/
                    greeks/
                        atm_term_structure/
                        surface/
                    volatility/
                        cubic_interpolation/
                        linear_interpolation/
                        nearest_interoplation/                      
~~~~


## Download Tick Option Data
The second step is to download current tick data for the set of active options on Deribit via its public API.

The steps are the following:
1. Get a list of all active options from the Deribit API.
2. Filter options and keep only options that are active
3. Loop through all filtered options name to get current data.

Step 1 is performed by the function `get_all_active_options()`:
```python
# Get a list of all active options from the Deribit API.
def get_all_active_options(coin):
    """

    :param coin: 'BTC' or 'ETH'
    :return: list of all active options from the Deribit API
    """

    # import modules
    import urllib.request
    import json
    import pandas as pd

    # url connection
    url = "https://test.deribit.com/api/v2/public/get_instruments?currency=" + coin + "&kind=option&expired=false"
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    data = pd.DataFrame(data['result']).set_index('instrument_name')
    data['creation_date'] = pd.to_datetime(data['creation_timestamp'], unit='ms')
    data['expiration_date'] = pd.to_datetime(data['expiration_timestamp'], unit='ms')

    print(f'{data.shape[0]} active options')

    return data
```

Step 2 is performed by the function `filter_options()`:
```python
# Filter options based on data available from 'get_instruments'
def filter_options(price, active_options):
    """

    :param price: current coin price
    :param active_options: list of active options
    :return: list of active options after filtration
    """

    # import modules
    import pandas as pd

    # Get Put/Call information
    pc = active_options.index.str.strip().str[-1]

    # Set "moneyness"
    active_options['m'] = active_options['strike'] / price
    active_options.loc[pc == 'P', 'm'] = -active_options['m']

    # Set days until expiration
    active_options['t'] = (active_options['expiration_date'] - pd.Timestamp.today()).dt.days

    return active_options
```

Step 3 is performed by the functions `get_tick_data()` and `get_all_option_data()`:
```python
# Get Tick data for a given instrument from the Deribit API
def get_tick_data(instrument_name):

    # import modules
    import urllib.request
    import json
    import pandas as pd

    # url connection
    url = f"https://test.deribit.com/api/v2/public/ticker?instrument_name={instrument_name}"
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())

    # convert json to pandas.DataFrame
    data = pd.json_normalize(data['result'])
    data.index = [instrument_name]

    return data


# Loop through all filtered options to get the current 'ticker' data
def get_all_option_data(coin):

    # get tick data Perpetual
    option_data = get_tick_data(coin + '-PERPETUAL')

    # get active options
    options = filter_options(price=option_data['last_price'][0], active_options=get_all_active_options(coin=coin))
    for o in options.index:
        option_data = option_data.append(get_tick_data(o))

    return option_data
```

## Data Preprocessing
The third step is to clean and prepare the data for the analysis.

The function the performs the task is `data_preprocessing()` and pull from the data the following metrics:
1. time-to-maturity.
2. strike price.
3. expiration day.
4. moneyness.

```python
# data pre-processing
def data_preprocessing(coin):
    """

    :param coin: 'BTC' or 'ETH'
    :return: pandas.DataFrame with relevant financial data
    """

    # import modules
    import pandas as pd
    import numpy as np

    # disable false positive warning, default='None'
    pd.options.mode.chained_assignment = None

    # get data
    print('Get ' + coin + ' options data')
    df = get_all_option_data(coin=coin)

    # add additional metrics to data
    df['t'] = np.nan
    df['strike'] = np.nan
    df['expiration'] = np.nan

    # indexing index
    index = df[1:].index.map(lambda x: x.split('-'))

    # calculate days until expiration
    days = [element[1] for element in index]
    maturity = days
    days = (pd.to_datetime(days) - pd.Timestamp.today()).days

    # add days to expiration
    df.t[1:] = np.array(days)

    # Pull strike from instrument name
    strike = [int(element[2]) for element in index]

    # add strike
    df.strike[1:] = strike

    # calculate moneyness
    df['m'] = df['strike'] / df['last_price'][0]

    # pull maturity
    maturity = pd.to_datetime(maturity) + pd.DateOffset(hours=10)
    maturity = maturity.astype('int64')
    df.expiration[1:] = maturity

    # consider only t>0
    df = df.query('t>0')

    print('additional metrics added')
    print('----------------------------------------------------------------------')

    return df
```

## Implied Volatility Smile
The fourth step consists in looking at the first stylized fact of the options: the implied volatility smile.
The function `iv_smile()` does the following:
1. Subset the full dataset and keep only Call Options.
2. Consider only options whose time to maturity is closer to `90 days`.
3. Sort the values by `moneyness`.
4. Produce and store the Volatility Smile plot.

```python
#  volatility smile plot
def iv_smile(coin_df, coin, time_to_maturity, cwd):

    # import modules
    import os
    import matplotlib.pyplot as plt
    import datetime as dt

    # file path
    source_path = os.path.abspath(os.getcwd())
    file_path = source_path.replace(cwd, "/reports/images/" + coin + "/volatility/volatility-smile.pdf")

    # subset df
    call_df = coin_df[coin_df['instrument_name'].str.contains('-C')]

    # pull days to maturity
    days_to_maturity = list(call_df['t'].unique())
    maturity = min(days_to_maturity, key=lambda x: abs(x-time_to_maturity))

    # subset df for the maturity
    df = call_df[call_df['t'] == maturity].sort_values('m')

    # plot volatility smile
    if coin == 'BTC':
        color = 'C0'
    else:
        color = 'C1'

    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=coin + ' Implied Volatility Smile \n' + dt.date.today().strftime("%B %d, %Y"),
             x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.text(0.06, 0.5, 'Implied Volatility [%]', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.04, 'Moneyness (K/S)', ha='center', va='center')
    ax.plot(df.m, df.mark_iv, linestyle='--', marker='o', color=color)
    ax.legend(['Observed IV, ' + str(int(maturity)) + ' days to maturity'], bbox_to_anchor=(.5, 0.0),
              loc="lower center", bbox_transform=fig.transFigure, ncol=len(days_to_maturity), frameon=False)
    plt.savefig(file_path, dpi=160)  # save fig
    plt.close()

    print(coin + ' volatility smile plot: done!')

    return print('----------------------------------------------------------------------')
```

## Implied Volatility Surface and ATM term-structure
The fifth step consists in estimating the Implied Volatility Surface and the ATM Implied Volatility term-structure.
The function `implied_vol()` does the following:
1. Subset the Data-set for Call Options solely.
2. Set the x, y and z values as `moneyness`, `time-to-maturity` and `market implied volatility` respectively.
3. Creates Griddata for 3D interpolation.
4. Interpolates the data in three methods: `nearest`, `linear` and `cubic`.
5. Produce the Implied Volatility surface for all the three methods and store the results.
6. Extrapolates the ATM observations from the interpolated data.
7. Produce the ATM implied volatility term-structure plot and store the results.

```python
# iVol surface and ATM structure
def implied_vol(coin_df, coin, cwd):

    # import modules
    import os
    import numpy as np
    import datetime as dt
    import pandas as pd
    from scipy import interpolate
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # source path
    source_path = os.path.abspath(os.getcwd())

    # subset df
    call_df = coin_df[coin_df['instrument_name'].str.contains('-C')]
    call_df = call_df.sort_values(['t', 'm']).query('t>0')

    # x, y, z
    x = call_df['m']
    y = call_df['t']
    z = call_df['mark_iv'] / 100

    # points, values
    points = np.array([x, y]).T
    values = np.array(z)

    # grid_x, grid_y
    grid_x = np.linspace(np.min(x), np.max(x), 5*len(x))
    grid_y = np.linspace(np.min(y), np.max(y), 5*len(y))

    # grid
    X, Y = np.meshgrid(grid_x, grid_y)

    # interpolation
    interpolations = ['nearest', 'linear', 'cubic']

    if coin == 'BTC':
        color = 'C0'
    else:
        color = 'C1'

    # try different interpolation methods
    for interpolation in interpolations:

        # interpolate Z
        Z = interpolate.griddata(points, values, (X, Y), method=interpolation)
        Z = np.array(pd.DataFrame(Z).bfill().ffill().iloc[1:-1, 1:-1])
        X = np.array(pd.DataFrame(X).iloc[1:-1, 1:-1])
        Y = np.array(pd.DataFrame(Y).iloc[1:-1, 1:-1])

        # Surface plot
        plt.rcParams['font.family'] = 'serif'  # set font family: serif
        fig = plt.figure(3)
        ax = plt.axes(projection='3d')
        ax.set_title(coin + ' Implied Volatility Surface, ' + interpolation + ' interpolation \n' +
                     dt.date.today().strftime("%B %d, %Y"))
        ax.set_zlabel('Implied Volatility')
        plt.xlabel('Moneyness (K/S)')
        plt.ylabel('Days To Expiration')
        ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: '{:.0%}'.format(z)))
        ax.scatter3D(x, y, z, label='Observed IV')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8, cmap='RdYlGn_r')
        ax.set_zlim(bottom=0)
        ax.legend(['Observed IV'], bbox_to_anchor=(.5, 0.0),
                  loc="lower center", bbox_transform=fig.transFigure, ncol=1, frameon=False)

        # save plot
        file_path = source_path.replace(cwd, '/reports/images/' + coin + "/volatility/" + interpolation +
                                        "_interpolation/volatility-surface.pdf")
        plt.savefig(file_path, dpi=160)
        plt.close()

        print(coin + ' volatility surface with ' + interpolation + ' interpolation: done!')

        # ATM interpolated term structure
        atm_position = (np.abs(grid_x - 0)).argmin()
        x_atm = Y[:, atm_position]
        y_atm = Z[:, atm_position] * 100

        # ATM plot
        plt.rcParams['font.family'] = 'serif'  # set font family: serif
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        fig.text(s=coin + ' ATM Implied Volatility Interpolated Term Structure \n' + dt.date.today().strftime(
            "%B %d, %Y"), x=0.5, y=0.95, fontsize=20, ha='center', va='center')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.text(0.06, 0.5, 'Implied Volatility [%]', ha='center', va='center', rotation='vertical')
        fig.text(0.5, 0.04, 'Time to Maturity [days]', ha='center', va='center')
        ax.plot(x_atm, y_atm, linestyle='--', color=color)

        # save plot
        file_path = source_path.replace(cwd, '/reports/images/' + coin + "/volatility/" + interpolation +
                                        "_interpolation/atm-vol-structure.pdf")
        plt.savefig(file_path, dpi=160)
        plt.close()

        print(coin + ' ATM volatility structure with ' + interpolation + ' interpolation: done!')

    return print('----------------------------------------------------------------------')
```

## Greeks: Surface and ATM term-structure
The sixth step is to look at the different Greeks surfaces and ATM term-structure. The Greeks under considerations are: `Delta`, `Gamma`, `Rho`, `Theta`.
The function `greeks()` does the following:
1. Subset the Data-set for Call Options solely.
2. Sort values by `time-to-maturity` and `moneyness`.
3. Set x, y as `moneyness` and `time-to-maturity`.
4. Interpolate with `linear` method for each greek.
5. Produce a Greek surface and store the results.
6. Extrapolates the ATM observations from the interpolated data.
7. Produce the ATM greek term-structure plot and store the results for each greek.

```python
# Greeks Surface and ATM term structure plots
def greeks(coin_df, coin, cwd):

    # import modules
    import os
    import numpy as np
    import datetime as dt
    import pandas as pd
    from scipy import interpolate
    import matplotlib.pyplot as plt

    # file path
    source_path = os.path.abspath(os.getcwd())

    # subset df
    call_df = coin_df[coin_df['instrument_name'].str.contains('-C')]
    call_df = call_df.sort_values(['t', 'm']).query('t>0')

    # greeks
    greeks_list = ['greeks.delta', 'greeks.gamma', 'greeks.rho', 'greeks.theta']

    # x, y
    x = call_df['m']
    y = call_df['t']

    # points
    points = np.array([x, y]).T

    # grid_x, grid_y
    grid_x = np.linspace(np.min(x), np.max(x), 5*len(x))
    grid_y = np.linspace(np.min(y), np.max(y), 5*len(y))

    # grid
    X, Y = np.meshgrid(grid_x, grid_y)

    if coin == 'BTC':
        color = 'C0'
    else:
        color = 'C1'

    # plot for each greek
    for greek in greeks_list:

        # z values
        z = call_df[greek]
        values = np.array(z)
        greek = greek.split('.', 1)[1].title()

        # Z: linear interpolation
        Z = interpolate.griddata(points, values, (X, Y), method='linear')
        Z = np.array(pd.DataFrame(Z).bfill().ffill().iloc[1:-1, 1:-1])
        X = np.array(pd.DataFrame(X).iloc[1:-1, 1:-1])
        Y = np.array(pd.DataFrame(Y).iloc[1:-1, 1:-1])

        # Surface plot
        plt.rcParams['font.family'] = 'serif'  # set font family: serif
        fig = plt.figure(3)
        ax = plt.axes(projection='3d')
        ax.azim = 240
        ax.set_title(coin + ' ' + greek + ' Call Surface \n' + dt.date.today().strftime("%B %d, %Y"))
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(greek, rotation=90)
        plt.xlabel('Moneyness (K/S)')
        plt.ylabel('Days To Expiration')
        ax.scatter3D(x, y, z, label='Observed' + greek)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8, cmap='RdYlGn_r')
        ax.legend(['Observed ' + greek], bbox_to_anchor=(.5, 0.0),
                  loc="lower center", bbox_transform=fig.transFigure, ncol=1, frameon=False)

        # file path
        file_path = source_path.replace(cwd,
                                        '/reports/images/{0}/greeks/surface/{1}-surface.pdf'.format(coin, greek))
        plt.savefig(file_path, dpi=160)
        plt.close()
        print(coin + ' ' + greek + ' surface plot: done!')

        # ATM interpolated term structure
        atm_position = (np.abs(grid_x - 0)).argmin()
        x_atm = Y[:, atm_position]
        y_atm = Z[:, atm_position]

        # term structure plot
        plt.rcParams['font.family'] = 'serif'  # set font family: serif
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        fig.text(s=coin + ' ATM Calls Interpolated ' + greek + ' Structure \n' + dt.date.today().strftime("%B %d, %Y"),
                 x=0.5, y=0.95, fontsize=20, ha='center', va='center')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.text(0.06, 0.5, greek, ha='center', va='center', rotation='vertical')
        fig.text(0.5, 0.04, 'Time to Maturity [days]', ha='center', va='center')
        ax.plot(x_atm, y_atm, linestyle='--', color=color)
        if greek == 'Delta':
            ax.set_ylim([0, 1])

        # file path
        file_path = source_path.replace(cwd,
                                        '/reports/images/{0}/greeks/atm_term_structure/atm-{1}-structure.pdf'.format(
                                            coin, greek))
        plt.savefig(file_path, dpi=160)
        plt.close()
        print(coin + ' ' + greek + ' atm structure plot: done!')

    return print('----------------------------------------------------------------------')
```

## Implied Volatility-Delta Surface
The seventh and last step is to look at the Implied Volatility-Delta Surface. 
The function that performs the task is `iv_delta_surface()` and does the following:

1. Subset the Data-set for Call Options and sort values by `time-to-maturity` and `Delta`.
2. Subset the Data-set for Put Options and sort values by `time-to-maturity` and `Delta`.
3. Concat the Call and Put data-sets and sort values by `tim-to-maturity` and `Delta`.   
3. Set x, y and, z as `Delta`, `time-to-maturity` and `market implied volatility` respectively.
4. Create a Gridddata with the parameters and interpolate with a linear method.
5. Produce the iVol-Delta Surface plot and store the results.

```python
# iVol Delta Surface
def iv_delta_surface(coin_df, coin, cwd):

    # import modules
    import os
    import numpy as np
    import datetime as dt
    import pandas as pd
    from scipy import interpolate
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # file path
    source_path = os.path.abspath(os.getcwd())
    file_path = source_path.replace(cwd, '/reports/images/' + coin + "/volatility/iv-delta-surface.pdf")

    # subset call df
    call_df = coin_df[coin_df['instrument_name'].str.contains('-C')]
    call_df = call_df.sort_values(['t', 'greeks.delta']).query('t>0 & m>=0')

    # subset put df
    put_df = coin_df[coin_df['instrument_name'].str.contains('-P')]
    put_df = put_df.sort_values(['t', 'greeks.delta']).query('t>0 & m<=0')

    # df
    df = pd.concat([call_df, put_df], axis=0)
    df = df.sort_values(['t', 'greeks.delta'])

    # x, y, z
    x = df['greeks.delta']
    y = df['t']
    z = df['mark_iv'] / 100

    # X, Y
    # min_? is minimum bound, max_? is maximum bound,
    #   dim_? is the granularity in that direction
    min_x, max_x, dim_x = (np.min(x), np.max(x), 5*len(x))
    min_y, max_y, dim_y = (np.min(y), np.max(y), 5*len(y))
    X, Y = np.meshgrid(np.linspace(min_x, max_x, dim_x), np.linspace(min_y, max_y, dim_y))

    # Z: linear interpolation
    Z = interpolate.griddata(np.array([x, y]).T, np.array(z), (X, Y), method='linear')
    Z = np.array(pd.DataFrame(Z).bfill().ffill().iloc[:, 1:-1])
    X = np.array(pd.DataFrame(X).iloc[:, 1:-1])
    Y = np.array(pd.DataFrame(Y).iloc[:, 1:-1])

    # plot
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig = plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.set_title(coin + ' Implied Volatility - Delta Surface \n' + dt.date.today().strftime("%B %d, %Y"))
    ax.set_zlabel('Implied Volatility')
    plt.xlabel('Delta')
    plt.ylabel('Days To Expiration')
    ax.zaxis.set_major_formatter(FuncFormatter(lambda z, _: '{:.0%}'.format(z)))
    ax.scatter3D(x, y, z, label='Observed IV')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.8, cmap='RdYlGn_r')
    ax.set_xlim([-1, 1])
    ax.set_zlim(bottom=0)
    ax.set_xticks([-.8, -.4, 0, .4, .8])
    ax.set_xticklabels(['10P', '30P', 'ATM', '30C', '10C'])
    ax.legend(['Observed IV'], bbox_to_anchor=(.5, 0.0),
              loc="lower center", bbox_transform=fig.transFigure, ncol=1, frameon=False)

    # save plot
    plt.savefig(file_path, dpi=160)
    plt.close()
    print(coin + ' iVol Delta surface: done!')

    return print('----------------------------------------------------------------------')
```