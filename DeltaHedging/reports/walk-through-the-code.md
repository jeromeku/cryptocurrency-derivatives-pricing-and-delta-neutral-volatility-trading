# Walk through the code

#### Last Update September 17, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

## Project description
In this report I explain how to back-test and analyze a delta-neutral trading strategy.

The function described are in [`../DeltaHedging/src/utils.py`](../src/utils.py), [`../DeltaHedging/src/backtest.py`](../src/backtest.py) 
and, [`../DeltaHedging/src/variables.py`](../src/variables.py).

## Table of contents

1. [Main Variables](#main-variables)
2. [Load data and pre-processing](#load-data-and-pre-processing)
3. [Back-test for a single Delta-Neutral strategy](#back-test)
4. [Back-testing all strategies](#back-testing-all-strategies)
5. [Quantitative strategy performance](#quantitative-strategy-performance)
6. [Visualize portfolios performance](#visualize-portfolios-performance)


## Main Variables
The first step is to set up the main variables that will be used later on.
The variables are in [`../DeltaHedging/src/variables.py`](../src/variables.py).
Changing them will change the back-test configuration and the final output.

```python
# Main Variables
#
# - lag:              the trade is entered with a lag after the trade opportunity arise
#                     (i.e. lag=1 --> 5min later, lag=2 --> 10min)
# - fee:              percentage fee applied every time a transition is made: open, close
#                     (i.e. fee = 6% )
# - margin:           percentage of the capital posted as collateral when a short position is entered
#                     (i.e. margin = .5 --> short 1 and 1 as collateral)
# - quantile_iv:      tail events to consider to enter a trade evaluated as median - quantile_iv
#                     (i.e. quantile_iv = .4 --> 0.5-0.4=0.1 --> 10% tail left and 10% tail right)
# - transaction_cost: percentage fee applied at the end of every day to re-balance the portfolio

lag = 1
fee = .06
margin = .5
quantile_iv = .4
transaction_cost = .06
```

## Load data and pre-processing
The second step is to load the data from [`/GetServerData/data`](/GetServerData/data) to the working environment and to process the data.
The function is `load_data()` and it does the following:

1. read the feather data-set.
2. pull `strike`, `datetime`, `time-to-maturity`

In [`../src/utils.py`](../src/utils.py) the function `load_data()` performs these tasks:
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

## Back-test
The third step is to set up the back-test. 
All the functions described here are in [`../src/backtest.py](../src/backtest.py).

The steps are the following:
1. [Evaluate the trading positions](#evaluate-the-trading-positions)
2. [Evaluate the Call, Put and, Underlying cumulative returns, positions, delta](#evaluate-the-call-put-and-underlying-cumulative-returns-positions-delta)
3. [Strategy optimization: close positions at margin calls and evaluate new positions](#strategy-optimization-close-positions-at-margin-calls-and-evaluate-new-positions)
4. [Dollar performance for Call, Put and, Underlying given the final positions](#dollar-performance-for-call-put-and-underlying-given-the-final-positions)
5. [Delta-Neutral strategy performance and positions](#delta-neutral-strategy-performance-and-positions)

### Evaluate the trading positions
The first sub-step consists in evaluating the trading positions given the trading signals.
The function is `trading_position()` and it does the following:

1. Keep a position open until a convergence.
2. Shift the positions with a lag to avoid look-ahead bias.

```python
# evaluate trading position
def trading_position(signal, df, lag):

    # import modules
    import pandas as pd
    import numpy as np

    # trading signal: check
    signal = pd.Series(signal)
    signal.index = df['timestamp']

    # position: empty
    position = pd.Series(np.empty(len(df))).round()

    # running position up to convergence
    for i in range(len(position)-1):
        if (signal.iloc[i+1] != position[i]) & (position[i] == 1) & \
                (signal.iloc[i+1] <= 0) & (signal.iloc[i+1] == 0):
            position[i+1] = position[i]
        else:
            if (signal.iloc[i+1] != position[i]) & (position[i] == -1) & \
                    (df['difference'].iloc[i+1] >= 0) & (signal.iloc[i+1] == 0):
                position[i+1] = position[i]
            else:
                position[i+1] = signal.iloc[i+1]

    # lag position
    position = position.shift(lag)
    position.index = df['timestamp']
    position.iloc[:lag] = 0

    return position
```

### Evaluate the Call, Put and, Underlying cumulative returns, positions, delta
The second sub-step consists in evaluating the cumulative returns, considering the trading positions, for a Call, a Put and the underlying.
The function is `cumret_strategy()` and it does the following:

1. Evaluate the returns of each instrument.
2. Charge a percentage fee every time a position is entered and exited.
3. Evaluate the performance considering the collateral.
4. Close the position when a margin call arise and evaluate new performance.
5. Store `Delta`, `Position` and `Balance` for each instrument.

```python
# evaluate call, put, underlying performance and position
def cumret_strategy(df, position, fee, margin):

    # import modules
    import pandas as pd
    import numpy as np

    # position
    position[-1] = 0
    trading_signal = pd.Series(df['initial_trading_signal'])
    trading_signal.index = df['timestamp']

    # initialize results
    df_result = pd.DataFrame()

    # elements
    elements = ['call', 'put', 'underlying']
    for element in elements:

        # account and collateral
        account = np.empty(len(df))
        collateral = np.empty(len(df))
        account[0] = 1

        # returns
        ret = df[element + '_price'] / df[element + '_price'].shift(1) - 1
        ret.index = df['timestamp']

        if element == 'underlying':
            delta_option_portfolio = df_result['call_delta'] * df_result['call_position'] + \
                                     df_result['put_delta'] * df_result['put_position']
            position = np.where(delta_option_portfolio > 0, -1, np.where(delta_option_portfolio < 0, 1, 0))
            position = pd.Series(position)
            position.index = df['timestamp']

        # fees
        fees = np.where(
            (position != position.shift(1)) & (position != 0), fee, np.where(
                (position == position.shift(1)) & (position != position.shift(-1)) & (position.shift(1) != 0),
                fee, 0))

        # evaluate performance
        for i in range(1, len(account)):

            if position[i] != -1:  # long position
                account[i] = account[i-1] * (1 + (ret[i] * position[i]) - fees[i])

            else:  # short position
                if position[i-1] != -1:  # first short
                    collateral[i] = margin * account[i-1]
                else:  # while short
                    collateral[i] = collateral[i-1]

                account[i] = ((1 - margin) * account[i - 1]) * (1 + (ret[i] * position[i]) - fees[i]) + collateral[i]

                # Collateral check
                if (account[i] <= margin*collateral[i]) & (position[i+1] == -1) & (trading_signal[i+1] == 0):
                    position[i+1] = 0
                    fees[i + 1] = 0
                    fees[i] = fee
                    ret[i] = 1 - (margin / account[i - 1]) + 0.00001
                else:
                    if (account[i] <= margin*collateral[i]) & (position[i+1] == -1) & (trading_signal[i+1] == -1):
                        position[i] = 0
                    else:
                        if (account[i] <= margin * collateral[i]) & (position[i + 1] != -1):
                            fees[i] = fee
                            ret[i] = 1 - (margin / account[i - 1]) + 0.00001

                # re-evaluate account
                account[i] = ((1 - margin) * account[i - 1]) * (1 + (ret[i] * position[i]) - fees[i]) + collateral[i]

        # account
        account = pd.Series(account)
        account.index = df['timestamp']

        # Delta
        if element != 'underlying':  # call and put delta
            delta = df['market_delta_' + element]
            delta.index = df['timestamp']
            df_result[element + '_delta'] = delta

        # add results
        df_result[element + '_account'] = account
        df_result[element + '_position'] = position

    return df_result
```

### Strategy optimization: close positions at margin calls and evaluate new positions
The third sub-step consists in optimize the strategy for each instrument. 
To make it clear: once run the function discussed [above](#evaluate-the-call-put-and-underlying-cumulative-returns-positions-delta)
the resulting `pandas.DataFrame` will have the following structure:

* 3 columns for Delta (call, put, underlying)
* 3 columns for Trading Positions (call, put, underlying)
* 3 columns for account value (call, put, underlying)

Thus, it is probable that when an instrument received a Margin Call the others didn't and the trading positions are different.
Therefore, to have a delta-neutral straddle when a position on a specific instrument is closed (opened) it has to be done for all the other instruments too.

The function is `optim_cumret_strategy()` and it does the following:
1. Evaluate initial trading positions given the trading signals.
2. Check if trading positions for all the instruments are the same.
3. If positions are different re-evalute the strategy to have same positions, obtaining the final positions.
4. If postions are the same: final positions = initial positions.

```python
# optimize strategy
def optim_cumret_strategy(df, fee, margin, lag):

    # evaluate initial position
    trading_signal = df['initial_trading_signal']
    initial_position = trading_position(signal=trading_signal, df=df, lag=lag)

    # evaluate strategy
    strategy = cumret_strategy(df=df, position=initial_position, fee=fee, margin=margin)

    # zeros
    zeros = strategy[['call_position', 'put_position', 'underlying_position']].isin([0]).sum()

    if (zeros[0] == zeros[1]) & (zeros[1] == zeros[2]):  # the strategy is optimized
        optim_strategy = strategy
    else:  # optimize strategy
        column = strategy[['call_position', 'put_position', 'underlying_position']].isin([0]).sum().idxmax(axis=1)
        position = strategy[column]
        optim_strategy = cumret_strategy(df=df, position=position, fee=fee, margin=margin)

    return optim_strategy
```

### Dollar performance for Call, Put and, Underlying given the final positions
The fourth sub-step is to evaluate the monetary - dollar - performance of each instrument.
The function is `usd_strategy()`:

```python
# USD strategy
def usd_strategy(df, fee, margin, lag):

    # import modules
    import pandas as pd
    import numpy as np

    # index price
    index_price = df['index_price']
    index_price.index = df['timestamp']

    # optimal strategy
    optim_cumret_strategy_df = optim_cumret_strategy(df=df, fee=fee, margin=margin, lag=lag)
    position = optim_cumret_strategy_df['call_position']

    # call, put, underlying in USD terms
    elements = ['call', 'put', 'underlying']
    usd_df = pd.DataFrame()

    for element in elements:

        # initialize USD instrument
        usd_instrument = np.zeros(len(optim_cumret_strategy_df))
        usd_instrument = pd.Series(usd_instrument)
        usd_instrument.index = df['timestamp']

        # evaluate USD instrument
        for i in range(len(optim_cumret_strategy_df)):

            # Call, Put
            if element != 'underlying':
                if position[i] != 0:
                    usd_instrument.iloc[i] = optim_cumret_strategy_df[element + '_account'].iloc[i] * \
                                        df[element + '_price'].iloc[0] * index_price.iloc[i]
                else:
                    usd_instrument.iloc[i] = usd_instrument.iloc[i - 1]

            # Underlying
            else:
                if position.iloc[i] != 0:
                    usd_instrument.iloc[i] = optim_cumret_strategy_df[element + '_account'].iloc[i] * \
                                        df[element + '_price'].iloc[0]
                else:
                    usd_instrument.iloc[i] = usd_instrument.iloc[i - 1]

        # pd.Series()
        usd_instrument = pd.Series(usd_instrument)
        usd_instrument.index = df['timestamp']

        # add result
        usd_df[element + '_usd'] = usd_instrument
        usd_df[element + '_position'] = optim_cumret_strategy_df[element + '_position']
        if element != 'underlying':
            usd_df[element + '_delta'] = optim_cumret_strategy_df[element + '_delta']

    return usd_df
```

### Delta-Neutral strategy performance and positions
The fifth and last sub-step consists in evaluating the delta-neutral trading performance of a strategy, and the trading positions.
The function is `delta_neutral_dollar_strategy()` and it does the following:

1. Evaluate the Delta Exposure of each instrument.
2. Evaluate the Delta of the portfolio made by a Call Option, and a Put Option.
3. Find the optimal weights for both the underlying and the option portfolio
4. Evaluate the trading performance of the delta-neutral straddle and store the positions.

```python
# Delta Neutral dollar strategy
def delta_neutral_dollar_strategy(df, fee, margin, lag):

    # import modules
    import pandas as pd
    import numpy as np

    # USD strategy
    usd_df = usd_strategy(df=df, fee=fee, margin=margin, lag=lag)

    # delta options
    delta_call_portfolio = usd_df['call_delta'] * usd_df['call_position']
    delta_put_portfolio = usd_df['put_delta'] * usd_df['put_position']
    delta_option_portfolio = delta_call_portfolio + delta_put_portfolio

    # delta underlying
    delta_underlying_portfolio = usd_df['underlying_position']

    # weight
    weight_underlying = (-delta_option_portfolio / (delta_underlying_portfolio-delta_option_portfolio)).fillna(0)
    weight_option = (-delta_underlying_portfolio / (delta_option_portfolio-delta_underlying_portfolio)).fillna(0)

    # portfolio
    option_portfolio = usd_df['call_usd'] + usd_df['put_usd']
    portfolio = pd.Series(np.ones(len(option_portfolio)))
    portfolio.index = df['timestamp']

    # initialize position
    portfolio_position = pd.Series(np.ones(len(option_portfolio)))
    portfolio_position.index = df['timestamp']
    portfolio_position.replace(1, 0, inplace=True)

    # ret
    ret_option = option_portfolio / option_portfolio.shift(1) - 1
    ret_underlying = usd_df['underlying_usd'] / usd_df['underlying_usd'].shift(1) - 1

    ret_option.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
    ret_underlying.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

    # portfolio value and position
    for i in range(len(portfolio)):
        if (weight_underlying.iloc[i] == 0) & (weight_option.iloc[i] == 0):
            portfolio.iloc[i] = portfolio.iloc[i-1]
            portfolio_position.iloc[i] = 0
        else:
            portfolio.iloc[i] = weight_option.iloc[i] * portfolio.iloc[i-1] * (1 + ret_option.iloc[i]) + \
                                weight_underlying.iloc[i] * portfolio.iloc[i-1] * (1 + ret_underlying.iloc[i])
            if portfolio.iloc[i] <= 0:
                portfolio.iloc[i] = portfolio.iloc[i-1]
                portfolio_position.iloc[i] = 0
            else:
                portfolio_position.iloc[i] = 1

    return portfolio, portfolio_position
```

## Back-testing all strategies
The fourth step is to back-test the trading strategy explained [above](#back-test) for all the pairs of Call and Put options together with the underlying, and the cash collateral that are available in the market.
The function is `all_strategies()` and it does the following:

1. Subset the Call options, and the Put options.
2. Evaluate the target implied volatility.
3. Finds the trading signals based on the `quantile of the implied volatility` desired (e.g. bottom/top 10% in this case).
4. Reduce the size of the Call dataset and Put dataset keeping only the variables of interest.
5. Change `0` prices to `1e-08` to avoid `np.Nan` and `np.inf` when evaluating the returns.
6. back-test the strategy for all the pairs and insert the values in a pandas.DataFrame where each column contains the `delta-neutral straddles` value and positions over time.


Note: this function as you can easily imagine required a lot of time. 
At the time when this paper is submitted there are almost 2800 strategy to be back-tested for `BTC` and 4000 for `ETH`. 
Furthermore, consider that data have 5-minutes interval and at the time this paper is submitted there are 182 trading days.
Our local machine works for `5-6` hours per coin to obtain the results.

```python
# evaluate the performance of all the strategies
def all_strategies(lag, coin_df, quantile_iv, coin, fee, margin):
    """

    :param lag: int. lag to enter a position after a trading signal
    :param coin_df: pd.DataFrame from load_data()
    :param quantile_iv: float. 0 < quantile_iv < 0.5
    :param coin: str. 'BTC' or 'ETH'
    :param fee: float. i.e. 0.06
    :param margin: float. i.e. 0.5
    :return: 2 pd.DataFrame() with strategies results and position
    """

    # import modules
    import os
    import numpy as np
    import pandas as pd
    pd.options.mode.chained_assignment = None  # default='warn'
    from datetime import datetime, timedelta
    from tqdm import tqdm

    # source path
    source_path = os.path.abspath(os.getcwd())
    file_path = source_path.replace('deliverables', 'reports/data/' + coin + '/')

    # subset calls and puts
    df_calls = coin_df[coin_df['instrument_name'].str.contains('-C')]
    df_puts = coin_df[coin_df['instrument_name'].str.contains('-P')]

    # remove '-C' and '-P' from instrument_name
    df_calls['instrument_name'] = df_calls['instrument_name'].str.rstrip('-C')
    df_puts['instrument_name'] = df_puts['instrument_name'].str.rstrip('-P')

    # target iVol
    target_iv = df_calls['mark_iv'].median()
    df_calls['difference'] = df_calls['mark_iv'] - target_iv

    # target buy and sell based on target_iv
    buy_signal = df_calls['difference'].quantile(q=0.5 - quantile_iv)
    sell_signal = df_calls['difference'].quantile(q=0.5 + quantile_iv)

    # trading signal
    trading_signal = np.where(
        df_calls['difference'] >= sell_signal, -1,
        (np.where(df_calls['difference'] <= buy_signal, 1, 0)))
    df_calls['initial_trading_signal'] = trading_signal

    # create df_option with variables of interest
    df_calls = df_calls[
        ['timestamp', 'index_price', 'underlying_price', 'instrument_name', 'mark_price', 'mark_iv', 'greeks.delta',
         'strike', 't', 'difference', 'initial_trading_signal', 'underlying_index']]
    df_calls.columns = ['timestamp', 'index_price', 'underlying_price', 'instrument_name', 'call_price',
                        'mark_iv', 'market_delta_call', 'strike', 't',
                        'difference', 'initial_trading_signal', 'underlying_index']
    df_puts = df_puts[
        ['timestamp', 'instrument_name', 'mark_price', 'greeks.delta']]
    df_puts.columns = ['timestamp', 'instrument_name', 'put_price', 'market_delta_put']

    df_option = df_puts.merge(df_calls, how='inner', on=['timestamp', 'instrument_name'])

    # change 0 prices to 1e-08
    df_option.loc[df_option['call_price'] == 0, 'call_price'] = 1e-08
    df_option.loc[df_option['put_price'] == 0, 'put_price'] = 1e-08

    # Datetime index for the back-test pd.DataFrame
    start_date = str(df_option.timestamp.min().round('1d'))
    start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    end_date = str(df_option.timestamp.max().round('1d'))
    end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    min_gap = 5

    full_datetime = [(start_date + timedelta(hours=min_gap*i/60)).strftime("%Y-%m-%d %H:%M:%S")
                     for i in range(int((end_date-start_date).total_seconds() / 60.0 / min_gap))]

    # back-test variables
    flag = False
    options = df_option['instrument_name'].unique()
    pbar = tqdm(total=len(options))

    # loop over each pair of instruments
    for option in options:

        # initialize join_df
        index = full_datetime
        join_df = pd.DataFrame(index, columns=['timestamp'])
        join_df.set_index('timestamp', inplace=True)

        # initialize join_position_df
        join_position_df = join_df

        # subset by instrument and sort values
        df = df_option[df_option['instrument_name'] == option]
        df = df.sort_values('timestamp')
        all_zeros = not np.any(df['initial_trading_signal'])

        # if there is no trading signal: skip
        if all_zeros is True:
            pbar.update(1)

        # if there is at least one trading signal != 0: apply strategy
        else:

            # strategy return and position
            result = delta_neutral_dollar_strategy(df=df, fee=fee, margin=margin, lag=lag)

            # results
            cum_ret_delta_hedged = pd.DataFrame(result[0])
            cum_ret_delta_hedged.columns = [option]

            # position
            df_position = pd.DataFrame(result[1])
            df_position.columns = [option]

            # create df_final with performance of the delta-hedged strategy for each pair
            if not flag:  # first iteration

                # fill df
                join_df.reset_index(inplace=True)
                join_df = join_df.set_index(pd.to_datetime(join_df['timestamp']))

                df_initial = join_df.join(cum_ret_delta_hedged)
                df_initial = df_initial[~df_initial.index.duplicated(keep='first')]

                df_final = df_initial
                df_final = df_final.iloc[:, 1:]

                # fill df_final_position
                join_position_df.reset_index(inplace=True)
                join_position_df = join_position_df.set_index(pd.to_datetime(join_position_df['timestamp']))

                df_initial_position = join_position_df.join(df_position)
                df_initial_position = df_initial_position[~df_initial_position.index.duplicated(keep='first')]

                df_final_position = df_initial_position
                df_final_position = df_final_position.iloc[:, 2:]

                # remove df_initial from the interpreter
                del df_initial
                del df_initial_position
                flag = True

            else:  # from the second iteration

                # concat df
                join_df.reset_index(inplace=True)
                join_df = join_df.set_index(pd.to_datetime(join_df['timestamp']))

                df_second = join_df.join(cum_ret_delta_hedged)
                df_second = df_second.iloc[:, 1]
                df_second = df_second[~df_second.index.duplicated(keep='first')]

                df_final = pd.concat([df_final, df_second], axis=1)

                # concat position_df
                join_position_df.reset_index(inplace=True)
                join_position_df = join_position_df.set_index(pd.to_datetime(join_position_df['timestamp']))

                df_second_position = join_position_df.join(df_position)
                df_second_position = df_second_position.iloc[:, 2]
                df_second_position = df_second_position[~df_second_position.index.duplicated(keep='first')]

                df_final_position = pd.concat([df_final_position, df_second_position], axis=1)

                # remove df_second from the interpreter
                del df_second
                del df_second_position

            # update progress bar
            pbar.update(1)

    # close progress bar
    pbar.close()

    # store portfolio results
    df_final.to_parquet(file_path + 'df_final.parquet')
    df_final_position.to_parquet(file_path + 'df_final_position.parquet')

    # print
    print(coin + ' all strategies result: done')

    return df_final, df_final_position
```

## Quantitative strategy performance
The fifth step consists in evaluating the performance of an equally weighted portfolio with all the strategies back-tested.
The function is `strategy_performance()` and it does the following:

1. Look at the trading opportunities over time
2. Plot the trading opportunities and the portfolio composition
3. Evaluate the trading performance, and the descriptive statistics of each delta-neutral straddle.
4. Back-test a quantitative strategy with daily re-balance consisting in an equally-weighted portfolio of all the strategies.
5. Evaluate daily returns and cumulative returns.

```python
# quantitative strategy
def strategy_performance(coin, transaction_cost):

    # import modules
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # directories
    source_path = os.path.abspath(os.getcwd())
    portfolios_path = source_path.replace('deliverables', 'reports/data/' + coin + '/df_final.parquet')
    positions_path = source_path.replace('deliverables', 'reports/data/' + coin + '/df_final_position.parquet')

    # load data
    df_portfolios = pd.read_parquet(portfolios_path)
    df_position = pd.read_parquet(positions_path)

    # remove columns of non interest
    if coin == 'BTC':
        df_portfolios = df_portfolios.drop('BTC-31DEC21-68000', 1)
        df_position = df_position.drop('BTC-31DEC21-68000', 1)
        color = 'C0'
    else:
        color = 'C1'

    # trading opportunities
    trading_opportunities = df_position.sum(axis=1)
    trading_opportunities_perc = trading_opportunities / df_position.count(axis=1)
    trading_opportunities_perc = trading_opportunities_perc.replace(np.nan, 0)

    # plot trading opportunities
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=coin + ' trading opportunities', x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Trading Opportunities', ha='center', va='center', rotation='vertical')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(trading_opportunities, color=color)
    ax.hlines(y=trading_opportunities.mean(), xmin=trading_opportunities.index[0], xmax=trading_opportunities.index[-1],
              linestyle='--', color='black', lw=.4)
    plt.margins(x=0)
    plt.margins(y=0)
    fig.legend(['Trading opportunities', 'Mean:' + str(round(trading_opportunities.mean(), 2))],
               bbox_to_anchor=(.5, 0.03), loc="lower center",
               bbox_transform=fig.transFigure, ncol=2, frameon=False)
    # save plot
    file_path = source_path.replace('deliverables',
                                    'reports/data/' + coin + '/trading_opportunities.pdf')
    plt.savefig(file_path, dpi=160)
    plt.close()

    # plot trading opportunities %
    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig.text(s=coin + ' trading opportunities as ratio of trading  available options', x=0.5, y=0.95, fontsize=20,
             ha='center', va='center')
    fig.text(0.06, 0.5, 'Trading Opportunities', ha='center', va='center', rotation='vertical')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(trading_opportunities_perc[1:-1], color=color)
    ax.hlines(y=trading_opportunities_perc[1:-1].mean(), xmin=trading_opportunities_perc[1:-1].index[0],
              xmax=trading_opportunities_perc[1:-1].index[-1], linestyle='--', color='black', lw=.4)
    plt.margins(x=0)
    plt.margins(y=0)
    fig.legend(['Trading opportunities', 'Mean:' + str(round(trading_opportunities_perc[1:-1].mean(), 2))],
               bbox_to_anchor=(.5, 0.03), loc="lower center",
               bbox_transform=fig.transFigure, ncol=2, frameon=False)
    # save plot
    file_path = source_path.replace('deliverables',
                                    'reports/data/' + coin + '/trading_opportunities_perc.pdf')
    plt.savefig(file_path, dpi=160)
    plt.close()

    # performance
    performance = (df_portfolios.ffill(axis=0).iloc[-1, :] - 1)
    prob_succ = sum(performance > 0) / len(performance)
    upside_potential = performance[performance > 0].mean()
    downside_risk = performance[performance < 0].mean()
    summary_performance = performance.describe()
    summary_performance.T['skew'] = performance.skew()
    summary_performance.T['kurt'] = performance.kurt()
    summary_performance.T['prob'] = prob_succ
    summary_performance.T['upside'] = upside_potential
    summary_performance.T['downside'] = downside_risk
    print(coin + ' summary performance all strategies: done')
    file_path = source_path.replace('deliverables', 'reports/data/' + coin + '/summary_performance_all.csv')
    summary_performance.to_csv(file_path)

    # quantitative strategy
    def quant_strategy(transaction_cost):

        # ffill and bfill NA to calcualte returns
        df = df_portfolios.fillna(method='ffill')
        df = df.fillna(method='bfill')

        # evaluate log-returns
        log_ret = np.log(df / df.shift(1))
        log_ret = log_ret * df_position

        # daily rebalance
        log_ret = log_ret.resample('D').sum()

        # account for transaction costs
        log_ret[log_ret != 0] = log_ret - transaction_cost

        # equally weighted portfolio
        log_ret = log_ret.div(log_ret[log_ret != 0].count(axis=1), axis='index')
        ret = log_ret.sum(axis=1)

        # summary
        summary = ret.describe()
        summary.T['skew'] = ret.skew()
        summary.T['kurt'] = ret.kurt()
        file_path = source_path.replace('deliverables', 'reports/data/' + coin + '/summary_final_ret.csv')
        summary.to_csv(file_path)
        print(coin + ' summary statistics final strategy returns: done')

        return ret

    # strategy performance
    ret = quant_strategy(transaction_cost=transaction_cost)
    cum_ret = (1 + ret).cumprod()

    return ret, cum_ret
```

## Visualize portfolios performance
The sixth and final step consists in visualize the trading performance.
The function is `plot_strategy()`:

```python
# plot of the cumulative performance and returns
def plot_strategy(coin, ret, cum_ret):

    # import modules
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    source_path = os.path.abspath(os.getcwd())
    plot_path = source_path.replace('deliverables', 'reports/data/' + coin + '/' + coin + '_stategy_performance.pdf')

    cum_ret.loc[pd.to_datetime('2021-03-08 00:00:00')] = 1
    cum_ret = cum_ret.sort_index()

    plt.rcParams['font.family'] = 'serif'  # set font family: serif
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    fig.text(s=coin + ' Equally Weighted Portfolio of Delta-Neutral Straddles', x=0.5, y=0.95, fontsize=20, ha='center', va='center')
    if coin == 'BTC':
        color = 'C0'
    else:
        color = 'C1'

    ax[0].plot(cum_ret, color=color)
    ax[0].hlines(y=1, xmin=cum_ret.index[0], xmax=cum_ret.index[-1], linestyle='--', color='black', lw=0.8)
    ax[1].plot(ret, color=color)
    ax[1].hlines(y=0, xmin=cum_ret.index[0], xmax=cum_ret.index[-1], linestyle='--', color='black', lw=0.8)

    # margins
    ax[0].margins(x=0)
    ax[0].margins(y=0.1)
    ax[1].margins(x=0)
    ax[1].margins(y=0.1)

    # remove spines
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    # remove lables
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # name labels
    fig.text(0.04, 0.7, 'Cumulative log-returns', ha='center', va='center', rotation='vertical')
    fig.text(0.04, 0.3, 'Daily log-returns', ha='center', va='center', rotation='vertical')

    # savefig
    plt.savefig(plot_path, dpi=160)
    plt.clf()

    return print(coin + ' performance plot: done')
```