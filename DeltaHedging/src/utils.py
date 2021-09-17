from DeltaHedging.src.backtest import *


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