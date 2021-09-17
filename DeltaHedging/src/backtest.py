""" Set of Functions to perform the back-test """


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
