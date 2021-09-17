""" Delta-Hedged Trading strategy based on Median-reverting implied Volatility """

"""
Author: Matteo Bottacini, matteo.bottacini@usi.ch
Last update: June 23, 2021
"""

# import modules
from DeltaHedging.src.utils import *
from DeltaHedging.src.variables import *


# load data
btc_data = load_data(coin='btc', cwd='DeltaHedging/deliverables')
eth_data = load_data(coin='eth', cwd='DeltaHedging/deliverables')

# evaluate all strategies performance
btc_all_strategies = all_strategies(lag=lag, coin_df=btc_data, quantile_iv=quantile_iv, coin='BTC',
                                    fee=fee, margin=margin)
eth_all_strategies = all_strategies(lag=lag, coin_df=eth_data, quantile_iv=quantile_iv, coin='ETH',
                                    fee=fee, margin=margin)

# quantitative strategy performance
btc_strategy = strategy_performance(coin='BTC', transaction_cost=transaction_cost)
eth_strategy = strategy_performance(coin='ETH', transaction_cost=transaction_cost)

# plot final results
plot_strategy(coin='BTC', ret=btc_strategy[0], cum_ret=btc_strategy[1])
plot_strategy(coin='ETH', ret=eth_strategy[0], cum_ret=eth_strategy[1])
