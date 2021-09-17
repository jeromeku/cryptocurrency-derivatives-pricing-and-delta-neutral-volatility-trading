""" Time series analysis: iVol analysis """

"""
Author: Matteo Bottacini, matteo.bottacini@usi.ch
last update: July 7, 2021
"""

# import modules
from TimeSeriesAnalysis.src.utils import *

# main variables
local_folder = 'deliverables'
min_time_interval = 5
max_time_interval = 720
step = 5
atm_skew = 1
itm_skew = 1.75
otm_skew = 0.25

# create environment
create_env(local_folder=local_folder,
           min_time_interval=min_time_interval, max_time_interval=max_time_interval, step=step)

# load data
btc_data = load_data(coin='btc', cwd='TimeSeriesAnalysis/desliverables')
eth_data = load_data(coin='eth', cwd='TimeSeriesAnalysis/deliverables')

# index price analytics
index = underlying_analytics(local_folder=local_folder, btc_data=btc_data, eth_data=eth_data)

# iVol distribution analytics
iv_distribution(coin_df=btc_data, coin='BTC', local_folder=local_folder)
iv_distribution(coin_df=eth_data, coin='ETH', local_folder=local_folder)

# model calibration
model_calibration(coin_df=btc_data, min_time_interval=min_time_interval, max_time_interval=max_time_interval,
                  step=step, atm_skew=atm_skew, itm_skew=itm_skew, otm_skew=otm_skew,
                  local_folder='deliverables', coin='BTC')
model_calibration(coin_df=eth_data, min_time_interval=min_time_interval, max_time_interval=max_time_interval,
                  step=step, atm_skew=atm_skew, itm_skew=itm_skew, otm_skew=otm_skew,
                  local_folder='deliverables', coin='ETH')









