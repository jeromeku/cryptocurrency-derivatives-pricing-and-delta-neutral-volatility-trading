""" Run Spot Derivatives Analysis """

"""
Author: Matteo Bottacini, matteo.bottacini@usi.ch
last update: July 7, 2021
"""

# import modules
from SpotAnalysis.src.utils import *

# create environment
create_env(local_folder='deliverables')

# preprocessing data
btc_df = data_preprocessing(coin='BTC')
eth_df = data_preprocessing(coin='ETH')

# iVol Smile plots - 3 months to maturity
iv_smile(coin_df=btc_df, coin='BTC', time_to_maturity=90, cwd='deliverables')
iv_smile(coin_df=eth_df, coin='ETH', time_to_maturity=90, cwd='deliverables')

# iVol Surface and ATM term structure
implied_vol(coin_df=btc_df, coin='BTC', cwd='deliverables')
implied_vol(coin_df=eth_df, coin='ETH', cwd='deliverables')

# Greeks Surface and ATM term structure
greeks(coin_df=btc_df, coin='BTC', cwd='deliverables')
greeks(coin_df=eth_df, coin='ETH', cwd='deliverables')

# iVol Delta Surface
iv_delta_surface(coin_df=btc_df, coin='BTC', cwd='deliverables')
iv_delta_surface(coin_df=eth_df, coin='ETH', cwd='deliverables')

