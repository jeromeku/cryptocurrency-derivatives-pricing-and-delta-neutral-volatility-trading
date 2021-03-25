""" Data pre-processing """

"""
Author: Matteo Bottacini, matteo.bottacini@usi.ch
Last Update: March 19, 2021
"""

# import modules
import pandas as pd
import numpy as np
from scipy import interpolate
import statsmodels.api as sm
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
