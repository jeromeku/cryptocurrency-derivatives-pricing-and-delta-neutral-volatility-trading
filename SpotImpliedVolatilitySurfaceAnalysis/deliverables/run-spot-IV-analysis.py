""" Run spot Implied Volatility analysis """

"""
Author: Matteo Bottacini, matteo.bottacini@usi.ch
last update: March 20, 2021
"""

# import modules
from SpotImpliedVolatilitySurfaceAnalysis.src.utils import *
from SpotImpliedVolatilitySurfaceAnalysis.src.preprocessing import *

# set font family: serif
plt.rcParams['font.family'] = 'serif'

# get current data
print('Get BTC options data')
df_btc = data_preprocessing('BTC')

print('Get ETH options data')
df_eth = data_preprocessing('ETH')

# interpolate Implied Volatility using a cubic spline
spline_result_btc = cubic_spline_interpolation(df=df_btc)
spline_result_eth = cubic_spline_interpolation(df=df_eth)


# Plot Implied Volatility Surface
# BTC
plot_iv_surf(spline_result_btc[1],
             spline_result_btc[2],
             spline_result_btc[3],
             'BTC',
             spline_result_btc[0].x,
             spline_result_btc[0].y,
             spline_result_btc[0].z,
             'Observed IV')
print('BTC Implied volatility surface: DONE')

# ETH
plot_iv_surf(spline_result_eth[1],
             spline_result_eth[2],
             spline_result_eth[3],
             'ETH',
             spline_result_eth[0].x,
             spline_result_eth[0].y,
             spline_result_eth[0].z,
             'Observed IV')
print('ETH Implied volatility surface: DONE')

# At-the-Money (ATM) Adjusted Volatility
# BTC
df_adj_iv_btc = atm_adjusted_IV_data(df_btc,
                                     spline_result_btc[0],
                                     spline_result_btc[1],
                                     spline_result_btc[2],
                                     spline_result_btc[4],
                                     spline_result_btc[5])

# ETH
df_adj_iv_eth = atm_adjusted_IV_data(df_eth,
                                     spline_result_eth[0],
                                     spline_result_eth[1],
                                     spline_result_eth[2],
                                     spline_result_eth[4],
                                     spline_result_eth[5])

# plot
plot_atm_adj_iv(df_adj_iv_btc, df_adj_iv_eth)
print('ATM adjusted implied volatility: DONE')

# Observation weight by moneyness and open Interest
moneyness_openinterest(df_btc, df_eth)
print('Observation weight by moneyness and open interest: DONE')


# Weighted Linear Regression
# BTC
btc_wls_data = weighted_lr_data(df_btc)
btc_model = sm.WLS(btc_wls_data[0], btc_wls_data[1], btc_wls_data[2]).fit()
save_wls_model_summary(model=btc_model, coin='BTC')
print('BTC WLS linear regression model summary: stored')

# ETH
eth_wls_data = weighted_lr_data(df_eth)
eth_model = sm.WLS(eth_wls_data[0], eth_wls_data[1], eth_wls_data[2]).fit()
save_wls_model_summary(model=eth_model, coin='ETH')
print('ETH WLS linear regression model summary: stored')

# Simple Parametrized Surface Model
# BTC
btc_parametrized_surface_result = simple_parametrized_surface_model(df=df_btc,model=btc_model,X=btc_wls_data[1])
print('BTC simple parametrized surface model: DONE')

# ETH
eth_parametrized_surface_result = simple_parametrized_surface_model(df=df_eth,model=eth_model,X=eth_wls_data[1])
print('ETH simple parametrized surface model: DONE')

# plot new IV surface
# BTC
plot_iv_surf(x=btc_parametrized_surface_result[0],
             y=btc_parametrized_surface_result[1],
             z=btc_parametrized_surface_result[2],
             coin='BTC Simple Parametrized Model',
             x2=btc_parametrized_surface_result[3]['x'],
             y2=btc_parametrized_surface_result[3]['y'],
             z2=btc_parametrized_surface_result[3]['z'],
             label='Observed IV')

# ETH
plot_iv_surf(x=eth_parametrized_surface_result[0],
             y=eth_parametrized_surface_result[1],
             z=eth_parametrized_surface_result[2],
             coin='ETH Simple Parametrized Model',
             x2=btc_parametrized_surface_result[3]['x'],
             y2=btc_parametrized_surface_result[3]['y'],
             z2=btc_parametrized_surface_result[3]['z'],
             label='Observed IV')

btc_iv_df_fit = pd.DataFrame(btc_parametrized_surface_result[2],
                             index=np.linspace(5, np.max(btc_parametrized_surface_result[3]['y']), 100),
                             columns=np.linspace(.95, 1.05, 99))
eth_iv_df_fit = pd.DataFrame(eth_parametrized_surface_result[2],
                             index=np.linspace(5, np.max(eth_parametrized_surface_result[3]['y']), 100),
                             columns=np.linspace(.95, 1.05, 99))

# fitted implied volatility by moneyness
fitted_iv_moneyness(btc_iv_df_fit, eth_iv_df_fit)
print('Fitted Implied volatility by moneyness: DONE')

# move .png files to ../reports/images
move_files(cwd='deliverables', dwd='reports/images', endswith='.png')
print('.png files are moved to ../reports/images')
