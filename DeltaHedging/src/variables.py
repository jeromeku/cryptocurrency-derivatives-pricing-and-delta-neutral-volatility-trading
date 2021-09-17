""" Main Variables to try different set up """

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
