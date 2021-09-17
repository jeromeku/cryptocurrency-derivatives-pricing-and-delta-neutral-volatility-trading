# Delta-Neutral Volatility Trading

#### Last Update September 17, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

The fifth and final step is to back-test and analyze the trading performance of a delta-neutral strategy.
In these scripts are described, the codes, the intuition behind them and, the results obtained.


Folder structure:
~~~~
    ../DeltaHedging/
        README.md
        deliverables/
            run-delta-hedged-stratetgy.py
        src/
            backtest.py
            utis.py
            varibales.py
        reports/
            walk-thorugh-the-code.md
            detla-neutral-trading-strategy.md
            data/
                BTC/
                    df_final.parquet
                    df_final_postion.parquet
                    trading_opportunities.png
                    trading_opportunities_perc.png
                    strategy_performance.png
                    summary_final_ret.csv
                    summary_performance_all.csv
                ETH/
                    df_final.parquet
                    df_final_postion.parquet
                    trading_opportunities.png
                    trading_opportunities_perc.png
                    strategy_performance.png
                    summary_final_ret.csv
                    summary_performance_all.csv           
~~~~


# Reports

1. [Walk trough the code](../DeltaHedging/reports/walk-through-the-code.md)
2. [Delta-Neutral Volatility trading strategy](../DeltaHedging/reports/delta-neutral-trading-strategy.md)

# Instruction
* [`../DeltaHedging/deliverables/run-delta-hedged-strategy.py`](../DeltaHedging/deliverables/run-delta-hedged-strategy.py) is the script to run to perform the analysis.
* In [`../DeltaHedging/src/utils.py`](../DeltaHedging/src/utils.py) and in [`../DeltaHedging/src/backtest.py`](../DeltaHedging/src/backtest.py) are the set of functions to perform the backtest.
* In [`../DeltaHedging/src/variables.py`](../DeltaHedging/src/variables.py) are the main variables that you can change.