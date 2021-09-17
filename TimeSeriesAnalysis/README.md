# Time-Series Derivatives Analysis

#### Last Update September 16, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

The fourth step is to analyze the time-series of the derivatives
In these scripts are described, the codes, the intuition behind them and, the results obtained.


Folder structure:
~~~~
    ../SpotAnalysis/
        README.md
        deliverables/
            run-time-series-analysis.py
        src/
            utis.py
        reports/
            walk-though-the-code.md
            time-series-derivatives-analysis.md
            BTC/
                ...
                720_min_calibration/
                    data/
                        BTC_atm_iv.parquet
                        BTC_parameters.parquet
                        BTC_skew.parquet
                    images/
                        BTC_atm_iv.pdf
                        BTC_atm_iv_term_structure.pdf
                        BTC_parameters.pdf
                        BTC_skew.pdf
                    tables/
                        BTC_atm_iv.parquet
                        BTC_parameters.parquet
                        BTC_skew.parquet
                ...
            ETH/
                ...
                720_min_calibration/
                    data/
                        ETH_atm_iv.parquet
                        ETH_parameters.parquet
                        ETH_skew.parquet
                    images/
                        ETH_atm_iv.pdf
                        ETH_atm_iv_term_structure.pdf
                        ETH_parameters.pdf
                        ETH_skew.pdf
                    tables/
                        ETH_atm_iv.parquet
                        ETH_parameters.parquet
                        ETH_skew.parquet
                ...
            images/
            summary/
            
~~~~


# Reports

1. [Walk trough the code](../TimeSeriesAnalysis/reports/walk-through-the-code.md)
2. [Time-Series Derivatives Analysis](../TimeSeriesAnalysis/reports/time-series-derivatives-analysis.md)

# Instruction
* [`../TimeSeriesAnalysis/deliverables/run-time-series-analysis.py`](../TimeSeriesAnalysis/deliverables/run-time-series-analysis.py) is the script to run to perform the analysis.
* In [`../TimeSeriesAnalysis/src/utils.py`](../TimeSeriesAnalysis/src/utils.py) are the set of functions to perform the tasks, the function are explained in [Walk trough the code](../TimeSeriesAnalysis/reports/walk-through-the-code.md).