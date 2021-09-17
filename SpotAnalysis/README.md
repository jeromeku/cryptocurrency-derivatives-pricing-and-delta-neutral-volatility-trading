# Spot Derivatives Analysis

#### Last Update September 16, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

The third step is to analyze the current derivatives market situation.
In these scripts are described, the codes, the intuition behind them and, the results obtained.


Folder structure:
~~~~
    ../SpotAnalysis/
        README.md
        deliverables/
            run-spot-analysis.py
        src/
            utis.py
        reports/
            walk-through-the-code.md
            spot-derivatives-analysis.md
            images/
                BTC/
                    greeks/
                        atm_term_structure/
                            ...
                        surface/
                            ...
                    volatility/
                        cubic_interpolation/
                            ...
                        linear_interpolation/
                            ...
                        nearest_interpolation/
                            ...
                        volatility-smile.png
                        iv-delta-surface.png
                ETH/
                    greeks/
                        atm_term_structure/
                            ...
                        surface/
                            ...
                    volatility/
                        cubic_interpolation/
                            ...
                        linear_interpolation/
                            ...
                        nearest_interpolation/
                            ...
                        volatility-smile.png
                        iv-delta-surface.png                        

~~~~


# Reports

1. [Walk trough the code](../SpotAnalysis/reports/walk-through-the-code.md)
2. [Spot Derivatives Analysis](../SpotAnalysis/reports/spot-derivatives-analysis.md)

# Instruction
* [`../SpotAnalysis/deliverables/run-spot-analysis.py`](../SpotAnalysis/deliverables/run-spot-analysis.py) is the script to run to perform the analysis.
* In [`../SpotAnalysis/src/utils.py`](../SpotAnalysis/src/utils.py) are the set of functions to perform the tasks, the function are explained in [Walk trough the code](../SpotAnalysis/reports/walk-through-the-code.md).