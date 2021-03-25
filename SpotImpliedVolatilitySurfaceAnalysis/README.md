# Spot Implied Volatility Surface Analysis

#### Last Update March 23, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

The second step is to analyse current option data and visualize the implied volatility structure.
In these scripts are described both the codes and the intuition behind them.


Folder structure:
~~~~
    ../SpotImpliedVolatilitySurfaceAnalysis/
        deliverables/
            run-spot-IV-analysis.py
        src/
            utils.py
            preprocessing.py
        reports/
            src--walk-through-the-code.md
            report-spot-iv-analaysis.md
            images/
                BTC-IV-surface.png
                ETH-IV-surface.png
                atm_adj_iv.png
                moneyness_openinterest.png
                WLS-BTC.png
                WLS-ETH.png
                BTC Simple Paratmetrize Model-IV-surface.png
                ETH Simple Paratmetrize Model-IV-surface.png
                fitted_iv_moneyness.png
~~~~


# Reports

1. [src: walk though the code](../SpotImpliedVolatilitySurfaceAnalysis/reports/src--walk-though-the-code.md)
2. [Spot Implied Volatility Surface Analysis ](reports/report-spot-iv-analaysis.md)

# Instruction
* In [`../SpotImpliedVolatilitySurfaceAnalysis/src/utils.py`](../SpotImpliedVolatilitySurfaceAnalysis/src/utils.py) are coded a set of functions to retrieve current option data of all active options from [Deribit](https://www.deribit.com), to produce the plots and to move `.png` files to their [destination directory](../SpotImpliedVolatilitySurfaceAnalysis/reports/images/).
* In [`../SpotImpliedVolatilitySurfaceAnalysis/src/preprocessing.py`](../SpotImpliedVolatilitySurfaceAnalysis/src/preprocessing.py) are coded a set of functions to process and analyse option data.
* In [`../SpotImpliedVolatilitySurfaceAnalysis/deliverables/run-spot-IV-analysis.py`](../SpotImpliedVolatilitySurfaceAnalysis/deliverables/run-spot-IV-analysis.py) is coded the script to perform the whole analysis: you need to run just this code.