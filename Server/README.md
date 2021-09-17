# Server

#### Last Update March 20, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

The first step is to setup the remote Ubuntu server to download and store the financial data needed.
In these scripts are described both the codes and the intuition behind them.


Folder structure:
~~~~
    ../server/
        src/
            option-data-download.py
            move-files.sh
            server-setup.sh
        reports/
            report-ubuntu-server-setup.md
            report-data-download.md
            images/
                database-output.png
                log-sample.png
~~~~


# Reports

1. [Bash Setup script for Ubuntu servers](../Server/reports/report-ubuntu-server-setup.md)
2. [Download Option data from Deribit](../Server/reports/report-data-download.md)

# Instruction
* In [`../Server/src/option-data-download.py`](../Server/src/option-data-download.py) option data of BTC and ETH are collected from Deribit via the public API and `Python3`.
* In [`../Server/src/move-files.sh`](../Server/src/move-files.sh) is coded a `SHELL` script to move zipped files from the initial directory to the destination one.
* In [`../Server/src/server.setup.sh`](../Server/src/server-setup.sh) is coded a `SHELL` script to setup the Ubuntu remote server.