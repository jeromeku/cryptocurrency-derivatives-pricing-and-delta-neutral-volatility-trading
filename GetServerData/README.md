# Get Remote Server data

#### Last Update March 24, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

The second step is to get option data from the remote server and to store them locally to perform all the tasks
In these scripts are described both the codes and the intuition behind them.


Folder structure:
~~~~
    ../GetServerData/
        data/
            btc_option_data.ftr
            eth_option_data.ftr
        deliverables/
            get-server-data.py
        src/
            utils.py
            credentials.py
        reports/
            walk-thorugh-the-code.md
            images/
              event-log.png
              storage.png
              read_write_all.png
              read_write.png
~~~~


# Reports

1. [Walk through the code](reports/walk-through-the-code.md)


# Instruction
* In [`../GetServerData/src/credentials.py`](../GetServerData/src/credentials.py) are written the personal credentials to SSH into the remote server.
* In [`../GetServerData/src/utils.py`](../GetServerData/src/utils.py) are coded a set of functions to connect and download data from the remote server.
* In [`../GetServerData/deliverables/get-server-data.py`](../GetServerData/deliverables/get-server-data.py) is coded the script to get option data: this is the only one that has to be run.
* In [`../GetServerData/data`](../GetServerData/data) are stored the the `.ftr` files containing the BTC and the ETH options data from the remote server.
    * BTC data: [`../GetServerData/data/btc_option_data.ftr`](../GetServerData/data/btc_option_data.ftr)
    * ETH data: [`../GetServerData/data/eth_option_data.ftr`](../GetServerData/data/eth_option_data.ftr)
