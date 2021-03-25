# Download real time Cryptocurrency Option Data from Deribit and store entries in a SQLite3 remote DataBase

#### Last Update March 19, 2021 ####
#### Matteo Bottacini, [matteo.bottacini@usi.ch](mailto:matteo.bottacini@usi.ch) ####

This is a setup script to automate the downloading of option data available on [Deribit](https://www.deribit.com) via a public API.
The following `Python3` script is run on a remote Ubuntu server, and the new entries are then stored in a `SQLite3` remote DataBase.

Deribit’s api provides access to option data traded on the exchange. 

Acquiring the data is a two-step process:
1. Get a list of all active options.
2. Loop through the list of options to retrieve the relevant data.


## Table of contents
1. [Get instrument name and settlement](#get-instrument-name-and-settlement)
2. [Get option data](#get-option-data)
3. [Download data](#download-data)
4. [Prepare data for SQLite3 remote DataBase](#prepare-data-for-sqlite3-remote-database)
5. [Connect with SQLite3 remote DataBase and upload new data](#connect-with-sqlite3-remote-database-and-upload-new-data)
6. [Print Statements for logs](#print-statements-for-logs)
7. [Supported versions](#supported-versions)
8. [References](#references)


## Get instrument name and settlement
To get a list of all active options both for BTC and ETH we need to use the query `/get_instruments` and then set the coin : either `BTC` for Bitcoin Options and `ETH` for Ethereum Options.

The function make uses of `json`, `requests` and `pandas` modules.

Here is the code that you can find in [`../0_server/src/option-data-download.py`](../src/option-data-download.py):

```python
# import modules
import json
import requests
import pandas as pd


# get a list of all active options and settlement by coin
def get_option_name_and_settlement(coin):
    """
    :param coin: crypto-currency coin name ('BTC', 'ETH')
    :return: 2 lists:
                        1.  list of traded options for the selected coin;
                        2.  list of settlement period for the selected coin.
    """

    # requests public API
    r = requests.get("https://test.deribit.com/api/v2/public/get_instruments?currency=" + coin + "&kind=option")
    result = json.loads(r.text)

    # get option name
    name = pd.json_normalize(result['result'])['instrument_name']
    name = list(name)

    # get option settlement period
    settlement_period = pd.json_normalize(result['result'])['settlement_period']
    settlement_period = list(settlement_period)

    return name, settlement_period
```

## Get option data
The second step consists in loop through the list options retrieved with `get_option_name_and_settlement(coin)`.

The function do the following:
1. get the list of option instrument names.
2. get all the availble data for the selected options (33 features).
3. remove useless features (`state`, `delivery_price`) and add the feature `settlement_period`.
4. returns pandas DataFrame with 32 columns (features) and N rows (traded active options).

```python
# import modules
import json
import requests
import pandas as pd
from tqdm import tqdm

# get pandas dataframe of relevant option data by coin
def get_option_data(coin):
    """
    :param coin: crypto-currency coin name ('BTC', 'ETH')
    :return: pandas data frame with all option data for a given coin
    """

    # get option name and settlement
    coin_name = get_option_name_and_settlement(coin)[0]
    settlement_period = get_option_name_and_settlement(coin)[1]

    # initialize data frame
    coin_df = []

    # initialize progress bar
    pbar = tqdm(total=len(coin_name))

    # loop to download data for each Option Name
    for i in range(len(coin_name)):
        # download option data -- requests and convert json to pandas
        r = requests.get('https://test.deribit.com/api/v2/public/get_order_book?instrument_name=' + coin_name[i])
        result = json.loads(r.text)
        df = pd.json_normalize(result['result'])

        # add settlement period
        df['settlement_period'] = settlement_period[i]

        # append data to data frame
        coin_df.append(df)

        # update progress bar
        pbar.update(1)

    # finalize data frame
    coin_df = pd.concat(coin_df)

    # remove useless columns from coin_df
    columns = ['state', 'estimated_delivery_price']
    coin_df.drop(columns, inplace=True, axis=1)

    # close the progress bar
    pbar.close()

    return coin_df
```

## Download data
Now data are downloaded for both coins: `BTC` and `ETH`.

```python
# download data -- BTC and ETH Options
btc_data = get_option_data('BTC')
eth_data = get_option_data('ETH')
```

## Prepare data for SQLite3 remote DataBase
All the entries are then converted to`strings` with the `.astype(str)` function to enable data to be filled in the SQLite3 remote database.

```python
# transform each element of the data frames into strings for sqlite3
btc_data = btc_data.astype(str)
eth_data = eth_data.astype(str)
```

## Connect with SQLite3 remote DataBase and upload new data
The first step is to create a connection with the remote Ubuntu server using the module `sqlite3` and the function `sqlite3.connect()`.
The function require to insert the remote path in the following way `/remote/path/database.db`. 
If the database already exists it is just opened, otherwise a new one is created.

The second step is convert the `pandas.DataFrame` into an `SQL` file and this is done with the function `pandas.DataFrame.to_sql()`.
Important settings are then `if_exists=append` and `index=False` that enables to append data to the ones already collected and since there is no indexing there is no risk to occure in double-indexing problem.

```python
# import modules
import sqlite3
import pandas as pd

# connect to the SQLite3 database -- option_data.db
conn = sqlite3.connect('/home/bottama/deliverables/option-data-5min.db')

# create/update BTC and ETH tables in the database
btc_data.to_sql(name='btc_option_data', con=conn, if_exists='append', chunksize=None, index=False)
eth_data.to_sql(name='eth_option_data', con=conn, if_exists='append', chunksize=None, index=False)
```

## Print statements for logs
After the connection is executed and files are appended are also printed different statements.

```python
import datetime
# print data and time for log
print('Date and time: ' +  datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ' , format: dd/mm/yyyy hh:mm:ss')

# after the connection is established
print('Connection established with SQLite3 Server: option-data-5min.db')

# after BTC data are appended
print('BTC data appended')

# after ETH data are appended
print('ETH data appended')
```

## Supported versions
This setup script has been tested against Ubuntu 20.04.2 LTS (GNU/Linux 5.4.0-56-generic x86_64), SQLite version 3.31.1, Python 3.8.5 and Deribit API v2.0.1.

## References
[Deribit API v2](https://docs.deribit.com/?python#deribit-api-v2-0-1)

[pandas.DataFrame.to_sql](#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html)

