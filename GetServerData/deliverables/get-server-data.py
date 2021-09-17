""" Get Option data from remote server """

"""
Author: Matteo Bottacini, matteo.bottacini@usi.ch
last update: March 24, 2021
"""

# import modules
from GetServerData.src.credentials import *
from GetServerData.src.utils import *

# pull option data from the remote ubuntu server
get_server_data(host, port, username, password, 'deliverables')
