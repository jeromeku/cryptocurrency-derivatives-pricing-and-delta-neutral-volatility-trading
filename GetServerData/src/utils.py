# import modules
import os
import pandas as pd
import paramiko
import shutil
import zipfile
from scp import SCPClient
import pyarrow.feather as feather


# create the directories and sub-directories needed to get data
def create_env(local_folder):

    # source path
    source_path = os.path.abspath(os.getcwd())

    # create /zip_files
    destination_path = source_path.replace(local_folder, 'zip_files')
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    # create sub-directory: ../zip_files/btc_option_data
    sub_directory = destination_path + '/btc_option_data'
    if not os.path.exists(sub_directory):
        os.mkdir(sub_directory)

    # create sub-directory: ../zip_files/eth_option_data
    sub_directory = destination_path + '/eth_option_data'
    if not os.path.exists(sub_directory):
        os.mkdir(sub_directory)

    # create /csv
    destination_path = destination_path.replace('zip_files', 'data')
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    return print('Directory created: ../zip_files , ../zip_files/btc_option_data , ../zip_files/eth_option_data , '
                 '../data')


# ssh connect into remote server
def ssh_remote_server(host, port, username, password, local_folder):

    # SSH into the server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, port=port, username=username, password=password)
    print('Connected to the server')

    # scp bitcoin option data in ../zip_files/btc_option_data/
    source_path = os.path.abspath(os.getcwd())
    local_path = source_path.replace(local_folder, 'zip_files/btc_option_data/')

    with SCPClient(ssh.get_transport(), sanitize=lambda x: x) as scp:
        scp.get(remote_path='/home/bottama/zip_files/btc_option_data/*.zip',
                local_path=local_path)
    print('btc option data: collected')

    # scp ethereum option data in ../zip_files/eth_option_data/
    local_path = source_path.replace(local_folder, 'zip_files/eth_option_data/')
    with SCPClient(ssh.get_transport(), sanitize=lambda x: x) as scp:
        scp.get(remote_path='/home/bottama/zip_files/eth_option_data/*.zip',
                local_path=local_path)
        print('eth option data: collected')

    # close connection
    scp.close()


# unzip & convert data to a pickle file and clean the environment
def clean_data_and_env(local_folder):

    # source path
    source_path = os.path.abspath(os.getcwd())

    # unzip bitcoin data
    dir_name = source_path.replace(local_folder, 'zip_files/btc_option_data')
    extension = ".zip"

    # change directory from working dir to dir with files
    os.chdir(dir_name)

    btc_data_df = pd.DataFrame()
    flag = False

    # loop through items in dir
    print(str(len(os.listdir(dir_name))) + ' days of data are available')
    for item in os.listdir(dir_name):

        # check for ".zip" extension
        if item.endswith(extension):

            # get full path of files
            file_name = os.path.abspath(item)

            # create zipfile object
            zip_ref = zipfile.ZipFile(file_name)

            # extract file to dir
            zip_ref.extractall(dir_name)

            # convert to pandas DataFrame
            local_path = dir_name + '/csv_files/btc_option_data.csv'
            df = pd.read_csv(local_path, header=0)

            if not flag:
                btc_data_df = df
                flag = True
            else:
                btc_data_df = pd.concat([btc_data_df, df])

            # close file
            zip_ref.close()

            # delete zipped file
            os.remove(file_name)

    # convert pandas to feather ZSTD file
    local_path = source_path.replace(local_folder, 'data/btc_option_data.ftr')
    feather.write_feather(btc_data_df, local_path, compression='zstd')
    print('btc_option_data.ftr created')

    # unzip ethereum data
    dir_name = source_path.replace(local_folder, 'zip_files/eth_option_data')
    extension = ".zip"

    # change directory from working dir to dir with files
    os.chdir(dir_name)

    eth_data_df = pd.DataFrame()
    flag = False

    # loop through items in dir
    for item in os.listdir(dir_name):
        if item.endswith(extension):
            file_name = os.path.abspath(item)
            zip_ref = zipfile.ZipFile(file_name)
            zip_ref.extractall(dir_name)

            # convert to pandas DataFrame
            local_path = dir_name + '/csv_files/eth_option_data.csv'
            df = pd.read_csv(local_path, header=0)
            if not flag:
                eth_data_df = df
                flag = True
            else:
                eth_data_df = pd.concat([eth_data_df, df])

            zip_ref.close()
            os.remove(file_name)

    # convert pandas to feather ZSTD file
    local_path = source_path.replace(local_folder, 'data/eth_option_data.ftr')
    feather.write_feather(eth_data_df, local_path, compression='zstd')
    print('eth_option_data.ftr created')

    # clean the environment: remove zip_files directory
    local_path = source_path.replace(local_folder, 'zip_files')
    shutil.rmtree(local_path, ignore_errors=True)
    print('environment clean')


# get data with a unique function
def get_server_data(host, port, username, password, local_folder):

    # create environment
    create_env(local_folder)

    # ssh remote server
    ssh_remote_server(host, port, username, password, local_folder)

    # get data and clean the environment
    clean_data_and_env(local_folder)
