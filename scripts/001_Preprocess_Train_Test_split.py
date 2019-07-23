#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../")

import pandas as pd
import itertools
import pathlib
import logging
import os
import numpy as np
import pickle
import argparse
import helpers.train_val_split_helpers as tvh
from collections import Counter

DATA_TRAIN_PATH = pathlib.Path('../data/train.csv')
DATA_TEST_PATH = pathlib.Path('../data/test.csv')
DATA_ITEMS_PATH = pathlib.Path('../data/item_metadata.csv')

OUTPUT_DF_TR = 'df_steps_tr.csv'
OUTPUT_DF_VAL = 'df_steps_val.csv'
OUTPUT_DF_TRAIN = 'df_steps_train.csv'
OUTPUT_DF_TEST = 'df_steps_test.csv'
OUTPUT_DF_SESSIONS = 'df_sessions.csv'
OUTPUT_ENCODING_DICT = 'enc_dicts_v02.pkl'
OUTPUT_CONFIG = 'config.pkl'

DEFAULT_SPLIT = 'normal'
DEFAULT_PREPROC_DIR_NAME = 'data_processed_vnormal'

def setup_args_parser():
    parser = argparse.ArgumentParser(description='Train val test split')
    parser.add_argument('--processed_data_dir_name', help='path to preprocessed data', default=DEFAULT_PREPROC_DIR_NAME)
    parser.add_argument('--split_option', help='split type. Options: normal, future', default=DEFAULT_SPLIT)
    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    return parser

def setup_logger(debug):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger

def main():
    parser = setup_args_parser()
    args = parser.parse_args()
    logger = setup_logger(args.debug)
    logger.info(100*'-')
    logger.info('Running 001_Preprocess_Train_Test_split.py')
    logger.info(100*'-')
    logger.info('split option: %s' % args.split_option)
    logger.info('processed data directory name: %s' % args.processed_data_dir_name)

    #Set up arguments
    # split_option
    if args.split_option=='normal':
        SPLIT_OPTION = 'normal'
    elif args.split_option=='future':
        SPLIT_OPTION = 'leave_out_only_clickout_with_nans'

    # processed data path
    NEW_DATA_PATH = '../data/' + args.processed_data_dir_name + '/'
    os.makedirs(NEW_DATA_PATH) if not os.path.exists(NEW_DATA_PATH) else None
    logger.info('processed data path: %s' % NEW_DATA_PATH)
    # End of set up arguments

    pad_values_dict = dict()

    pad_values_dict['action_type'] = ['0', '<S>']
    pad_values_dict['reference'] = ['0', '<S>']
    pad_values_dict['city'] = ['0', '<S>']
    pad_values_dict['country'] = ['0', '<S>']
    pad_values_dict['current_filters'] = ['0', '<S>']

    pad_values_dict['reference_aug'] = ['0', '<S>']

    ## new part
    pad_values_dict['reference|current filters|filter selection'] = ['0', '<S>']
    pad_values_dict['reference| |search for poi'] = ['0', '<S>']
    pad_values_dict['reference| |change of sort order'] = ['0', '<S>']
    pad_values_dict['reference|impressions|all item actions|interaction item image|clickout item|interaction item info|interaction item deals|interaction item rating|search for item'] = ['0', '<S>']

    pad_values_dict['reference_list| |search for poi'] = ['0', '<S>']
    pad_values_dict['reference_list| |change of sort order'] = ['0', '<S>']


    pad_values_dict['user_id'] = []
    pad_values_dict['platform'] = []
    pad_values_dict['device'] = []


    config = {'split_option': SPLIT_OPTION,
              'tr_days': [1,2,3,4],
              'val_days': [5,6],
              'encodings': pad_values_dict}



    df_train = tvh.load_df(DATA_TRAIN_PATH)
    df_test = tvh.load_df(DATA_TEST_PATH)


    enc_dicts, dec_dicts = tvh.get_enc_and_dec_dicts(df_train, df_test, config['encodings'])

    df_train, df_test = tvh.rename_sess(df_train, df_test)

    df_tr, df_val = tvh.train_val_split_all_options(df_train, tr_days=config['tr_days'],
                                                              val_days=config['val_days'],
                                                              option=config['split_option'])


    df_train_new, df_test_new = tvh.resplit_train_test_split_cheat(df_train, df_test, option=config['split_option'])


    def split_dfs(df):
        session_features = ['session_id', 'session_id_original', 'user_id', 'platform', 'device', 'day']
        steps_features = ['session_id', 'step', 'timestamp', 'action_type',
           'reference', 'city', 'current_filters',
           'impressions', 'prices', 'reference_true']

        df_sessions = df[session_features].drop_duplicates().reset_index(drop=True)
        df_steps = df[steps_features]
        return df_sessions, df_steps


    _ , df_steps_tr = split_dfs(df_tr)
    _ , df_steps_val = split_dfs(df_val)



    df_sessions_train , df_steps_train = split_dfs(df_train_new)
    df_sessions_test , df_steps_test = split_dfs(df_test_new)

    df_sessions = pd.concat([df_sessions_train, df_sessions_test], axis=0).reset_index(drop=True)


    with open(NEW_DATA_PATH+OUTPUT_ENCODING_DICT, 'wb') as handle:
        pickle.dump(enc_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(NEW_DATA_PATH+OUTPUT_CONFIG, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_sessions.to_csv(NEW_DATA_PATH+OUTPUT_DF_SESSIONS, index=False)

    df_steps_tr.to_csv(NEW_DATA_PATH+OUTPUT_DF_TR, index=False)
    df_steps_val.to_csv(NEW_DATA_PATH+OUTPUT_DF_VAL, index=False)

    df_steps_train.to_csv(NEW_DATA_PATH+OUTPUT_DF_TRAIN, index=False)
    df_steps_test.to_csv(NEW_DATA_PATH+OUTPUT_DF_TEST, index=False)



if __name__ == "__main__":
    main()
