#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import pathlib
import pickle
import os
import itertools
import argparse
import logging
import helpers.feature_helpers as fh

from collections import Counter


OUTPUT_DF_TR = 'df_steps_tr.csv'
OUTPUT_DF_VAL = 'df_steps_val.csv'
OUTPUT_DF_TRAIN = 'df_steps_train.csv'
OUTPUT_DF_TEST = 'df_steps_test.csv'
OUTPUT_DF_SESSIONS = 'df_sessions.csv'
OUTPUT_ENCODING_DICT = 'enc_dicts_v02.pkl'
OUTPUT_CONFIG = 'config.pkl'
OUTPUT_NORMLIZATIONS_VAL = 'Dwell_normalizations_val.pkl'
OUTPUT_NORMLIZATIONS_SUBM = 'Dwell_normalizations_submission.pkl'

DEFAULT_FEATURES_DIR_NAME = 'nn_vnormal'
DEFAULT_PREPROC_DIR_NAME = 'data_processed_vnormal'


def setup_args_parser():
    parser = argparse.ArgumentParser(description='Create cv features')
    parser.add_argument('--processed_data_dir_name', help='path to preprocessed data', default=DEFAULT_PREPROC_DIR_NAME)
    parser.add_argument('--features_dir_name', help='features directory name', default=DEFAULT_FEATURES_DIR_NAME)
    #parser.add_argument('--split_option', help='split type. Options: normal, future', default=DEFAULT_SPLIT)
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
    #logger.info('split option: %s' % args.split_option)
    logger.info(100*'-')
    logger.info('Running 013_Features_Dwell.py')
    logger.info(100*'-')
    logger.info('processed data directory name: %s' % args.processed_data_dir_name)
    logger.info('features directory name: %s' % args.features_dir_name)

    #Set up arguments
    # # split_option
    # if args.split_option=='normal':
    #     SPLIT_OPTION = 'normal'
    # elif args.split_option=='future':
    #     SPLIT_OPTION = 'leave_out_only_clickout_with_nans'

    # processed data path
    DATA_PATH = '../data/' + args.processed_data_dir_name + '/'
    #os.makedirs(DATA_PATH) if not os.path.exists(DATA_PATH) else None
    logger.info('processed data path: %s' % DATA_PATH)

    # features data path
    FEATURES_PATH = '../features/' + args.features_dir_name + '/'
    #os.makedirs(FEATURES_PATH) if not os.path.exists(FEATURES_PATH) else None
    logger.info('features path: %s' % FEATURES_PATH)
    # End of set up arguments


    config = pickle.load(open(DATA_PATH+OUTPUT_CONFIG, "rb" ))
    config


    # ### read data
    df_steps_tr = pd.read_csv(DATA_PATH+OUTPUT_DF_TR)
    df_steps_val = pd.read_csv(DATA_PATH+OUTPUT_DF_VAL)
    df_steps_train = pd.read_csv(DATA_PATH+OUTPUT_DF_TRAIN)
    df_steps_test = pd.read_csv(DATA_PATH+OUTPUT_DF_TEST)
    df_sessions = pd.read_csv(DATA_PATH+OUTPUT_DF_SESSIONS)

    enc_dict = pickle.load(open(DATA_PATH+OUTPUT_ENCODING_DICT, "rb" ))

    # ## Concatenate all data
    # #### validation

    df_tr = df_steps_tr.merge(df_sessions, on='session_id')
    df_val = df_steps_val.merge(df_sessions, on='session_id')
    df_all_cv = pd.concat([df_tr, df_val], axis=0).reset_index(drop=True)
    del df_tr, df_val, df_steps_tr, df_steps_val

    # #### all

    df_test_new = df_steps_test.merge(df_sessions, on='session_id')
    df_train_new = df_steps_train.merge(df_sessions, on='session_id')
    df_all = pd.concat([df_train_new, df_test_new], axis=0).reset_index(drop=True)
    del df_train_new, df_test_new, df_steps_train, df_steps_test
    del df_sessions

    # ### create a dataframe with impressions list¶

    idx = df_all.action_type=='clickout item'
    df_all_imp_list = df_all.loc[idx,['session_id', 'step', 'impressions']].reset_index(drop=True)
    df_all_imp_list['impressions_list_enc'] = df_all_imp_list.impressions.fillna('').str.split('|') \
                                            .apply(lambda s: [enc_dict['reference'].get(i) for i in s])
    df_all_imp_list.drop('impressions', axis=1, inplace=True)


    # # Get Dwell

    VAR_GROUPBY = 'session_id'
    FEATURE_NAME = 'past_dwell_with_items_%s' % VAR_GROUPBY
    print (FEATURE_NAME)

    df_all_cv = df_all_cv.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)
    df_all = df_all.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)


    # ### validation

    VARS_ = ['session_id', 'step', 'timestamp', 'action_type', 'reference']
    df = df_all_cv[VARS_].copy()
    FILE_NAME = 'Dcv_%s.gz' % FEATURE_NAME
    print (FILE_NAME)


    df['reference_enc'] = df.reference.apply(lambda s: str(enc_dict['reference'].get(s)))
    df = df.drop('reference', axis=1)

    df['next_timestamp'] = df.groupby('session_id').timestamp.shift(-1)

    df['duration'] = df.next_timestamp-df.timestamp
    df['duration'] = df['duration'].fillna(0)
    df = df.drop(['timestamp', 'next_timestamp'], axis=1)

    df['ref_dwell_dict'] = df.apply(lambda row: dict([(row.reference_enc, row.duration)]), axis=1).apply(Counter)
    df = df.drop(['reference_enc', 'duration'], axis=1)

    df['cumsum_dwell_dict'] = df.groupby('session_id').ref_dwell_dict.transform(pd.Series.cumsum)
    df['cumsum_dwell_dict_shift'] = df.groupby('session_id').cumsum_dwell_dict.shift()
    df = df.drop(['ref_dwell_dict', 'cumsum_dwell_dict'], axis=1)

    df_feat = df.merge(df_all_imp_list, on=['session_id', 'step'])
    df_feat[FEATURE_NAME] = df_feat.apply(lambda row: [row.cumsum_dwell_dict_shift.get(str(s), -1) for s in row.impressions_list_enc] \
                                if pd.notnull(row.cumsum_dwell_dict_shift) else [-1 for s in row.impressions_list_enc], axis=1)

    df_feat = df_feat[['session_id', 'step', FEATURE_NAME]]
    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')

    print (FEATURES_PATH+FILE_NAME)


    def get_imp_means_and_stds(df_tr_=None, var_group = 'seq_num_new'):
        aux = df_tr_[[var_group]].reset_index(drop=True)[var_group]

        lista=list(itertools.chain.from_iterable(aux))
        listasemnan = [s for s in lista if s!=-1]
        means = np.mean(listasemnan)

        stds = np.std(listasemnan)
        maxv = np.max(listasemnan)

        return means, stds, maxv

    def get_log_imp_means_and_stds(df_tr_=None,  var_group = 'seq_num_new'):
        aux = df_tr_[[var_group]].reset_index(drop=True)[var_group]

        lista=list(itertools.chain.from_iterable(aux))
        listasemnan = np.log(np.array([s for s in lista if s!=-1])+1.9)
        means = np.mean(listasemnan)

        stds = np.std(listasemnan)
        maxv = np.max(listasemnan)

        return means, stds,maxv


    normalizations_dict = {}
    normalizations_dict['dwell_times'] = {}
    means, stds, maxv = get_imp_means_and_stds(df_tr_=df_feat, var_group = 'past_dwell_with_items_session_id')

    normalizations_dict['dwell_times']['means'] = means
    normalizations_dict['dwell_times']['stds'] = stds
    normalizations_dict['dwell_times']['max'] = maxv

    normalizations_dict['dwell_times_log'] = {}
    means, stds, maxv = get_log_imp_means_and_stds(df_tr_=df_feat, var_group = 'past_dwell_with_items_session_id')

    normalizations_dict['dwell_times_log']['means'] = means
    normalizations_dict['dwell_times_log']['stds'] = stds
    normalizations_dict['dwell_times_log']['max'] = maxv


    with open(FEATURES_PATH+OUTPUT_NORMLIZATIONS_VAL, 'wb') as handle:
        pickle.dump(normalizations_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # ### all

    VARS_ = ['session_id', 'step', 'timestamp', 'action_type', 'reference']
    df = df_all[VARS_].copy()
    FILE_NAME = 'D_%s.gz' % FEATURE_NAME
    print (FILE_NAME)

    df['reference_enc'] = df.reference.apply(lambda s: str(enc_dict['reference'].get(s)))
    df = df.drop('reference', axis=1)

    df['next_timestamp'] = df.groupby('session_id').timestamp.shift(-1)

    df['duration'] = df.next_timestamp-df.timestamp
    df['duration'] = df['duration'].fillna(0)
    df = df.drop(['timestamp', 'next_timestamp'], axis=1)

    df['ref_dwell_dict'] = df.apply(lambda row: dict([(row.reference_enc, row.duration)]), axis=1).apply(Counter)
    df = df.drop(['reference_enc', 'duration'], axis=1)

    df['cumsum_dwell_dict'] = df.groupby('session_id').ref_dwell_dict.transform(pd.Series.cumsum)
    df['cumsum_dwell_dict_shift'] = df.groupby('session_id').cumsum_dwell_dict.shift()
    df = df.drop(['ref_dwell_dict', 'cumsum_dwell_dict'], axis=1)

    df_feat = df.merge(df_all_imp_list, on=['session_id', 'step'])
    df_feat[FEATURE_NAME] = df_feat.apply(lambda row: [row.cumsum_dwell_dict_shift.get(str(s), -1) for s in row.impressions_list_enc] \
                   if pd.notnull(row.cumsum_dwell_dict_shift) else [-1 for s in row.impressions_list_enc], axis=1)


    df_feat = df_feat[['session_id', 'step', FEATURE_NAME]]
    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')

    print (FEATURES_PATH+FILE_NAME)

    normalizations_dict = {}
    normalizations_dict['dwell_times'] = {}
    means, stds, maxv = get_imp_means_and_stds(df_tr_=df_feat, var_group = 'past_dwell_with_items_session_id')

    normalizations_dict['dwell_times']['means'] = means
    normalizations_dict['dwell_times']['stds'] = stds
    normalizations_dict['dwell_times']['max'] = maxv

    normalizations_dict['dwell_times_log'] = {}
    means, stds, maxv = get_log_imp_means_and_stds(df_tr_=df_feat, var_group = 'past_dwell_with_items_session_id')

    normalizations_dict['dwell_times_log']['means'] = means
    normalizations_dict['dwell_times_log']['stds'] = stds
    normalizations_dict['dwell_times_log']['max'] = maxv

    with open(FEATURES_PATH+OUTPUT_NORMLIZATIONS_SUBM, 'wb') as handle:
        pickle.dump(normalizations_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
