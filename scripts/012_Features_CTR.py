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

from collections import Counter
import helpers.feature_helpers as fh


OUTPUT_DF_TR = 'df_steps_tr.csv'
OUTPUT_DF_VAL = 'df_steps_val.csv'
OUTPUT_DF_TRAIN = 'df_steps_train.csv'
OUTPUT_DF_TEST = 'df_steps_test.csv'
OUTPUT_DF_SESSIONS = 'df_sessions.csv'
OUTPUT_ENCODING_DICT = 'enc_dicts_v02.pkl'
OUTPUT_CONFIG = 'config.pkl'

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
    logger.info('Running 012_Features_CTR.py')
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

    df_steps_tr = pd.read_csv(DATA_PATH+OUTPUT_DF_TR)
    df_steps_val = pd.read_csv(DATA_PATH+OUTPUT_DF_VAL)

    df_steps_train = pd.read_csv(DATA_PATH+OUTPUT_DF_TRAIN)
    df_steps_test = pd.read_csv(DATA_PATH+OUTPUT_DF_TEST)

    df_sessions = pd.read_csv(DATA_PATH+OUTPUT_DF_SESSIONS)

    enc_dict = pickle.load(open(DATA_PATH+OUTPUT_ENCODING_DICT, "rb" ))

    # # 1. Concatenate all data and sort

    df_tr = df_steps_tr.merge(df_sessions, on='session_id')
    df_val = df_steps_val.merge(df_sessions, on='session_id')
    df_all_cv = pd.concat([df_tr, df_val], axis=0).reset_index(drop=True)

    del df_tr, df_val, df_steps_tr, df_steps_val

    df_test_new = df_steps_test.merge(df_sessions, on='session_id')
    df_train_new = df_steps_train.merge(df_sessions, on='session_id')
    df_all = pd.concat([df_train_new, df_test_new], axis=0).reset_index(drop=True)

    del df_train_new, df_test_new, df_steps_train, df_steps_test
    del df_sessions

    # ### create a dataframe with impressions list
    idx = df_all.action_type=='clickout item'
    df_all_imp_list = df_all.loc[idx,['session_id', 'step', 'impressions']].reset_index(drop=True)#.merge(Xcv_feat, on=['session_id', 'step'])
    df_all_imp_list['impressions_list_enc'] = df_all_imp_list.impressions.fillna('').str.split('|')                                             .apply(lambda s: [enc_dict['reference'].get(i) for i in s])
    df_all_imp_list.drop('impressions', axis=1, inplace=True)

    # # 2. Create Features
    # ### 2.1 past actions with items

    action_list = [a for a in df_all.action_type.unique() if 'item' in a]

    # #### 2.1.1 user level

    VAR_GROUPBY = 'user_id'
    FEATURE_NAME = 'past_actions_with_items_%s' % VAR_GROUPBY
    print (FEATURE_NAME)

    df_all_cv = df_all_cv.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)
    df_all = df_all.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)


    df = df_all_cv
    FILE_NAME = 'Xcv_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    df_feat_all = fh.make_all_interaction_features(df, action_list, enc_dict=enc_dict,
                                                       df_all_imp_list=df_all_imp_list,
                                                       VAR_GROUPBY=VAR_GROUPBY)

    df_feat_add = fh.make_interaction_features('pa_user_id_all_interactions', df,
                                         action_list,
                                         enc_dict=enc_dict,
                                         df_all_imp_list=df_all_imp_list,
                                         VAR_GROUPBY=VAR_GROUPBY)
    df_feat_all = df_feat_all.merge(df_feat_add, on=['session_id', 'step'])
    df_feat_all.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.1.1.2 all

    df = df_all
    FILE_NAME = 'X_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    df_feat_all = fh.make_all_interaction_features(df, action_list, enc_dict=enc_dict,
                                                       df_all_imp_list=df_all_imp_list,
                                                       VAR_GROUPBY=VAR_GROUPBY)

    df_feat_add = fh.make_interaction_features('pa_user_id_all_interactions', df,
                                         action_list,
                                         enc_dict=enc_dict,
                                         df_all_imp_list=df_all_imp_list,
                                         VAR_GROUPBY=VAR_GROUPBY)
    df_feat_all = df_feat_all.merge(df_feat_add, on=['session_id', 'step'])
    df_feat_all.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.1.2 session level

    VAR_GROUPBY = 'session_id'
    FEATURE_NAME = 'past_actions_with_items_%s' % VAR_GROUPBY
    print (FEATURE_NAME)

    df_all_cv = df_all_cv.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)
    df_all = df_all.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)


    # #### 2.1.2.1 cv

    df = df_all_cv
    FILE_NAME = 'Xcv_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    df_feat_all = fh.make_all_interaction_features(df, action_list, enc_dict=enc_dict,
                                                       df_all_imp_list=df_all_imp_list,
                                                       VAR_GROUPBY=VAR_GROUPBY)

    df_feat_add = fh.make_interaction_features('pa_session_id_all_interactions', df,
                                         action_list,
                                         enc_dict=enc_dict,
                                         df_all_imp_list=df_all_imp_list,
                                         VAR_GROUPBY=VAR_GROUPBY)
    df_feat_all = df_feat_all.merge(df_feat_add, on=['session_id', 'step'])
    df_feat_all.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.1.2.2 all

    df = df_all
    FILE_NAME = 'X_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    df_feat_all = fh.make_all_interaction_features(df, action_list, enc_dict=enc_dict,
                                                       df_all_imp_list=df_all_imp_list,
                                                       VAR_GROUPBY=VAR_GROUPBY)

    df_feat_add = fh.make_interaction_features('pa_session_id_all_interactions', df,
                                         action_list,
                                         enc_dict=enc_dict,
                                         df_all_imp_list=df_all_imp_list,
                                         VAR_GROUPBY=VAR_GROUPBY)
    df_feat_all = df_feat_all.merge(df_feat_add, on=['session_id', 'step'])
    df_feat_all.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # ### 2.2. past_impressions

    # #### 2.2.1 user level

    VAR_GROUPBY = 'user_id'
    FEATURE_NAME = 'past_impressions_%s' % VAR_GROUPBY
    print (FEATURE_NAME)

    df_all_cv = df_all_cv.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)
    df_all = df_all.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)


    # #### 2.2.1.1 cv

    df = df_all_cv
    FILE_NAME = 'Xcv_%s.csv' % FEATURE_NAME
    print (FILE_NAME)
    df_feat = fh.make_impressions_features(df, enc_dict=enc_dict,
                                                   df_all_imp_list=df_all_imp_list,
                                                   VAR_GROUPBY=VAR_GROUPBY)
    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.2.1.2 all

    df = df_all
    FILE_NAME = 'X_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    df_feat = fh.make_impressions_features(df, enc_dict=enc_dict,
                                                   df_all_imp_list=df_all_imp_list,
                                                   VAR_GROUPBY=VAR_GROUPBY)
    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.2.2 session_id level

    VAR_GROUPBY = 'session_id'
    FEATURE_NAME = 'past_impressions_%s' % VAR_GROUPBY
    print (FEATURE_NAME)

    df_all_cv = df_all_cv.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)
    df_all = df_all.sort_values(['user_id', 'day','session_id', 'step', 'timestamp']).reset_index(drop=True)

    # #### 2.2.2.1 cv

    df = df_all_cv
    FILE_NAME = 'Xcv_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    df_feat = fh.make_impressions_features(df, enc_dict=enc_dict,
                                                   df_all_imp_list=df_all_imp_list,
                                                   VAR_GROUPBY=VAR_GROUPBY)
    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.2.2.2 all

    df = df_all
    FILE_NAME = 'X_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    df_feat = fh.make_impressions_features(df, enc_dict=enc_dict,
                                                   df_all_imp_list=df_all_imp_list,
                                                   VAR_GROUPBY=VAR_GROUPBY)
    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # ### 2.3 CTR, combining 2.1 and 2.2

    # #### 2.3.1 user_id level

    VAR_GROUPBY = 'user_id'
    FEATURE_NAME = 'CTR_%s' % VAR_GROUPBY
    print(FEATURE_NAME)

    # #### 2.3.1.2 cv

    FILE_NAME = 'Xcv_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    Xcv_piu = pd.read_csv(FEATURES_PATH+'Xcv_past_impressions_user_id'+'.csv', compression='gzip')
    Xcv_piu['pi_user_id'] = Xcv_piu['pi_user_id'].apply(lambda s: eval(s))

    Xcv_pau = pd.read_csv(FEATURES_PATH+'Xcv_past_actions_with_items_user_id'+'.csv', compression='gzip')
    Xcv_pau['pa_user_id_clickout_item'] = Xcv_pau['pa_user_id_clickout_item'].apply(lambda s: eval(s))

    df_feat = Xcv_piu.merge(Xcv_pau[['pa_user_id_clickout_item', 'session_id', 'step']], on=['session_id', 'step'])

    del Xcv_piu, Xcv_pau

    CPI_DEFAULT = 1.0 #how many times the item was on the list if it never was,
                      #maybe give 0.9 to distinguish from the ones that were on the list?
    df_feat[FEATURE_NAME] = df_feat.apply(lambda row:  \
                    [round(row.pa_user_id_clickout_item[i]/max(float(s), CPI_DEFAULT), ndigits=2) \
                        for i,s in enumerate(row.pi_user_id)], axis=1)


    df_feat = df_feat[['session_id', 'step', FEATURE_NAME]]
    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.3.1.2 all

    FILE_NAME = 'X_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    X_piu = pd.read_csv(FEATURES_PATH+'X_past_impressions_user_id'+'.csv', compression='gzip')
    X_piu['pi_user_id'] = X_piu['pi_user_id'].apply(lambda s: eval(s))

    X_pau = pd.read_csv(FEATURES_PATH+'X_past_actions_with_items_user_id'+'.csv', compression='gzip')
    X_pau['pa_user_id_clickout_item'] = X_pau['pa_user_id_clickout_item'].apply(lambda s: eval(s))

    df_feat = X_piu.merge(X_pau[['pa_user_id_clickout_item', 'session_id', 'step']], on=['session_id', 'step'])

    del X_piu, X_pau


    CPI_DEFAULT = 1.0 #how many times the item was on the list if it never was,
                      #maybe give 0.9 to distinguish from the ones that were on the list?
    df_feat[FEATURE_NAME] = df_feat.apply(lambda row: \
                    [round(row.pa_user_id_clickout_item[i]/max(float(s), CPI_DEFAULT), ndigits=2) \
                        for i,s in enumerate(row.pi_user_id)], axis=1)
    df_feat = df_feat[['session_id', 'step', FEATURE_NAME]]

    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.3.2 session_id level¶

    VAR_GROUPBY = 'session_id'
    FEATURE_NAME = 'CTR_%s' % VAR_GROUPBY
    print(FEATURE_NAME)

    # #### 2.3.2.1 cv

    FILE_NAME = 'Xcv_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    Xcv_pis = pd.read_csv(FEATURES_PATH+'Xcv_past_impressions_session_id'+'.csv', compression='gzip')
    Xcv_pis['pi_session_id'] = Xcv_pis['pi_session_id'].apply(lambda s: eval(s))

    Xcv_pas = pd.read_csv(FEATURES_PATH+'Xcv_past_actions_with_items_session_id'+'.csv', compression='gzip')
    Xcv_pas['pa_session_id_clickout_item'] = Xcv_pas['pa_session_id_clickout_item'].apply(lambda s: eval(s))

    df_feat = Xcv_pis.merge(Xcv_pas[['pa_session_id_clickout_item', 'session_id', 'step']], on=['session_id', 'step'])

    del Xcv_pis, Xcv_pas

    CPI_DEFAULT = 1.0 #how many times the item was on the list if it never was,
                      #maybe give 0.9 to distinguish from the ones that were on the list?
    df_feat[FEATURE_NAME] = df_feat.apply(lambda row: \
                    [round(row.pa_session_id_clickout_item[i]/max(float(s), CPI_DEFAULT), ndigits=2) \
                        for i,s in enumerate(row.pi_session_id)], axis=1)

    df_feat = df_feat[['session_id', 'step', FEATURE_NAME]]

    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # #### 2.3.2.2 all

    FILE_NAME = 'X_%s.csv' % FEATURE_NAME
    print (FILE_NAME)

    X_pis = pd.read_csv(FEATURES_PATH+'X_past_impressions_session_id'+'.csv', compression='gzip')
    X_pis['pi_session_id'] = X_pis['pi_session_id'].apply(lambda s: eval(s))

    X_pas = pd.read_csv(FEATURES_PATH+'X_past_actions_with_items_session_id'+'.csv', compression='gzip')
    X_pas['pa_session_id_clickout_item'] = X_pas['pa_session_id_clickout_item'].apply(lambda s: eval(s))

    df_feat = X_pis.merge(X_pas[['pa_session_id_clickout_item', 'session_id', 'step']], on=['session_id', 'step'])

    CPI_DEFAULT = 1.0 #how many times the item was on the list if it never was,
                      #maybe give 0.9 to distinguish from the ones that were on the list?
    df_feat[FEATURE_NAME] = df_feat.apply(lambda row: [round(row.pa_session_id_clickout_item[i]/max(float(s), CPI_DEFAULT), ndigits=2)                              for i,s in enumerate(row.pi_session_id)], axis=1)

    df_feat = df_feat[['session_id', 'step', FEATURE_NAME]]

    df_feat.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # ### 3. Put everything together

    # ### cv

    FILE_NAME = 'Xcv_CTR_ALL.csv'
    print (FILE_NAME)

    Xcv_piu = pd.read_csv(FEATURES_PATH+'Xcv_past_impressions_user_id'+'.csv', compression='gzip')
    Xcv_pis = pd.read_csv(FEATURES_PATH+'Xcv_past_impressions_session_id'+'.csv', compression='gzip')
    Xcv_pau = pd.read_csv(FEATURES_PATH+'Xcv_past_actions_with_items_user_id'+'.csv', compression='gzip')
    Xcv_pas = pd.read_csv(FEATURES_PATH+'Xcv_past_actions_with_items_session_id'+'.csv', compression='gzip')
    Xcv_ctru = pd.read_csv(FEATURES_PATH+'Xcv_CTR_user_id'+'.csv', compression='gzip')
    Xcv_ctrs = pd.read_csv(FEATURES_PATH+'Xcv_CTR_session_id'+'.csv', compression='gzip')


    df_ctr_feat_cv = Xcv_piu.merge(Xcv_pis, on=['session_id', 'step']) \
                             .merge(Xcv_pau, on=['session_id', 'step']) \
                             .merge(Xcv_pas, on=['session_id', 'step']) \
                             .merge(Xcv_ctru, on=['session_id', 'step']) \
                             .merge(Xcv_ctrs, on=['session_id', 'step'])

    df_ctr_feat_cv.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


    # ###  all

    FILE_NAME = 'X_CTR_ALL.csv'
    print (FILE_NAME)

    X_piu = pd.read_csv(FEATURES_PATH+'X_past_impressions_user_id'+'.csv', compression='gzip')
    X_pis = pd.read_csv(FEATURES_PATH+'X_past_impressions_session_id'+'.csv', compression='gzip')
    X_pau = pd.read_csv(FEATURES_PATH+'X_past_actions_with_items_user_id'+'.csv', compression='gzip')
    X_pas = pd.read_csv(FEATURES_PATH+'X_past_actions_with_items_session_id'+'.csv', compression='gzip')
    X_ctru = pd.read_csv(FEATURES_PATH+'X_CTR_user_id'+'.csv', compression='gzip')
    X_ctrs = pd.read_csv(FEATURES_PATH+'X_CTR_session_id'+'.csv', compression='gzip')

    df_ctr_feat_all = X_piu.merge(X_pis, on=['session_id', 'step']) \
                           .merge(X_pau, on=['session_id', 'step']) \
                           .merge(X_pas, on=['session_id', 'step']) \
                           .merge(X_ctru, on=['session_id', 'step']) \
                           .merge(X_ctrs, on=['session_id', 'step'])

    df_ctr_feat_all.to_csv(FEATURES_PATH+FILE_NAME, index=False, compression='gzip')


if __name__ == "__main__":
    main()
