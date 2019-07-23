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
from sklearn.preprocessing import MultiLabelBinarizer
import helpers.feature_helpers as fh

OUTPUT_DF_TR = 'df_steps_tr.csv'
OUTPUT_DF_VAL = 'df_steps_val.csv'
OUTPUT_DF_TRAIN = 'df_steps_train.csv'
OUTPUT_DF_TEST = 'df_steps_test.csv'
OUTPUT_DF_SESSIONS = 'df_sessions.csv'
OUTPUT_ENCODING_DICT = 'enc_dicts_v02.pkl'
OUTPUT_CONFIG = 'config.pkl'
OUTPUT_NORMLIZATIONS_VAL = 'normalizations_val.pkl'
OUTPUT_NORMLIZATIONS_SUBM = 'normalizations_submission.pkl'

DEFAULT_FEATURES_DIR_NAME = 'nn_vnormal'
DEFAULT_PREPROC_DIR_NAME = 'data_processed_vnormal'
DEFAULT_SPLIT = 'normal'


def setup_args_parser():
    parser = argparse.ArgumentParser(description='Create cv features')
    parser.add_argument('--processed_data_dir_name', help='path to preprocessed data', default=DEFAULT_PREPROC_DIR_NAME)
    parser.add_argument('--features_dir_name', help='features directory name', default=DEFAULT_FEATURES_DIR_NAME)
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
    logger.info('Running 015_Features_General_02.py')
    logger.info(100*'-')
    logger.info('split option: %s' % args.split_option)
    logger.info('processed data directory name: %s' % args.processed_data_dir_name)
    logger.info('features directory name: %s' % args.features_dir_name)

    #Set up arguments
    # # split_option
    # if args.split_option=='normal':
    #     SPLIT_OPTION = 'normal'
    # elif args.split_option=='future':
    #     SPLIT_OPTION = 'leave_out_only_clickout_with_nans'

    is_normal = args.split_option=='normal'

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


    # ### 0. read data

    df_steps_tr = pd.read_csv(DATA_PATH+OUTPUT_DF_TR)#,nrows=10000)
    df_steps_val = pd.read_csv(DATA_PATH+OUTPUT_DF_VAL)#,nrows=10000)
    df_steps_train = pd.read_csv(DATA_PATH+OUTPUT_DF_TRAIN)
    df_steps_test = pd.read_csv(DATA_PATH+OUTPUT_DF_TEST)
    df_sessions = pd.read_csv(DATA_PATH+OUTPUT_DF_SESSIONS)

    enc_dict = pickle.load(open(DATA_PATH+OUTPUT_ENCODING_DICT, "rb" ))

    df_items = pd.read_csv(FEATURES_PATH+'Item_Features.csv', index_col=['item_id_enc'])

    dic_items = df_items[['is_from_stars_nan','is_rating_nan','is_stars_nan','stars_enc','from_stars_enc','rating_enc','stars','from_stars','rating','n_prop','prop_list_enc']].to_dict()

    DATA_ITEMS_PATH = pathlib.Path('../data/item_metadata.csv')
    df_items = pd.read_csv(DATA_ITEMS_PATH)

    df_items['properties_list'] = df_items.properties.fillna('').str.split('|').apply(lambda s: [i for i in s if i!=''])

    mlb_items = MultiLabelBinarizer()
    mlb_items.fit(df_items.properties_list)

    dic_items['item2proplist']  = df_items[['item_id','properties_list']].set_index('item_id').to_dict()['properties_list']
    df_items['properties_count'] = df_items['properties_list'].apply(len)
    dic_items['item2propcount'] = df_items[['item_id','properties_count']].set_index('item_id').to_dict()['properties_count']

    feat_dict = dict()


    def make_sess_df(df, enc_dic,dic_items, mlb_items):

        feat_group_dict = dict()
        feat_list= ['session_id','step','reference_true_enc','impressions',
                    'time_since_prev_clickout_item',
                    'time_since_prev_action','time_since_prev_any_action','prices','check']

        new_vars_time = ['time_since_prev_search_for_item_corrected',
                         'prev_ref_search_for_item_corrected_enc',
                         'time_since_prev_clickout_item_corrected',
                         'prev_ref_clickout_item_corrected_enc',
                         'time_since_prev_interaction_item_image_corrected',
                         'prev_ref_interaction_item_image_corrected_enc',
                         'time_since_prev_interaction_item_info_corrected',
                         'prev_ref_interaction_item_info_corrected_enc',
                         'time_since_prev_interaction_item_deals_corrected',
                         'prev_ref_interaction_item_deals_corrected_enc',
                         'time_since_prev_interaction_item_rating_corrected',
                         'prev_ref_interaction_item_rating_corrected_enc',
                         'time_since_prev_any_item_action_corrected',
                         'prev_ref_any_item_action_corrected_enc',
                         'time_since_prev_any_all_action_corrected',
                         'prev_ref_any_all_action_corrected_enc',
                         'impressions_list_lastaction_enc',
                         'impressions_list_timesincelastaction']

        feat_list = feat_list+new_vars_time

        df_ = df.copy().reset_index(drop=True)

        #----------------acrescenta o token de início de sessão --------------------------
        df_start =  df_[df_["step"] == 1].copy().reset_index(drop=True)
        df_start['step'] = 0
        df_start['timestamp'] += 1
        df_start['action_type'] = '<S>'
        df_start['reference'] = '<S>'
        df_start['city'] = '<S>'
        df_start['country'] = '<S>'

        df_ = pd.concat((df_start, df_), sort=True).copy().sort_values(['session_id', 'step'])             .reset_index(drop=True)

        df_['reference_enc'] = df_['reference'].map(enc_dic['reference'])
        df_['reference_true_enc'] = df_['reference_true'].map(enc_dic['reference'])
        #---------------- time ---------------------------------------------------------
        print('time features')

        NAN_VAL = -1
        MAX_VAL = None

        df_ = fh.get_time_since_prev(df_, new_var_name = 'time_since_prev_clickout_item',
                                          action_list = ['clickout item'], nan_val = NAN_VAL, max_val = MAX_VAL)

        item_actions = [a for a in enc_dict['action_type'].keys() if 'item' in a]
        df_ = fh.get_time_since_prev(df_, new_var_name = 'time_since_prev_action',
                                          action_list = item_actions, nan_val = NAN_VAL, max_val = MAX_VAL)

        all_actions = enc_dict['action_type'].keys()
        df_ = fh.get_time_since_prev(df_, new_var_name = 'time_since_prev_any_action',
                                          action_list = all_actions, nan_val = NAN_VAL, max_val = MAX_VAL)


        ##---------- novas do tempo corrigidas --------------
        for act in item_actions:
            df_ = fh.get_time_since_prev_corrected(df_, action_list=[act],
                                                enc_dict=enc_dict['reference'], var_name=act.replace(' ', '_')+'_corrected',
                                                  nan_val=NAN_VAL)

        df_ = fh.get_time_since_prev_corrected(df_, action_list=item_actions,
                                         enc_dict=enc_dict['reference'], var_name='any_item_action_corrected',
                                              nan_val=-1)


        df_ = fh.get_time_since_prev_corrected(df_, action_list=all_actions,
                                         enc_dict = enc_dict['reference'], var_name='any_all_action_corrected',
                                              nan_val=NAN_VAL)



        ## novas do tempo em impressions ---------------
        df_ = fh.add_impressions_list_time_features(df_, action_list = item_actions,
                                               enc_dict=enc_dict['action_type'],
                                               var_name_action='impressions_list_lastaction_enc',
                                               var_name_time='impressions_list_timesincelastaction')

        #-------------- some new features -----------------------------------------------
        df_temp = df_.groupby('session_id').step.count().reset_index(name='num_steps_in_session')
        df_ = df_.merge(df_temp, on='session_id')

        df_['seq_t1_corr'] = df_.apply(lambda row: [row.time_since_prev_clickout_item_corrected], axis=1)
        df_['seq_t2_corr'] = df_.apply(lambda row: [row.time_since_prev_any_item_action_corrected], axis=1)
        df_['seq_t3_corr'] = df_.apply(lambda row: [row.time_since_prev_any_all_action_corrected], axis=1)

        feat_group_dict['df_seq_step_time'] = ['time_since_prev_any_action']

        df_['seq_step_time'] = df_.apply(lambda row: [row[f] for f in feat_group_dict['df_seq_step_time']], axis=1)

        feat_group_dict['seq_time_new'] = ['time_since_prev_search_for_item_corrected',
                            'time_since_prev_clickout_item_corrected',
                            'time_since_prev_interaction_item_image_corrected',
                            'time_since_prev_interaction_item_info_corrected',
                            'time_since_prev_interaction_item_deals_corrected',
                            'time_since_prev_interaction_item_rating_corrected',
                            'time_since_prev_any_item_action_corrected',
                            'time_since_prev_any_all_action_corrected']

        df_['seq_time_new'] = df_.apply(lambda row: [row[f] for f in feat_group_dict['seq_time_new']], axis=1)
        df_['check']= (df_.action_type=='clickout item')&(df_.reference.isnull())

        #-----------------------------------------------------
        df_click = df_[df_["action_type"] == "clickout item"].copy().reset_index(drop=True)
        df_click['ones'] = 1
        df_click['click_count'] = df_click.groupby('session_id').ones.cumsum()

        df_out = df_.merge(df_click[['session_id', 'step','click_count']], how='left', on=['session_id', 'step'])
        df_out['click_count'] =  df_out['click_count'].fillna(0)
        df_out['step_before_click'] = df_out.groupby('session_id')['click_count'].cumsum()

        #-----------------------------------------------------
        #ciclo para retirar as sub sessões ate aos clicks
        df = []
        step_max = df_out.click_count.max()
        for k in range(1,int(step_max)+1):
            if k%10==0: print('processing click number:',k)
            session_list = df_out[df_out.click_count == k]['session_id'].tolist()
            # retiro apenas as sessões que tem k clicks
            #Nota: o k*(k+1)/2 vem do segundo cumsum() que gera números triangulares ...
            df_temp = df_out[(df_out.step_before_click < (k*(k+1)/2)) & (df_out.session_id.isin(session_list))].reset_index(drop=True)

            df_sess_t1_corr = df_temp.groupby('session_id').seq_t1_corr.apply(list)
            df_sess_t1_corr = df_sess_t1_corr.to_frame()

            df_sess_t2_corr = df_temp.groupby('session_id').seq_t2_corr.apply(list)
            df_sess_t2_corr = df_sess_t2_corr.to_frame()

            df_sess_t3_corr = df_temp.groupby('session_id').seq_t3_corr.apply(list)
            df_sess_t3_corr = df_sess_t3_corr.to_frame()

            df_seq_step_time = df_temp.groupby('session_id').seq_step_time.apply(list)
            df_seq_step_time = df_seq_step_time.to_frame()

            df_seq_time_new = df_temp.groupby('session_id').seq_time_new.apply(list)
            df_seq_time_new = df_seq_time_new.to_frame()

            df_seq_step_time = df_seq_step_time.reset_index()
            #---------------------------------------------------------
            df_temp = df_out[df_out.click_count == k][feat_list].reset_index(drop=True).copy()
            df_temp['imp_list'] = df_temp.impressions.fillna('').str.split('|').apply(lambda list_s: [enc_dic['reference'][v] for v in list_s if v])
            df_temp['prices_list'] = df_temp.prices.fillna('').str.split('|')
            df_temp.drop('impressions', axis=1, inplace=True)
            df_temp.drop('prices', axis=1, inplace=True)

            df_sess = df_seq_step_time.merge(df_seq_time_new, how='left', on=['session_id'])

            df_sess = df_sess.merge(df_temp, how='left', on=['session_id'])

            df_sess = df_sess.merge(df_sess_t1_corr, how='left', on=['session_id'])
            df_sess = df_sess.merge(df_sess_t2_corr, how='left', on=['session_id'])
            df_sess = df_sess.merge(df_sess_t3_corr, how='left', on=['session_id'])
            df.append(df_sess)

        df_all = pd.concat(df).reset_index()

        df_temp = df_all.groupby('session_id').step.max().reset_index()
        df_temp['is_last_step'] = True

        df_all = df_all.merge(df_temp, how='left', on=['session_id', 'step'])
        df_all['is_last_step'] =  df_all['is_last_step'].fillna(False)

        return df_all, feat_group_dict


    def f_is_in(x):
        return x.reference_true_enc in x.imp_list

    def not_in_impressions_filter(df):

        df['is_in'] = df.apply(f_is_in, axis=1)
        df_= df[df.is_in==True].copy()

        df_.drop('is_in', axis=1, inplace=True)
        return df_


    def only_last_filter(df):
        df_= df[(df.is_last_step==True) & (df.check==True)].copy()
        return df_

    def all_except_last_filter(df):
        df_= df[df.is_last_step==False].copy()
        return df_

    def get_index(row):
        return row.imp_list.index(row.reference_true_enc)

    def make_target(df, is_test=False):
        if is_test:
            df['targ'] = 'nan'
        else:
            df['targ'] = df.apply(get_index, axis=1)

        df.drop('reference_true_enc', axis=1, inplace=True)
        return df

    def get_targ(row):
        return row.impr == row.reference_true_enc

    def make_target_ctr(df, is_test=False):
        if is_test:
            df['targ'] = 'nan'
        else:
            df['targ'] = df.apply(get_targ, axis=1)

        df.drop('reference_true_enc', axis=1, inplace=True)
        return df

    def explode_imp(df, variable_list=None, variable_name = None):
        OTHER_FEATURES = df.drop(['imp_list_info'], axis=1).columns.tolist()
        return df[variable_list].apply(pd.Series) \
                    .merge(df, right_index = True, left_index = True) \
                    .drop([variable_list], axis = 1) \
                    .melt(id_vars = OTHER_FEATURES, value_name = variable_name) \
                    .drop("variable", axis = 1)

    def make_ctr_features(df_in):
        df = df_in.copy()
        df['imp_list_info'] = df.imp_list.apply(lambda s: list(range(len(s))))
        df['imp_list_info'] = df.apply(lambda row: [(i,e) for i,e in enumerate(row.imp_list)], axis=1)#df.imp_list.apply(lambda s: list(range(len(s))))

        df.drop(['prices_list'], inplace=True, axis=1)

        df_ctr = explode_imp(df, variable_list='imp_list_info', variable_name = 'imp_info')

        df_ctr = df_ctr.dropna()

        if df_ctr.shape[0] != df.imp_list.apply(len).sum():
            print('diferente')
        else:
            print('done')

        df_ctr['pos'] = df_ctr.imp_info.apply(lambda s:s[0])
        df_ctr['impr'] = df_ctr.imp_info.apply(lambda s:s[1])
        df_ctr.drop('imp_info', axis=1, inplace=True)

        return df_ctr


    # ### 1. Compute feature

    # ### 1.1 df_val

    df_val = df_steps_val.merge(df_sessions, on='session_id')
    df_val_f, feat_group_dict = make_sess_df(df_val, enc_dict, dic_items, mlb_items)

    if not is_normal:
        df_to_join_tr = all_except_last_filter(df_val_f)

    df_val_f = only_last_filter(df_val_f)
    df_val_f = not_in_impressions_filter(df_val_f)
    df_val_f = make_target(df_val_f)


    # ### 1.2 df_tr

    df_tr = df_steps_tr.merge(df_sessions, on='session_id')
    df_tr_f, feat_group_dict = make_sess_df(df_tr, enc_dict,dic_items,mlb_items)

    if not is_normal:
        df_tr_f = pd.concat([df_tr_f,df_to_join_tr]).reset_index(drop=True)

    df_tr_f = not_in_impressions_filter(df_tr_f)
    df_tr_f = make_target(df_tr_f)
    df_val_f.drop([ 'is_last_step','targ','imp_list'], inplace=True, axis=1)
    df_tr_f.drop([ 'is_last_step','targ','imp_list'], inplace=True, axis=1)


    # #### compute normalizations

    var_list = ['time_since_prev_search_for_item_corrected',
                'time_since_prev_clickout_item_corrected',
                'time_since_prev_interaction_item_image_corrected',
                'time_since_prev_interaction_item_info_corrected',
                'time_since_prev_interaction_item_deals_corrected',
                'time_since_prev_interaction_item_rating_corrected',
                'time_since_prev_any_item_action_corrected',
                'time_since_prev_any_all_action_corrected']

    normalizations_dict = fh.get_means_and_stds_list(df_tr_=df_tr_f, df_val_ = df_val_f,var_list= var_list)

    normalizations_dict['seq_t1_corr'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'seq_t1_corr')

    normalizations_dict['seq_t1_corr']['means'] = means
    normalizations_dict['seq_t1_corr']['stds'] = stds
    normalizations_dict['seq_t1_corr']['max'] = maxv

    normalizations_dict['seq_t2_corr'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'seq_t2_corr')

    normalizations_dict['seq_t2_corr']['means'] = means
    normalizations_dict['seq_t2_corr']['stds'] = stds
    normalizations_dict['seq_t2_corr']['max'] = maxv

    normalizations_dict['seq_t3_corr'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'seq_t3_corr')

    normalizations_dict['seq_t3_corr']['means'] = means
    normalizations_dict['seq_t3_corr']['stds'] = stds
    normalizations_dict['seq_t3_corr']['max'] = maxv

    normalizations_dict['seq_time_new'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'seq_time_new')

    normalizations_dict['seq_time_new']['means'] = means
    normalizations_dict['seq_time_new']['stds'] = stds
    normalizations_dict['seq_time_new']['max'] = maxv

    normalizations_dict['seq_time_new_log'] = {}
    means, stds,maxv = fh.get_log_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'seq_time_new')

    normalizations_dict['seq_time_new_log']['means'] = means
    normalizations_dict['seq_time_new_log']['stds'] = stds
    normalizations_dict['seq_time_new_log']['max'] = maxv

    normalizations_dict['seq_step_time'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'seq_step_time')

    normalizations_dict['seq_step_time']['means'] = means
    normalizations_dict['seq_step_time']['stds'] = stds
    normalizations_dict['seq_step_time']['max'] = maxv

    normalizations_dict['seq_step_time_log'] = {}
    means, stds,maxv = fh.get_log_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'seq_step_time')

    normalizations_dict['seq_step_time_log']['means'] = means
    normalizations_dict['seq_step_time_log']['stds'] = stds
    normalizations_dict['seq_step_time_log']['max'] = maxv

    normalizations_dict['impressions_list_timesincelastaction'] = {}
    means, stds,maxv = fh.get_imp_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'impressions_list_timesincelastaction')

    normalizations_dict['impressions_list_timesincelastaction']['means'] = means
    normalizations_dict['impressions_list_timesincelastaction']['stds'] = stds
    normalizations_dict['impressions_list_timesincelastaction']['max'] = maxv

    normalizations_dict['impressions_list_timesincelastaction_log'] = {}
    means, stds,maxv = fh.get_log_imp_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'impressions_list_timesincelastaction')

    normalizations_dict['impressions_list_timesincelastaction_log']['means'] = means
    normalizations_dict['impressions_list_timesincelastaction_log']['stds'] = stds
    normalizations_dict['impressions_list_timesincelastaction_log']['max'] = maxv


    normalizations_dict['prices'] = {}
    means, stds,maxv = fh.get_prices_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'prices_list')

    normalizations_dict['prices']['means'] = means
    normalizations_dict['prices']['stds'] = stds
    normalizations_dict['prices']['max'] = maxv

    normalizations_dict['prices_log'] = {}
    means, stds,maxv = fh.get_log_prices_means_and_stds(df_tr_=df_tr_f, df_val_ = df_val_f, var_group = 'prices_list')

    normalizations_dict['prices_log']['means'] = means
    normalizations_dict['prices_log']['stds'] = stds
    normalizations_dict['prices_log']['max'] = maxv


    with open(FEATURES_PATH+OUTPUT_NORMLIZATIONS_VAL, 'wb') as handle:
        pickle.dump(normalizations_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # ### 1.4 df_test

    df_test_new = df_steps_test.merge(df_sessions, on='session_id')
    df_test_new_f, feat_group_dict = make_sess_df(df_test_new, enc_dict, dic_items,mlb_items)

    if not is_normal:
        df_to_join_train = all_except_last_filter(df_test_new_f)

    df_test_new_f = only_last_filter(df_test_new_f)
    df_test_new_f = make_target(df_test_new_f,is_test=True)
    df_train_new = df_steps_train.merge(df_sessions, on='session_id')
    df_train_new_f, feat_group_dict = make_sess_df(df_train_new, enc_dict, dic_items,mlb_items)

    if not is_normal:
        df_train_new_f = pd.concat([df_train_new_f,df_to_join_train]).reset_index(drop=True)

    df_train_new_f = not_in_impressions_filter(df_train_new_f)
    df_train_new_f = make_target(df_train_new_f)
    df_test_new_f.drop([ 'is_last_step','targ','imp_list'], inplace=True, axis=1)
    df_train_new_f.drop([ 'is_last_step','targ','imp_list'], inplace=True, axis=1)


    # #### Compute normalizations


    var_list = ['time_since_prev_search_for_item_corrected',
                'time_since_prev_clickout_item_corrected',
                'time_since_prev_interaction_item_image_corrected',
                'time_since_prev_interaction_item_info_corrected',
                'time_since_prev_interaction_item_deals_corrected',
                'time_since_prev_interaction_item_rating_corrected',
                'time_since_prev_any_item_action_corrected',
                'time_since_prev_any_all_action_corrected']

    normalizations_dict = fh.get_means_and_stds_list(df_tr_=df_train_new_f, df_val_ = df_test_new_f,var_list= var_list)

    normalizations_dict['seq_t1_corr'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'seq_t1_corr')

    normalizations_dict['seq_t1_corr']['means'] = means
    normalizations_dict['seq_t1_corr']['stds'] = stds
    normalizations_dict['seq_t1_corr']['max'] = maxv

    normalizations_dict['seq_t2_corr'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'seq_t2_corr')

    normalizations_dict['seq_t2_corr']['means'] = means
    normalizations_dict['seq_t2_corr']['stds'] = stds
    normalizations_dict['seq_t2_corr']['max'] = maxv

    normalizations_dict['seq_t3_corr'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'seq_t3_corr')

    normalizations_dict['seq_t3_corr']['means'] = means
    normalizations_dict['seq_t3_corr']['stds'] = stds
    normalizations_dict['seq_t3_corr']['max'] = maxv

    normalizations_dict['seq_time_new'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'seq_time_new')

    normalizations_dict['seq_time_new']['means'] = means
    normalizations_dict['seq_time_new']['stds'] = stds
    normalizations_dict['seq_time_new']['max'] = maxv


    normalizations_dict['seq_time_new_log'] = {}
    means, stds,maxv = fh.get_log_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'seq_time_new')

    normalizations_dict['seq_time_new_log']['means'] = means
    normalizations_dict['seq_time_new_log']['stds'] = stds
    normalizations_dict['seq_time_new_log']['max'] = maxv


    normalizations_dict['seq_step_time'] = {}
    means, stds,maxv = fh.get_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'seq_step_time')

    normalizations_dict['seq_step_time']['means'] = means
    normalizations_dict['seq_step_time']['stds'] = stds
    normalizations_dict['seq_step_time']['max'] = maxv

    normalizations_dict['seq_step_time_log'] = {}
    means, stds,maxv = fh.get_log_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'seq_step_time')

    normalizations_dict['seq_step_time_log']['means'] = means
    normalizations_dict['seq_step_time_log']['stds'] = stds
    normalizations_dict['seq_step_time_log']['max'] = maxv


    normalizations_dict['impressions_list_timesincelastaction'] = {}
    means, stds,maxv = fh.get_imp_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'impressions_list_timesincelastaction')

    normalizations_dict['impressions_list_timesincelastaction']['means'] = means
    normalizations_dict['impressions_list_timesincelastaction']['stds'] = stds
    normalizations_dict['impressions_list_timesincelastaction']['max'] = maxv

    normalizations_dict['impressions_list_timesincelastaction_log'] = {}
    means, stds,maxv = fh.get_log_imp_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'impressions_list_timesincelastaction')

    normalizations_dict['impressions_list_timesincelastaction_log']['means'] = means
    normalizations_dict['impressions_list_timesincelastaction_log']['stds'] = stds
    normalizations_dict['impressions_list_timesincelastaction_log']['max'] = maxv


    normalizations_dict['prices'] = {}
    means, stds,maxv = fh.get_prices_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'prices_list')

    normalizations_dict['prices']['means'] = means
    normalizations_dict['prices']['stds'] = stds
    normalizations_dict['prices']['max'] = maxv

    normalizations_dict['prices_log'] = {}
    means, stds,maxv = fh.get_log_prices_means_and_stds(df_tr_=df_train_new_f, df_val_ = df_test_new_f, var_group = 'prices_list')

    normalizations_dict['prices_log']['means'] = means
    normalizations_dict['prices_log']['stds'] = stds
    normalizations_dict['prices_log']['max'] = maxv


    with open(FEATURES_PATH+OUTPUT_NORMLIZATIONS_SUBM, 'wb') as handle:
        pickle.dump(normalizations_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # ### 2. Save
    df_tr_f.to_csv(FEATURES_PATH+'new_X_tr_f02.gz', index=False,compression='gzip')
    df_val_f.to_csv(FEATURES_PATH+'new_X_val_f02.gz', index=False,compression='gzip')

    del df_val_f

    df_train_new_f.to_csv(FEATURES_PATH+'new_X_train_f02.gz', index=False,compression='gzip')
    df_test_new_f.to_csv(FEATURES_PATH+'new_X_test_f02.gz', index=False,compression='gzip')

if __name__ == "__main__":
    main()
