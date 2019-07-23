#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../")

import pandas as pd
import pathlib
import numpy as np
import itertools,pickle
from collections import Counter
import argparse
import logging
import os
import helpers.feature_helpers as fh


DATA_ITEMS_PATH = pathlib.Path('../data/item_metadata.csv')

OUTPUT_ENCODING_DICT = 'enc_dicts_v02.pkl'

DEFAULT_FEATURES_DIR_NAME = 'nn_vnormal'
DEFAULT_PREPROC_DIR_NAME = 'data_processed_vnormal'

def setup_args_parser():
    parser = argparse.ArgumentParser(description='Create item features and encodings')
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
    logger.info('Running 011_Features_Items.py')
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
    os.makedirs(FEATURES_PATH) if not os.path.exists(FEATURES_PATH) else None
    logger.info('features path: %s' % FEATURES_PATH)
    # End of set up arguments


    df_items = pd.read_csv(DATA_ITEMS_PATH)

    df_items['properties_list'] = df_items.properties.str.split('|')
    all_properties = set(itertools.chain.from_iterable(df_items.properties_list.tolist()))


    df_items['stars']=df_items.properties_list \
                    .apply(lambda s: [u.split(' ')[0] for i,u in enumerate(s) \
                    if ('star' in u.lower() and 'from' not in u.lower())]) \
                    .apply(list).apply(lambda s: s[0] if len(s)>0 else -1) \
                    .astype(int)



    df_items['from_stars_list'] = df_items.properties_list \
                    .apply(lambda s: [int(u.split(' ')[1]) for i,u in enumerate(s) \
                    if ('from' in u.lower())]).apply(lambda s: np.sort(s))



    df_items['from_stars']=df_items['from_stars_list'].apply(lambda s: max(s) if len(s)>0 else -1).astype(int)
    df_items.drop('from_stars_list',axis=1, inplace=True)
    df_items.groupby(['stars', 'from_stars']).item_id.count()


    rating2number = {'Satisfactory Rating':1, 'Good Rating':2, 'Very Good Rating':3, 'Excellent Rating':4}


    df_items.properties_list.apply(lambda s: np.sort([rating2number.get(u) for i,u in enumerate(s) \
                                if ('rating' in u.lower())])).apply(str).value_counts()


    df_items['rating'] = df_items.properties_list.apply(lambda s: [rating2number.get(u) for i,u in enumerate(s)                          if ('rating' in u.lower())])                     .apply(lambda s: max(s) if len(s)>0 else -1)


    def compute_enc(df):

        pad_values_list = ['0']
        aux_tr = df.properties.dropna().str.split('|')
        all_set =  set(itertools.chain.from_iterable(aux_tr))

        all_set =  pad_values_list + list(all_set-set(pad_values_list))

        dec_dic = dict(enumerate(all_set))
        enc_dic = {x:k for k,x in enumerate(all_set)}

        return enc_dic, dec_dic


    enc_dic, dec_dic = compute_enc(df_items)
    enc_dict = pickle.load(open(DATA_PATH+OUTPUT_ENCODING_DICT, "rb" ))


    df_items['prop_list_enc0'] = df_items['properties_list'].apply(lambda list_s: [enc_dic[v] for v in list_s if v])
    df_items['prop_list_enc'] = df_items['properties_list'].apply(lambda list_s: [enc_dict['reference'][v] for v in list_s if v])

    df_items['n_prop'] = np.log(df_items.prop_list_enc.apply(len))
    df_items.groupby(['stars', 'rating']).item_id.count()

    df_items['item_id_enc'] = df_items['item_id'].astype(str).map(enc_dict['reference'])

    df_items = df_items.dropna()
    df_items.drop(['properties','properties_list'], inplace=True, axis=1)

    df_items['stars_enc']=df_items['stars'].map(fh.enc_dicts['stars']).astype(int)
    df_items['rating_enc']=df_items['rating'].map(fh.enc_dicts['rating']).astype(int)
    df_items['from_stars_enc']=df_items['from_stars'].map(fh.enc_dicts['from_stars']).astype(int)

    df_items['is_stars_nan']=df_items['stars']==-1
    df_items['is_rating_nan']=df_items['rating']==-1
    df_items['is_from_stars_nan']=df_items['rating']==-1

    df_items['item_id_enc'] = df_items['item_id_enc'].apply(int)
    df_items['stars'] = df_items['stars']/5
    df_items['from_stars'] = df_items['from_stars']/5
    df_items['rating'] = df_items['rating']/5
    df_items['n_prop'] = (df_items['n_prop']- df_items['n_prop'].mean())/df_items['n_prop'].std()


    df_items.to_csv(FEATURES_PATH+'Item_Features.csv', index=False)


if __name__ == "__main__":
    main()
