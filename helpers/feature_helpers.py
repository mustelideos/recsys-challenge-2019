import pandas as pd
from collections import Counter
import numpy as np
import itertools

def add_filter_properties_features(df_, dic_items, mlb):
    item2proplist = dic_items['item2proplist']
    item2propcount = dic_items['item2propcount']
    #df_ = df#.copy()
    new_vars = ['count_curr_filters_in_item_properties', 'count_curr_filters', 'count_item_properties',
                'avg_count_item_properties',
                'proportion_curr_filters_in_item_properties', 'average_proportion_curr_filters_in_item_properties',
                'curr_filters_in_item_properties_hot_enc', 'item_properties_hot_enc']
    idx = (df_.action_type=='clickout item')
    df_test = df_.loc[idx].copy()#.reset_index(drop=True)

    print ('getting impressions_list.11')
    df_test['impressions_list'] = df_test.impressions.fillna('').str.split('|') \
                                            .apply(lambda s: [i for i in s if i!=''])

    print ('getting current_filters_list.10')
    df_test['current_filters_list'] = df_test.current_filters.fillna('').str.split('|') \
                                            .apply(lambda s: [i for i in s if i!=''])

    print ('computing curr_filters_in_item_properties.9')
    df_test['curr_filters_in_item_properties'] = \
            df_test.apply(lambda row: [(set(item2proplist.get(int(s),[]))&set(row.current_filters_list)) \
                                                        for s in row.impressions_list], axis=1)

    print ('computing count_curr_filters_in_item_properties.8')
    df_test['count_curr_filters_in_item_properties'] = \
                df_test.curr_filters_in_item_properties.apply(lambda s: [len(i) for i in s])

    print ('computing count_curr_filters.7')
    df_test['count_curr_filters'] = df_test.current_filters_list.apply(list).apply(len)

    print ('computing count_item_properties.6')
    df_test['count_item_properties'] = df_test.apply(lambda row: [item2propcount.get(int(s),0) for s in row.impressions_list], axis=1)

    print ('computing avg_count_item_properties.5')
    df_test['avg_count_item_properties'] = df_test.count_item_properties.apply(np.mean)

    print ('computing proportion_curr_filters_in_item_properties.4')
    df_test['proportion_curr_filters_in_item_properties'] = \
    df_test.apply(lambda row: [round(s/row.count_curr_filters, 2) \
                                   if row.count_curr_filters>0 else 0 for s in row.count_curr_filters_in_item_properties], axis=1)

    print ('computing average_proportion_curr_filters_in_item_properties.3')
    df_test['average_proportion_curr_filters_in_item_properties'] = \
            df_test.proportion_curr_filters_in_item_properties.apply(np.mean)

    print ('computing curr_filters_in_item_properties_hot_enc.2')
    df_test['curr_filters_in_item_properties_hot_enc'] = df_test.curr_filters_in_item_properties.apply(mlb.transform)

    print ('computing item_properties_hot_enc.1')
    df_test['item_properties_hot_enc'] = df_test.impressions_list \
                            .apply(lambda s: [item2proplist.get(int(i),[]) for i in s]).apply(mlb.transform)
    for v in new_vars:
        df_[v]= np.nan
        df_.loc[idx, v]= df_test[v]
    return df_


def get_time_since_prev(df, new_var_name = None,
                            action_list = [], nan_val = -1, max_val = None):

    idx = df.action_type.isin(action_list)

    df.loc[idx, new_var_name] = df.loc[idx].groupby('session_id', sort=False).timestamp.diff()

    df[new_var_name] = df.groupby('session_id')[new_var_name].fillna(method='ffill')

    idx_nan = df[new_var_name].isnull()

    df[new_var_name+'_isnull'] = idx_nan.astype(int)

    if max_val:
        df.loc[~idx_nan, new_var_name]  = df.loc[~idx_nan, new_var_name].apply(lambda s: min(max_val, s))

    df[new_var_name] = df[new_var_name].fillna(nan_val)

    if max_val:
        df[new_var_name] = df[new_var_name]/max_val
    return df


def add_impressions_list_time_features(df, action_list = [],
                                           enc_dict=None,
                                           var_name_action='impressions_list_lastaction_enc',
                                           var_name_time='impressions_list_timesincelastaction',
                                           nan_val=-1):
    """USAGE:
    df = fh.add_impressions_list_time_features(df, action_list = item_actions,
                                           enc_dict=enc_dict['action_type'],
                                           var_name_action='impressions_list_lastaction_enc',
                                           var_name_time='impressions_list_timesincelastaction')
                                           """
    if 'impressions_list' not in df.columns:
        df['impressions_list'] = df.impressions.str.split('|')

    idx_items = (df.action_type.isin(action_list))
    df.loc[idx_items, 'ref_action_pair'] = df.loc[idx_items].apply(lambda row: [[row.reference, row.action_type]], axis=1)
    df.loc[idx_items, 'ref_timestamp_pair'] = df.loc[idx_items].apply(lambda row: [[row.reference, row.timestamp]], axis=1)

    df.loc[idx_items, 'lastaction'] = df.loc[idx_items].groupby('session_id').ref_action_pair.transform(pd.Series.cumsum)
    df.loc[idx_items, 'lastaction'] = df.loc[idx_items].groupby('session_id')['lastaction'].shift()

    df.loc[idx_items, 'lastaction_timestamp'] = df.loc[idx_items].groupby('session_id').ref_timestamp_pair.transform(pd.Series.cumsum)#.shift()
    df.loc[idx_items, 'lastaction_timestamp'] = df.loc[idx_items].groupby('session_id')['lastaction_timestamp'].shift()

    idx = (df.action_type=='clickout item')
    df.loc[idx,'item2lastaction'] = df.loc[idx, 'lastaction'].fillna('').apply(lambda s: dict(s))
    df.loc[idx,'item2lastaction_timestamp'] = df.loc[idx].fillna('').apply(list).apply(lambda row: \
                                                [[a[0], row.timestamp-a[1]] for a in row.lastaction_timestamp],
                                                        axis=1).apply(lambda s: dict(s))
    df.drop(['ref_action_pair', 'ref_timestamp_pair', 'lastaction', 'lastaction_timestamp'], axis=1, inplace=True)

    df.loc[idx, var_name_time] = df.loc[idx].apply(lambda row: [row['item2lastaction_timestamp'].get(s, nan_val) for s in row['impressions_list']], axis=1)
    df.loc[idx, var_name_action] = df.loc[idx].apply(lambda row: [enc_dict.get(row['item2lastaction'] \
                                                                        .get(s, '0')) for s in row['impressions_list']], axis=1)

    df.drop(['item2lastaction', 'item2lastaction_timestamp'], axis=1, inplace=True)

    return df


def get_past_impressions_session(df, enc_dict=None, do_sort=False, do_enc=True):
    """ returns impressions shown to user in the past in that session"""
    VAR_NAME = 'past_impressions_session'
    VAR_GROUPBY = 'session_id'
    idx = df.action_type=='clickout item'
    df_ = df.loc[idx, ['session_id', 'step', 'impressions']].copy()
    if do_sort:
        df_ = df_.sort_values(['session_id', 'step']).reset_index(drop=True)
    df_['impressions_list'] = df_.impressions.fillna('').str.split('|')
    df_['impressions_list_shift']= df_.groupby(VAR_GROUPBY, sort=False).impressions_list.shift().fillna('').apply(list)
    if do_enc:
        df_['impressions_list_shift']= df_.impressions_list_shift.apply(lambda s: [enc_dict.get(i) for i in s])
    df_[VAR_NAME] = df_.groupby(VAR_GROUPBY).impressions_list_shift \
                                                          .transform(pd.Series.cumsum) \
                                                          .apply(lambda s: dict(Counter(s)))
    _ = df_[VAR_NAME].apply(lambda s: s.pop('', None))
    # idx = df_[VAR_NAME].apply(lambda s: len(s))>0
    # df_=df_.loc[idx].reset_index(drop=True)
    return df_[['session_id', 'step', VAR_NAME]]


# def get_mean_std(df = None, feat_dict=None, feature_group=None, feature_name=None, exclude_vals = []):
#     loc_feat = feat_dict[feature_group].index(feature_name)
#     aux= df[feature_group].apply(lambda s: [i[loc_feat] for i in s])
#     all_values = list(itertools.chain.from_iterable(aux))
#     all_values = [i for i in all_values if i not in exclude_vals]
#     return np.mean(all_values), np.std(all_values)

def get_means_and_stds(df_tr_=None, df_val_ = None, var_group = 'seq_num_new'):
    aux = pd.concat([df_tr_[[var_group]], df_val_[[var_group]]], axis=0).reset_index(drop=True)[var_group]
    
    array = np.array(list(itertools.chain.from_iterable(aux)))
    array[array==-1] = np.nan
    
    means = np.nanmean(array, axis=0)
    means = means[:np.newaxis].T
    
    stds = np.nanstd(array, axis=0)
    stds = stds[:np.newaxis].T
    
    maxv = np.nanmax(array, axis=0)
    maxv = maxv[:np.newaxis].T
    #How to normalize:
    #     df_f2['test_normalized'] = df_f2[feature_group].apply(lambda s: (s-means)/stds)
    #     np.mean(list(itertools.chain.from_iterable(df_f['test_normalized'])), axis=0)
    #     np.std(list(itertools.chain.from_iterable(df_f['test_normalized'])), axis=0)
    return means, stds, maxv


def get_prices_means_and_stds(df_tr_=None, df_val_ = None, var_group = 'prices'):
    
    aux = pd.concat([df_tr_[[var_group]], df_val_[[var_group]]], axis=0).reset_index(drop=True)[var_group]
    all_prices =  np.array(list(itertools.chain.from_iterable(aux)),dtype='int32')
    means = np.mean(all_prices)
    stds = np.std(all_prices)
    maxv = np.max(all_prices, axis=0)

    return means, stds, maxv


def get_log_prices_means_and_stds(df_tr_=None, df_val_ = None, var_group = 'prices'):
    
    aux = pd.concat([df_tr_[[var_group]], df_val_[[var_group]]], axis=0).reset_index(drop=True)[var_group]

    all_prices =  np.array(list(itertools.chain.from_iterable(aux)),dtype='int32')
    
    means = np.mean(np.log(all_prices+1), axis=0)
    stds = np.std(np.log(all_prices+1), axis=0)
    maxv = np.max(np.log(all_prices+1), axis=0)

    return means, stds, maxv

def get_log_means_and_stds(df_tr_=None, df_val_ = None, var_group = 'seq_num_new'):
    aux = pd.concat([df_tr_[[var_group]], df_val_[[var_group]]], axis=0).reset_index(drop=True)[var_group]
    
    array = np.array(list(itertools.chain.from_iterable(aux)))
    array[array==-1] = np.nan

    means = np.nanmean(np.log(array+1.9), axis=0)
    means = means[:np.newaxis].T
    
    stds = np.nanstd(np.log(array+1.9), axis=0)
    stds = stds[:np.newaxis].T
    
    maxv = np.nanmax(np.log(array+1.9), axis=0)
    maxv = maxv[:np.newaxis].T    
    #How to normalize:
    #     df_f2['test_normalized'] = df_f2[feature_group].apply(lambda s: (s-means)/stds)
    #     np.mean(list(itertools.chain.from_iterable(df_f['test_normalized'])), axis=0)
    #     np.std(list(itertools.chain.from_iterable(df_f['test_normalized'])), axis=0)
    return means, stds, maxv

def get_imp_means_and_stds(df_tr_=None, df_val_ = None, var_group = 'seq_num_new'):
    aux = pd.concat([df_tr_[[var_group]], df_val_[[var_group]]], axis=0).reset_index(drop=True)[var_group]
    
    lista=list(itertools.chain.from_iterable(aux))
    listasemnan = [s for s in lista if s!=-1]
    means = np.mean(listasemnan)
    
    stds = np.std(listasemnan)
    maxv = np.max(listasemnan)
    #How to normalize:
    #     df_f2['test_normalized'] = df_f2[feature_group].apply(lambda s: (s-means)/stds)
    #     np.mean(list(itertools.chain.from_iterable(df_f['test_normalized'])), axis=0)
    #     np.std(list(itertools.chain.from_iterable(df_f['test_normalized'])), axis=0)
    return means, stds, maxv

def get_log_imp_means_and_stds(df_tr_=None, df_val_ = None, var_group = 'seq_num_new'):
    aux = pd.concat([df_tr_[[var_group]], df_val_[[var_group]]], axis=0).reset_index(drop=True)[var_group]
    
    lista=list(itertools.chain.from_iterable(aux))
    listasemnan = np.log(np.array([s for s in lista if s!=-1])+1.9)
    means = np.mean(listasemnan)
    
    stds = np.std(listasemnan)
    maxv = np.max(listasemnan)

    #How to normalize:
    #     df_f2['test_normalized'] = df_f2[feature_group].apply(lambda s: (s-means)/stds)
    #     np.mean(list(itertools.chain.from_iterable(df_f['test_normalized'])), axis=0)
    #     np.std(list(itertools.chain.from_iterable(df_f['test_normalized'])), axis=0)
    return means, stds,maxv

def get_means_and_stds_list(df_tr_=None, df_val_ = None, var_list = [], exclude_list=[-1]):
    norm_dic = dict()
    for col in var_list:
        norm_dic[col] = dict()
        norm_dic[col+'_log'] = dict()

        aux = pd.concat([df_tr_[[col]], df_val_[[col]]], axis=0).reset_index(drop=True)
        aux = aux.loc[~aux[col].isin(exclude_list)][col]
        
        norm_dic[col]['mean'] = aux.mean()
        norm_dic[col]['std'] = aux.std()
        norm_dic[col]['max'] = aux.max()

        norm_dic[col+'_log']['mean'] = np.log(aux+1.9).mean()
        norm_dic[col+'_log']['std'] = np.log(aux+1.9).std()    
        norm_dic[col+'_log']['max'] = np.log(aux+1.9).max()        

    return norm_dic




def get_time_since_prev_corrected(df, action_list=[],
                                    enc_dict=None, var_name=None, nan_val=-1):
    """USAGE:
    for act in item_actions:
    df = fh.get_time_since_prev_corrected(df, action_list=[act],
                                        enc_dict=enc_dict['reference'], var_name=act.replace(' ', '_'))
    df = fh.get_time_since_prev_corrected(df, action_list=item_actions,
                                     enc_dict=enc_dict['reference'], var_name='any_item_action')
                                     """
    var_list = action_list

    if not var_name:
        var_name = var_list[0].lower().replace(' ', '_')+'_corrected'

    idx = df.action_type.isin(var_list)

    time_var_name = 'time_since_prev_' + var_name
    reference_var_name = 'prev_ref_' + var_name + '_enc'
    print (time_var_name)
    print (reference_var_name)

    df.loc[idx, 'var_aux_time'] = df.loc[idx].timestamp
    df.loc[idx, 'var_aux_ref'] = df.loc[idx].reference
    df['var_aux_time'] = df.groupby('session_id')['var_aux_time'].shift()
    df['var_aux_ref'] = df.groupby('session_id')['var_aux_ref'].shift()
    df['var_aux_time'] = df.groupby('session_id')['var_aux_time'].fillna(method='ffill')
    df[reference_var_name] = df.groupby('session_id')['var_aux_ref'].fillna(method='ffill').fillna('0')
    df[reference_var_name] = df[reference_var_name].apply(lambda s: enc_dict[s])

    df[time_var_name] = df.apply(lambda row: row.timestamp-row['var_aux_time'], axis=1).fillna(nan_val)
    df.drop(['var_aux_time', 'var_aux_ref'], axis=1, inplace=True)
    return df


def get_past_impressions_user(df, enc_dict=None, do_sort=False, do_enc=True):
    VAR_NAME = 'past_impressions_user'
    VAR_GROUPBY = 'user_id'
    """ returns impressions shown to user in the past, should be applied to the whole data """
    idx = df.action_type=='clickout item'
    df_ = df.loc[idx, ['user_id', 'day', 'session_id', 'step', 'impressions']].reset_index(drop=True).copy()
    if do_sort:
        df_ = df_.sort_values(['user_id', 'day','session_id', 'step']).reset_index(drop=True)
    df_['impressions_list'] = df_.impressions.fillna('').str.split('|')
    df_['impressions_list_shift']= df_.groupby(VAR_GROUPBY, sort=False).impressions_list.shift().fillna('').apply(list)
    if do_enc:
        df_['impressions_list_shift']= df_.impressions_list_shift.apply(lambda s: [enc_dict.get(i) for i in s])
    df_[VAR_NAME] = df_.groupby(VAR_GROUPBY).impressions_list_shift \
                                                          .transform(pd.Series.cumsum) \
                                                          .apply(lambda s: dict(Counter(s)))
    _ = df_[VAR_NAME].apply(lambda s: s.pop('', None))
    # idx = df_[VAR_NAME].apply(lambda s: len(s))>0
    # df_=df_.loc[idx].reset_index(drop=True)
    return df_[['session_id', 'step', VAR_NAME]]


def get_past_impressions(df, enc_dict_ref=None, do_sort=False, do_enc=True, VAR_GROUPBY=None, feature_name=None):
    VAR_NAME = feature_name
    #VAR_GROUPBY = 'user_id'
    """ returns impressions shown to user in the past, should be applied to the whole data """
    idx = df.action_type=='clickout item'
    df_ = df.loc[idx, ['user_id', 'day', 'session_id', 'step', 'impressions']].reset_index(drop=True).copy()
    if do_sort:
        df_ = df_.sort_values(['user_id', 'day','session_id', 'step']).reset_index(drop=True)
    df_['impressions_list'] = df_.impressions.fillna('').str.split('|')
    df_['impressions_list_shift']= df_.groupby(VAR_GROUPBY, sort=False).impressions_list.shift().fillna('').apply(list)
    if do_enc:
        df_['impressions_list_shift']= df_.impressions_list_shift.apply(lambda s: [enc_dict_ref.get(i) for i in s])
    df_[VAR_NAME] = df_.groupby(VAR_GROUPBY).impressions_list_shift \
                                                          .transform(pd.Series.cumsum) \
                                                          .apply(lambda s: dict(Counter(s)))
    _ = df_[VAR_NAME].apply(lambda s: s.pop('', None))
    # idx = df_[VAR_NAME].apply(lambda s: len(s))>0
    # df_=df_.loc[idx].reset_index(drop=True)
    return df_[['session_id', 'step', VAR_NAME]]

def make_impressions_features(df, enc_dict=None, df_all_imp_list=None, VAR_GROUPBY = None):
    FEATURE_NAME = 'pi_%s' % (VAR_GROUPBY)
    df_feat = get_past_impressions(df, enc_dict_ref=enc_dict['reference'],
                                          VAR_GROUPBY='user_id', feature_name=FEATURE_NAME)
    df_feat = df_all_imp_list.merge(df_feat, on=['session_id', 'step'])
    df_feat[FEATURE_NAME] = \
        df_feat.apply(lambda row: [row[FEATURE_NAME].get(s,0) for s in row.impressions_list_enc], axis=1)
    df_feat = df_feat[['session_id', 'step', FEATURE_NAME]]
    return df_feat


def make_all_interaction_features(df, action_list, enc_dict=None, df_all_imp_list=None, VAR_GROUPBY = None):
    action = 'clickout item'
    print (action)
    feature_name = 'pa_%s_%s' % (VAR_GROUPBY, action.replace(' ', '_'))
    df_feat_all = make_interaction_features(feature_name, df, ['clickout item'],
                                            enc_dict=enc_dict, df_all_imp_list=df_all_imp_list,
                                            VAR_GROUPBY = VAR_GROUPBY)
    for action in set(action_list)-set(['clickout item']):
        print (action)
        feature_name = 'pa_%s_%s' % (VAR_GROUPBY, action.replace(' ', '_'))
        df_feat = make_interaction_features(feature_name, df, [action],
                                                enc_dict=enc_dict, df_all_imp_list=df_all_imp_list,
                                                VAR_GROUPBY = VAR_GROUPBY)
        df_feat_all = df_feat_all.merge(df_feat, on=['session_id', 'step'])
    return df_feat_all


def make_interaction_features(FEATURE_NAME, df, action_type, enc_dict=None, df_all_imp_list=None, VAR_GROUPBY = None):

    df_feat = get_past_action_any_interactions(df, enc_dict_ref=enc_dict['reference'], action_list=action_type,
                                                         feature_name = FEATURE_NAME,
                                                         VAR_GROUPBY = VAR_GROUPBY)
    df_feat = df_all_imp_list.merge(df_feat, on=['session_id', 'step'])
    df_feat[FEATURE_NAME] = \
            df_feat.apply(lambda row: [row[FEATURE_NAME].get(s,0) for s in row.impressions_list_enc], axis=1)
    df_feat = df_feat[['session_id', 'step', FEATURE_NAME]]
    return df_feat


def get_past_action_any_interactions(df, enc_dict_ref=None, action_list=[], feature_name=None, VAR_GROUPBY = None):
    #VAR_NAME = 'past_interated_user_all_interactions'
    #VAR_GROUPBY = 'user_id'
    VAR_NAME = feature_name
    """ returns references interacted by user in the past"""
    #action_list = [a for a in df.action_type.unique() if 'item' in a]
    idx_action = df.action_type.isin(action_list+['clickout item'])
    idx_click = df.action_type.isin(['clickout item'])

    df_ = df.loc[idx_action, ['user_id', 'day', 'session_id', 'step', 'reference']].copy()#.reset_index(drop=True).copy()

    if 'clickout item' not in action_list:
        df_.loc[idx_click, 'reference']='-2'

    df_['reference_shift']= df_.groupby(VAR_GROUPBY, sort=False).reference.shift().fillna('').str.split('|')
    #print (VAR_GROUPBY)
    df_['reference_shift_enc']= df_.reference_shift.apply(lambda s: [enc_dict_ref.get(i) for i in s])
    df_[VAR_NAME] = df_.groupby(VAR_GROUPBY, sort=False).reference_shift_enc \
                                            .transform(pd.Series.cumsum) \
                                            .apply(lambda s: dict(Counter(s)))
    _ = df_[VAR_NAME].apply(lambda s: s.pop(None, None))
    return df_.loc[idx_click, ['session_id', 'step',  VAR_NAME]].reset_index(drop=True)


def get_past_interated_user_all_interactions(df, enc_dict=None, action_list=[]):
    VAR_NAME = 'past_interated_user_all_interactions'
    VAR_GROUPBY = 'user_id'
    """ returns references interacted by user in the past"""
    #action_list = [a for a in df.action_type.unique() if 'item' in a]
    idx_action = df.action_type.isin(action_list)
    idx_click = df.action_type.isin(['clickout item'])
    df_ = df.loc[idx_action, ['user_id', 'day', 'session_id', 'step', 'reference']]#.reset_index(drop=True).copy()

    df_['reference_shift']= df_.groupby(VAR_GROUPBY, sort=False).reference.shift().fillna('').str.split('|')

    df_['reference_shift_enc']= df_.reference_shift.apply(lambda s: [enc_dict.get(i) for i in s])
    df_[VAR_NAME] = df_.groupby(VAR_GROUPBY, sort=False).reference_shift_enc \
                                            .transform(pd.Series.cumsum) \
                                            .apply(lambda s: dict(Counter(s)))
    _ = df_[VAR_NAME].apply(lambda s: s.pop(None, None))

    return df_.loc[idx_click, ['session_id', 'step',  VAR_NAME]].reset_index(drop=True)

def get_past_clicked_user(df, enc_dict=None, do_sort=False, do_enc=True):
    VAR_NAME = 'past_clicked_user'
    VAR_GROUPBY = 'user_id'
    """ returns references clicked by user in the past"""
    idx = df.action_type=='clickout item'
    df_ = df.loc[idx, ['user_id', 'day', 'session_id', 'step', 'reference']].reset_index(drop=True).copy()
    if do_sort:
        df_ = df_.sort_values(['user_id', 'day','session_id', 'step']).reset_index(drop=True)

    df_['reference_shift']= df_.groupby(VAR_GROUPBY, sort=False).reference.shift().fillna('').str.split('|')
    if do_enc:
        df_['reference_shift_enc']= df_.reference_shift.apply(lambda s: [enc_dict.get(i) for i in s])
        df_[VAR_NAME] = df_.groupby(VAR_GROUPBY, sort=False).reference_shift_enc \
                                                .transform(pd.Series.cumsum) \
                                                .apply(lambda s: dict(Counter(s)))
        _ = df_[VAR_NAME].apply(lambda s: s.pop(None, None))
    else:
        df_[VAR_NAME] = df_.groupby(VAR_GROUPBY, sort=False).reference_shift \
                                                .transform(pd.Series.cumsum) \
                                                .apply(lambda s: dict(Counter(s)))
        _ = df_[VAR_NAME].apply(lambda s: s.pop('', None))

    # idx = df_[VAR_NAME].apply(lambda s: len(s))>0
    # df_=df_.loc[idx].reset_index(drop=True)
    return df_[['session_id', 'step',  VAR_NAME]]


enc_dicts = {}

day_name_enc_dict = {'Thursday':4, 'Tuesday':2, 'Sunday':0, 'Saturday':6, 'Friday':5, 'Monday':1,
       'Wednesday':3}
enc_dicts['day_name'] = day_name_enc_dict


day_enc_dict = dict([(e,i) for i,e in enumerate(range(1,9))])
enc_dicts['day'] = day_enc_dict

enc_dicts['stars'] = dict([(-1,0),(1,1),(2,2),(3,3),(4,4),(5,5)])
enc_dicts['from_stars'] = dict([(-1,0),(2,2),(3,3),(4,4)])
enc_dicts['rating'] = dict([(-1,0),(1,1),(2,2),(3,3),(4,4)])



platform2country = {'AU': 'Australia',
 'BR': 'Brazil',
 'FI': 'Finland',
 'UK': 'United Kingdom',
 'US': 'USA',
 'MX': 'Mexico',
 'FR': 'France',
 'IT': 'Italy',
 'AT': 'Austria',
 'HK': 'Hong Kong',
 'RU': 'Russia',
 'IN': 'India',
 'CO': 'Colombia',
 'ES': 'Spain',
 'CL': 'Chile',
 'CH': 'Switzerland',
 'BE': 'Belgium',
 'AR': 'Argentina',
 'NL': 'Netherlands',
 'CA': 'Canada',
 'JP': 'Japan',
 'IE': 'Ireland',
 'SE': 'Sweden',
 'DE': 'Germany',
 'TH': 'Thailand',
 'MY': 'Malaysia',
 'HU': 'Hungary',
 'PH': 'Philippines',
 'ZA': 'South Africa',
 'PE': 'Peru',
 'ID': 'Indonesia',
 'NZ': 'New Zealand',
 'CZ': 'Czech Republic',
 'KR': 'South Korea',
 'RS': 'Serbia',
 'BG': 'Bulgaria',
 'DK': 'Denmark',
 'HR': 'Croatia',
 'TR': 'Turkey',
 'IL': 'Israel',
 'SG': 'Singapore',
 'EC': 'Ecuador',
 'SK': 'Slovakia',
 'PL': 'Poland',
 'NO': 'Norway',
 'AA': 'Saudi Arabia',
 'TW': 'Taiwan',
 'PT': 'Portugal',
 'RO': 'Romania',
 'UY': 'Uruguay',
 'GR': 'Greece',
 'AE': 'United Arab Emirates',
 'SI': 'Slovenia',
 'CN': 'China',
 'VN': 'Vietnam'}
