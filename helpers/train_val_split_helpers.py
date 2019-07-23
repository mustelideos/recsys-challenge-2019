import pandas as pd
import itertools

IS_NAN_VAL = 'is_nan_val'
IS_NAN_CLICKOUT_VAL = 'is_nan_clickout_val'
IS_NAN_REF = 'is_null_ref'
IS_SESSION_WITH_NANS = 'is_session_with_nans'
IS_SESSION_WITH_CLICKOUT_NANS = 'is_session_with_clickout_nans'
IS_PRE_CLICKOUT_NAN_SESSION = 'is_pre_clickout_nan_session'
IS_POST_CLICKOUT_NAN_SESSION = 'is_post_clickout_nan_session'


def add_date(df):
    df_out = df.copy()
    """to add date columns to dataframe. we can comment out the ones we don't need"""
    df_out['datetime'] = pd.to_datetime(df_out['timestamp'], unit='s')
    #df_out['date'] = df_out['datetime'].dt.date
    #df_out['day_name'] = df_out['datetime'].dt.day_name()
    df_out['day'] = df_out['datetime'].dt.day
    return df_out


def save_reference_true(df):
    df['reference_true']  = df.loc[:, 'reference']
    return df


def load_df(path, fill_search_for_destination=False, fill_filter_selection=False):
    """ loads dataframe and adds date columns"""
    df =  pd.read_csv(path)
    df['country'] = df.city.str.split(',').str[1].str.strip()
    #df['platform_country'] = df.platform.map(platform2country)
    df = add_date(df)
    df = save_reference_true(df)

    if fill_search_for_destination: # fillnas in reference for action 'search for destination' with the city
        idx_sd = df.action_type=='search for destination'
        idx_nan = df.reference.isnull()
        df.loc[idx_sd&idx_nan, 'reference'] = df.loc[idx_sd&idx_nan, 'city']

    if fill_filter_selection: # fillnas in reference for action 'filter selection' with the filter added to
                              #current_filters, if one and only one filter was added
        idx_a = df.action_type=='filter selection'
        idx_nan = df.reference.isnull()
        df['current_filters_list'] = df.current_filters.str.split('|').fillna('').apply(list)
        df['prev_current_filters_list'] = df.groupby('session_id').current_filters_list.shift()
        df['prev_current_filters_list'] = df['prev_current_filters_list'].fillna('').apply(list)
        df.loc[idx_a&idx_nan,'added_filters'] = df.loc[idx_a&idx_nan] \
                .apply(lambda row: set(row.current_filters_list)-set(row.prev_current_filters_list), axis=1)
        df.loc[idx_a&idx_nan,'length_added_filters'] = df.loc[idx_a&idx_nan,'added_filters'].apply(len)
        idx_one_added_filter = df.length_added_filters==1
        df.loc[idx_one_added_filter&idx_nan&idx_a, 'reference'] = df.loc[idx_one_added_filter&idx_nan&idx_a] \
                                                                    .added_filters.apply(lambda s: list(s)[0])
        df.drop(['current_filters_list', 'prev_current_filters_list',
                 'added_filters', 'length_added_filters'], axis=1, inplace=True)
    return df


def compute_enc(df_train, df_test, variable, pad_values_list=[]):

    variable_list = variable.split('|')
    all_set = set()
    if variable=='reference_aug':
        all_set = set(df_train['reference']) | set(df_test['reference'])
        aux_tr = df_train.impressions.dropna().str.split('|')
        aux_te = df_test.impressions.dropna().str.split('|')
        aux_filters_tr = df_train.current_filters.dropna().str.split('|')
        aux_filters_te = df_test.current_filters.dropna().str.split('|')
        aux_cities_tr = df_train.city.unique()
        aux_cities_te = df_test.city.unique()

        all_impressions_set =  set(itertools.chain.from_iterable(aux_tr))| \
                               set(itertools.chain.from_iterable(aux_te)) | \
                               set(itertools.chain.from_iterable(aux_filters_tr)) | \
                               set(itertools.chain.from_iterable(aux_filters_te)) | \
                               set(aux_cities_tr) | \
                               set(aux_cities_te)
        all_set = all_impressions_set|all_set
    elif len(variable_list)>2 and variable_list[0]=='reference': # for new part
        which_actions = variable_list[2:]
        idx_tr = df_train.action_type.isin(which_actions)
        idx_te = df_test.action_type.isin(which_actions)
        all_set = set(df_train.loc[idx_tr, 'reference']) |  set(df_test.loc[idx_te, 'reference'])
    elif len(variable_list)>2 and variable_list[0]=='reference_list':
        which_actions = variable_list[2:]
        idx_tr = df_train.action_type.isin(which_actions)
        idx_te = df_test.action_type.isin(which_actions)

        aux_poi_words_tr = df_train.loc[idx_tr, 'reference'].dropna().str.split(' ')
        aux_poi_words_te = df_test.loc[idx_te, 'reference'].dropna().str.split(' ')
        all_poi_words_set =  set(itertools.chain.from_iterable(aux_poi_words_tr)) | \
                               set(itertools.chain.from_iterable(aux_poi_words_te))
        all_set = all_poi_words_set-set([''])
    else: # old part
        all_set = set(df_train[variable]) | set(df_test[variable])

    if len(variable_list)>2 and variable_list[1]=='impressions': # for new part
        aux_tr = df_train.impressions.dropna().str.split('|')
        aux_te = df_test.impressions.dropna().str.split('|')
        all_impressions_set =  set(itertools.chain.from_iterable(aux_tr))| \
                               set(itertools.chain.from_iterable(aux_te))
        all_set = all_impressions_set | all_set

    if len(variable_list)>2 and variable_list[1]=='current_filters': # for new part
        aux_filters_tr = df_train.current_filters.dropna().str.split('|')
        aux_filters_te = df_test.current_filters.dropna().str.split('|')
        all_current_filters_set =  set(itertools.chain.from_iterable(aux_filters_tr)) | \
                               set(itertools.chain.from_iterable(aux_filters_te))
        all_set = all_current_filters_set | all_set

    ## old part
    if variable=='reference':
        aux_tr = df_train.impressions.dropna().str.split('|')
        aux_te = df_test.impressions.dropna().str.split('|')
        aux_filters_tr = df_train.current_filters.dropna().str.split('|')
        aux_filters_te = df_test.current_filters.dropna().str.split('|')

        all_impressions_set =  set(itertools.chain.from_iterable(aux_tr))| \
                               set(itertools.chain.from_iterable(aux_te)) | \
                               set(itertools.chain.from_iterable(aux_filters_tr)) | \
                               set(itertools.chain.from_iterable(aux_filters_te))
        all_set = all_impressions_set|all_set



    all_set =  pad_values_list + list(all_set-set(pad_values_list))


    dec_dic = dict(enumerate(all_set))
    enc_dic = {x:k for k,x in enumerate(all_set)}

    if ('Gay Friendly' in enc_dic)&('Gay-friendly' in enc_dic):
        #dec_dic.pop(enc_dic['Gay Friendly'])
        enc_dic['Gay Friendly'] = enc_dic['Gay-friendly']
    return enc_dic, dec_dic

# def compute_enc(df_train, df_test, variable, pad_values_list=[]):
#
#     all_set = set(df_train[variable]) |  set(df_test[variable])
#
#     if variable=='reference':
#         aux_tr = df_train.impressions.dropna().str.split('|')
#         aux_te = df_test.impressions.dropna().str.split('|')
#         aux_filters_tr = df_train.current_filters.dropna().str.split('|')
#         aux_filters_te = df_test.current_filters.dropna().str.split('|')
#
#         all_impressions_set =  set(itertools.chain.from_iterable(aux_tr))| \
#                                set(itertools.chain.from_iterable(aux_te)) | \
#                                set(itertools.chain.from_iterable(aux_filters_tr)) | \
#                                set(itertools.chain.from_iterable(aux_filters_te))
#         all_set = all_impressions_set|all_set
#
#     all_set =  pad_values_list + list(all_set-set(pad_values_list))
#
#     dec_dic = dict(enumerate(all_set))
#     enc_dic = {x:k for k,x in enumerate(all_set)}
#
#     return enc_dic, dec_dic

def get_enc_and_dec_dicts(df_train, df_test, pad_values_dict):
    enc_dicts = dict()
    dec_dicts = dict()
    for key in pad_values_dict:

        try:
            aux = key.split('|')
            key_name = (aux[0] + ' ' + aux[2]).strip().replace(' ', '_')
        except:
            key_name = key
        print (key)
        print (key_name)
        enc_dicts[key_name], dec_dicts[key_name] = compute_enc(df_train, df_test,
                                                     variable=key,
                                                     pad_values_list=pad_values_dict[key])
        print('reference size: ', len(enc_dicts[key_name]))
        for v in pad_values_dict[key]:
            print('pad label check: ', (v, enc_dicts[key_name][v]))
    #        print('pad label check: ', dec_dicts[key][0])
        print(' ')
    return enc_dicts, dec_dicts


def rename_sess(df_train, df_test):

    col_names = list(df_train.columns.values)
    df_train['is_train'] = True
    df_test['is_train'] = False

    df_all = pd.concat([df_train, df_test], axis=0).sort_values(['session_id', 'timestamp', 'step']).reset_index(drop=True)
    df_all['is_first_step']= df_all.step==1
    df_all['repeated_session_count'] = df_all.groupby(['session_id']).is_first_step.cumsum().astype(int)

    sessions_with_repeat_ids = df_all[df_all.repeated_session_count>1].session_id.unique()
    df_all['session_id_new'] = df_all.session_id
    idx = df_all.session_id.isin(sessions_with_repeat_ids)
    df_all.loc[idx,'session_id_new'] = df_all.loc[idx].apply(lambda row: '%s_%s' % (row.session_id, row.repeated_session_count), axis=1)

    df_all['session_id_original'] =  df_all['session_id'].copy()
    df_all['session_id'] =  df_all['session_id_new'].copy()
    df_all =  df_all.drop('session_id_new', axis=1)

    col_names_out = col_names+['session_id_original']

    df_tr = df_all.loc[df_all.is_train, col_names_out].reset_index(drop=True)
    df_te = df_all.loc[~df_all.is_train, col_names_out].reset_index(drop=True)
    return df_tr, df_te



def set_nans_val(df):
    """Adds a column 'reference_true' with the values of 'reference' and
    sets to None the last activity in the largest(s) session(s)"""

    df_temp = df.groupby('user_id').step.max().reset_index()
    df_temp.loc[:, IS_NAN_VAL] = True
    df_out = df.merge(df_temp, how='left', on=['user_id', 'step'])
    df_out[IS_NAN_VAL] =  df_out[IS_NAN_VAL].fillna(False)

    idx_nan = df_out[IS_NAN_VAL]==True
    df_out.loc[idx_nan, 'reference'] = None

    return df_out


def add_is_nan_reference(df):
    if df.size>0:
        idx = df.reference.isnull()
        df.loc[~idx, IS_NAN_VAL] = False
        df.loc[idx, IS_NAN_VAL] = True
    return df

def add_is_nan_clickout_reference(df):
    if df.size>0:
        idx = (df.reference.isnull())&(df.action_type=='clickout item')
        df.loc[~idx, IS_NAN_CLICKOUT_VAL] = False
        df.loc[idx, IS_NAN_CLICKOUT_VAL] = True
    return df

def add_column_session_with_clickout_nans(df):
    if IS_SESSION_WITH_CLICKOUT_NANS in df:
        df = df.drop([IS_SESSION_WITH_CLICKOUT_NANS], axis=1)
    df_out = df.copy()
    df_sessions_with_clickout_nans = df_out.groupby('session_id')[IS_NAN_CLICKOUT_VAL].any()
    df_temp = df_sessions_with_clickout_nans.reset_index()
    df_temp.columns = ['session_id', IS_SESSION_WITH_CLICKOUT_NANS]
    df_out = df_out.merge(df_temp, on='session_id')
    return df_out

def add_column_session_with_nans(df):
    if IS_SESSION_WITH_NANS in df:
        df = df.drop([IS_SESSION_WITH_NANS], axis=1)
    df_out = df.copy()
    df_sessions_with_nans = df_out.groupby('session_id')[IS_NAN_VAL].any()
    df_temp = df_sessions_with_nans.reset_index()
    df_temp.columns = ['session_id', IS_SESSION_WITH_NANS]
    df_out = df_out.merge(df_temp, on='session_id')
    return df_out

def add_set_of_boolean_columns(df_tr_, df_val_):
    print ('adding is_nan_clickout_reference')
    df_val_ = add_is_nan_clickout_reference(df_val_)
    df_tr_ = add_is_nan_clickout_reference(df_tr_)

    print ('adding column is_session_with_nans')
    df_tr_ = add_column_session_with_nans(df_tr_)
    df_val_ = add_column_session_with_nans(df_val_)

    print ('adding column is_session_with_clickout_nans')
    df_val_ = add_column_session_with_clickout_nans(df_val_)
    df_tr_ = add_column_session_with_clickout_nans(df_tr_)

    print ('adding pre and post nan sessions labels')
    df_val_ = label_pre_and_post_nan_val(df_val_)
    df_tr_ = label_pre_and_post_nan_tr(df_tr_)

    return df_tr_, df_val_


def train_val_split_normal(df, tr_days = [1,2,3,4], val_days = [5,6]):
    """Does train-val split according ot the chosen days, sets to None the
       'reference' of the last activity in the largest(s) session(s)"""

    #train
    idx_tr = df.day.isin(tr_days)
    df_tr = df.loc[idx_tr].reset_index(drop=True)

    #validation
    idx_val = df.day.isin(val_days)
    df_val = df.loc[idx_val].reset_index(drop=True)

    #set to None values that are supposed to be Nones
    df_val = set_nans_val(df_val)
    df_tr = add_is_nan_reference(df_tr)

    df_tr, df_val = add_set_of_boolean_columns(df_tr, df_val)

    return df_tr, df_val


def label_pre_and_post_nan_val(df):
    Vars = ['user_id', 'session_id', 'timestamp', 'step', IS_SESSION_WITH_CLICKOUT_NANS]
    df_sort = df[Vars].sort_values(['user_id', 'session_id', 'timestamp', 'step']).reset_index(drop=True)
    df_sort_inv = df[Vars].sort_values(['user_id', 'session_id', 'timestamp', 'step'],
                                               ascending=[True, True, False, False]).reset_index(drop=True)

    df_sort[IS_PRE_CLICKOUT_NAN_SESSION] = df_sort.groupby('user_id')[IS_SESSION_WITH_CLICKOUT_NANS].cumsum()<1
    df_sort_inv[IS_POST_CLICKOUT_NAN_SESSION] = df_sort_inv.groupby('user_id')[IS_SESSION_WITH_CLICKOUT_NANS].cumsum()<1
    return df.merge(df_sort, on=Vars).merge(df_sort_inv, on=Vars)

def label_pre_and_post_nan_tr(df):
    df[IS_PRE_CLICKOUT_NAN_SESSION] = True
    df[IS_POST_CLICKOUT_NAN_SESSION] = False
    return df


def resplit_train_test_split_cheat(df_tr, df_val, option='leave_in_all_pre_nans', is_train_test=True):
    """ do cheating train_val split
    option: 'normal', 'leave_in_all_pre_nans', 'leave_out_only_session_with_nans' or
    'leave_out_only_clickout_with_nans'"""

    # add missing columns if its resplitting df_train and df_test
    if is_train_test:
        print ('adding is_nan_reference')
        df_val = add_is_nan_reference(df_val)
        df_tr = add_is_nan_reference(df_tr)

        df_tr, df_val = add_set_of_boolean_columns(df_tr, df_val)

    print ('concat')
    df_reb = pd.concat([df_tr, df_val], axis=0, sort=False).reset_index(drop=True)

    print ('splitting %s' % option)
    if option=='leave_in_all_pre_session_with_clickout_nans':
        idx_tr = df_reb[IS_PRE_CLICKOUT_NAN_SESSION]
        df_tr = df_reb.loc[idx_tr]
        df_val = df_reb.loc[~idx_tr]

    elif option=='leave_in_all_pre_session_with_clickout_nans_and_out_only_session_with_clickout_nans':
        idx_val = df_reb[IS_SESSION_WITH_CLICKOUT_NANS]
        idx_tr = df_reb[IS_PRE_CLICKOUT_NAN_SESSION]
        df_tr = df_reb.loc[idx_tr]
        df_val = df_reb.loc[idx_val]

    elif option=='leave_out_only_session_with_clickout_nans':
        idx_val = df_reb[IS_SESSION_WITH_CLICKOUT_NANS]
        df_tr = df_reb.loc[~idx_val]
        df_val = df_reb.loc[idx_val]

    elif option=='leave_out_only_session_with_clickout_nans_and_in_only_sessions_without_nans':
        idx_cn = (df_reb[IS_SESSION_WITH_CLICKOUT_NANS])
        idx_n = (df_reb[IS_SESSION_WITH_NANS])
        df_tr = df_reb.loc[~idx_n]
        df_val = df_reb.loc[idx_cn]

    elif option=='leave_out_only_clickout_with_nans':
        idx_val = (df_reb.is_nan_val)&(df_reb.action_type=='clickout item')
        df_tr = df_reb.loc[~idx_val]
        df_val = df_reb.loc[idx_val]

    else:
        print ('invalid option!')


    return df_tr.reset_index(drop=True), df_val.reset_index(drop=True)



def train_val_split_all_options(df, tr_days=[1,2,3,4], val_days=[5,6], option='normal'):
    """ do cheating train_val split
    option: 'normal', 'leave_in_all_pre_nans', 'leave_out_only_session_with_nans' or
    'leave_out_only_clickout_with_nans'"""

    # start by using the standard split that also determines which are the nans
    print ('making standard split')
    df_tr, df_val = train_val_split_normal(df, tr_days=tr_days, val_days=val_days)

    if option!='normal':
        df_tr, df_val = resplit_train_test_split_cheat(df_tr, df_val, option=option, is_train_test=False)
    return df_tr, df_val
