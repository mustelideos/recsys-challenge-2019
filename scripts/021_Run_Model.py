#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import random
import os, pickle,math
import argparse
import logging

from tqdm import tqdm
import torch, torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

OUTPUT_NORMLIZATIONS_VAL = 'normalizations_val.pkl'
OUTPUT_NORMLIZATIONS_SUBM = 'normalizations_submission.pkl'

DEFAULT_FEATURES_DIR_NAME = 'nn_vnormal'
DEFAULT_PREPROC_DIR_NAME = 'data_processed_vnormal'
DEFAULT_SPLIT = 'normal'

MODELS_PATH = '../models'
os.makedirs(MODELS_PATH) if not os.path.exists(MODELS_PATH) else None

PREDICTIONS_PATH = '../predictions'
os.makedirs(PREDICTIONS_PATH) if not os.path.exists(PREDICTIONS_PATH) else None

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


def ReciprocalRank(predicted,actual):
    if actual in predicted:
        rank_query = predicted.index(actual) + 1
        rr = 1.0 / rank_query
    else:
        rr = 0.0
    return rr

def MeanReciprocalRank(predicted,actual):
    return np.mean([ReciprocalRank(p,a) for a,p in zip(actual, predicted)])

def sumReciprocalRank(predicted,actual):
    return np.sum([ReciprocalRank(p,a) for a,p in zip(actual, predicted)])


class m2oAttentionA2(nn.Module):
    # many to one Attention - Bahdanau (sum)
    # ver por exemplo https://arxiv.org/pdf/1602.02068.pdf

    def __init__( self, in1_dim, in2_dim, hidden_dim):

        super(m2oAttentionA2, self).__init__()

        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(in1_dim, self.hidden_dim,bias=False)
        self.W2 = nn.Linear(in2_dim, self.hidden_dim)

        self.V = nn.Linear(self.hidden_dim, 1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input, z, mask):
        batch_size, S, dim = input.shape

        w1h = self.W1(input)
        w2h = self.W2(z).unsqueeze(1)

        u = torch.tanh(w1h + w2h)
        att = self.V(u)

        att = att.masked_fill(mask.unsqueeze(2) == 0, -1e10)

        att = self.softmax(att).transpose(1, 2)
        out =  torch.bmm(att, input).squeeze(1)

        return out


class m2oAttentionA3(nn.Module):
    # many to one Attention - Bahdanau (sum)
    # ver por exemplo https://arxiv.org/pdf/1602.02068.pdf

    def __init__( self, in1_dim, in2_dim, hidden_dim):

        super(m2oAttentionA3, self).__init__()

        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(in1_dim, self.hidden_dim,bias=False)
        self.W2 = nn.Linear(in2_dim, self.hidden_dim)

        self.W3 = nn.Linear(in1_dim, self.hidden_dim,bias=False)
        self.W4 = nn.Linear(in2_dim, self.hidden_dim)

        self.V = nn.Linear(self.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp1, inp2, mask):
        batch_size, S, dim = inp1.shape

        w1h = self.W1(inp1).repeat(25, 1, 1)

        w2h = self.W2(inp2).view(batch_size * 25, self.hidden_dim).unsqueeze(1)

        u = torch.tanh(w1h + w2h)

        att = self.V(u).view(batch_size , 25, -1)

        att = att.masked_fill(mask.unsqueeze(1).repeat(1, 25, 1) == 0, -1e10)

        att = self.softmax(att)

        inp1_av =  torch.bmm(att, inp1)

        w3h = self.W3(inp1_av)
        w4h = self.W4(inp2)

        u = torch.tanh(w3h + w4h)

        return u


class self_Attention(nn.Module):
    def __init__( self, in1_dim, in2_dim, hidden_dim):

        super(self_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(in1_dim, self.hidden_dim,bias=False)
        #self.W2 = nn.Linear(in2_dim, self.hidden_dim)

        self.W1a = nn.Linear(in1_dim, self.hidden_dim,bias=False)
        self.W2a = nn.Linear(in2_dim, self.hidden_dim)

        self.V = nn.Linear(self.hidden_dim, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, mask=None):

        #-----     Attention     ----------
        batch_size, S, dim = x1.shape

        w1ha = self.W1a(x1).repeat(S, 1, 1)

        w2ha = self.W2a(x2).view(batch_size * S, self.hidden_dim).unsqueeze(1)

        u = torch.tanh(w1ha + w2ha)

        att = self.V(u).view(batch_size , S, -1)

        att = att.masked_fill(mask.unsqueeze(1).repeat(1, S, 1) == 0, -1e10)

        att = self.softmax(att)

        x =  torch.bmm(att, x1)
        #-----------------------------------------------

        w1h = self.W1(x)


        out = torch.tanh(w1h)
        return out



class m2oAttentionB2(nn.Module):
    # many to one Attention - Bahdanau (sum)
    def __init__( self, in1_dim, in2_dim, hidden_dim):

        super(m2oAttentionB2, self).__init__()

        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(in1_dim, self.hidden_dim,bias=False)
        self.W2 = nn.Linear(in2_dim, self.hidden_dim)

        self.V = nn.Linear(self.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input,z, mask):
        batch_size, S, T, dim = input.shape

        input = input.view(batch_size * S, -1,dim)
        mask = mask.view(batch_size * S,-1)

        w1h = self.W1(input)
        w2h = self.W2(z).view(batch_size * S, -1).unsqueeze(1)

        u = torch.tanh(w1h+w2h)
        att = self.V(u)

        att = att.masked_fill(mask.unsqueeze(2) == 0, -1e10)

        att = self.softmax(att).transpose(1, 2)
        out =  torch.bmm(att, input).squeeze(1).view(batch_size , S, -1)
        return out


class combine_head(nn.Module):

    def __init__(self,hidden_in1,hidden_in2,hidden_out):
        self.hidden_in1 = hidden_in1
        self.hidden_in2 = hidden_in2
        self.hidden_out = hidden_out

        super(combine_head, self).__init__()

        self.W1 = nn.Linear(self.hidden_in1, self.hidden_out,bias=False)
        self.W2 = nn.Linear(self.hidden_in2, self.hidden_out)

    def forward(self, impr,z):

        w1e = self.W1(impr)
        w2h = self.W2(z).unsqueeze(1)

        u = torch.tanh(w1e + w2h)
        return u


class fc_head(nn.Module):

    def __init__(self,pl_dim):
        self.pl_dim = pl_dim
        super(fc_head, self).__init__()

        self.inter_dim = 32

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.pl_dim,256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256,self.inter_dim),
            )

        self.layer_norm = nn.LayerNorm(self.inter_dim*25)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.L = nn.Linear(self.inter_dim*25,25)

    def forward(self, x):
        batch_size, S, dim = x.shape
        x = self.relu(self.layer_norm(self.fc(x).view(batch_size, -1)))
        x = self.dropout(x)
        x = self.L(x)
        x = 10*torch.tanh(x)
        return x


class RecNet(nn.Module):

    def __init__(self):
        super(RecNet,self).__init__()


        self.embs_dic ={'impf_stars_enc': (6,2) ,
                        'impf_from_stars_enc':(6,2),
                        'impf_rating_enc':(5,2),
                        'reference':(967102,128),
                        'action':(12,4),
                        'platform':(55,8),
                        'device':(3,1),
                        'city':(37845,64),
                        'country':(224,16),
                        'position':(26,1),
                        'user_id':(948041,128),
                        'properties':(158,32)}

        impr_num_dim = 46 #numeric feature number
        ses_num_dim = 7  #numeric feature number
        seq_num_dim = 4+8  #numeric feature number

        hidden_size = 128
        att_hidden_size = 32
        out_dim = 128

        self.seq_cat_embs_list = ['action','reference','city','country']
        self.ses_cat_embs_list = ['action','reference','platform','device']
        self.imp_cat_embs_list = ['impf_stars_enc','impf_from_stars_enc','impf_rating_enc']
        self.imp_feat_list = ['reference','position','properties']#

        self.imp_cat_embs = nn.ModuleList([nn.Embedding(self.embs_dic[f][0],self.embs_dic[f][1],padding_idx=0) for f in self.imp_cat_embs_list])

        self.prop_emb = nn.Embedding(self.embs_dic['properties'][0],self.embs_dic['properties'][1], padding_idx=0)

        seq_cat_dim = sum(self.embs_dic[k][1] for k in self.seq_cat_embs_list)
        ses_cat_dim = sum(self.embs_dic[k][1] for k in self.ses_cat_embs_list)
        imp_cat_dim = sum(self.embs_dic[k][1] for k in self.imp_cat_embs_list)
        imp_feat_dim = sum(self.embs_dic[k][1] for k in self.imp_feat_list)

        imp_all_dim = imp_feat_dim+impr_num_dim+imp_cat_dim
        prop_emb_dim = self.embs_dic['properties'][1]


        self.ref_emb = nn.Embedding(self.embs_dic['reference'][0],self.embs_dic['reference'][1], padding_idx=0)
        self.act_emb = nn.Embedding(self.embs_dic['action'][0],self.embs_dic['action'][1], padding_idx=0)
        self.pla_emb = nn.Embedding(self.embs_dic['platform'][0],self.embs_dic['platform'][1])
        self.dev_emb = nn.Embedding(self.embs_dic['device'][0],self.embs_dic['device'][1])
        self.pos_emb = nn.Embedding(self.embs_dic['position'][0],self.embs_dic['position'][1], padding_idx=0)
        self.cit_emb = nn.Embedding(self.embs_dic['city'][0],self.embs_dic['city'][1], padding_idx=0)
        self.cou_emb = nn.Embedding(self.embs_dic['country'][0],self.embs_dic['country'][1], padding_idx=0)

        self.gru = nn.GRU(seq_cat_dim+seq_num_dim, hidden_size, num_layers=1, dropout=0, batch_first=True)

        self.fc_head = fc_head(6*out_dim)

        self.W1 = nn.Linear(imp_all_dim,out_dim)

        self.seq_att = m2oAttentionA2(hidden_size, hidden_size, hidden_size)
        self.seq_att2 = m2oAttentionA3(hidden_size,imp_all_dim,hidden_size)
        self.ses_filt_att = m2oAttentionA2(self.embs_dic['reference'][1], hidden_size,hidden_size)
        self.prop_att = m2oAttentionB2(prop_emb_dim, self.embs_dic['reference'][1], att_hidden_size)
        self.self_att = self_Attention(imp_all_dim,imp_all_dim,hidden_size)

        self.combine_head1 = combine_head(imp_all_dim,hidden_size,out_dim)
        self.combine_head2 = combine_head(imp_all_dim,hidden_size,out_dim)
        self.combine_head3 = combine_head(imp_all_dim,ses_cat_dim+ses_num_dim,out_dim)
        self.combine_head4 = combine_head(imp_all_dim,2*hidden_size,out_dim)

        self.Wh0 = nn.Linear(ses_cat_dim+ses_num_dim,hidden_size)
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)


    def forward(self, seq_cat, seq_num, ses_cat, ses_num, ses_filt, impr, impr_num, imp_cat, prop_list, length):
        batch_size, T, dim = seq_cat.shape
        #--------------------------------------------------------------

        #1- seq embendings
        es1 = self.act_emb(seq_cat[:,:,0])
        es2 = self.ref_emb(seq_cat[:,:,1])
        es3 = self.cit_emb(seq_cat[:,:,2])
        es4 = self.cou_emb(seq_cat[:,:,3])

        seq_emb = torch.cat((es1,es2,es3,es4,seq_num), 2)
        #--------------------------------------------------------------
        #2- session cat embendings
        e3 = self.act_emb(ses_cat[:,0])
        e4 = self.ref_emb(ses_cat[:,1])
        e5 = self.pla_emb(ses_cat[:,2])
        e6 = self.dev_emb(ses_cat[:,3])
        e7 = self.ref_emb(ses_cat[:,4])

        sess_feat = torch.cat((e3,e4,e5,e6,ses_num), 1)

        #3-  RNN sequence encoding
        h0 = self.Wh0(sess_feat)

        xx = torch.nn.utils.rnn.pack_padded_sequence(seq_emb, length, batch_first=True)
        yy, h = self.gru (xx, h0.unsqueeze(0))
        seq_h, _ = torch.nn.utils.rnn.pad_packed_sequence(yy, batch_first=True)

        seq_feat = h.view(batch_size,-1)  #hidden (one direction)

        mask_seq = (seq_cat.sum(2)!=0)
        _,sl,_ = seq_h.shape

        att_seq_feat = self.seq_att(seq_h,seq_feat,mask_seq[:,:sl])

       #--------------------------------------------------------------

        #3.2 - current_filters_enc  -----------------------------------
        ses_filt_emb = self.ref_emb(ses_filt)
        mask = (ses_filt!=0)
        att_ses_filt = self.ses_filt_att(ses_filt_emb,seq_feat,mask)
        ses_filt_feat = torch.cat((att_ses_filt,e7), 1)

        #--------------------------------------------------------------
        #4.0 - impressions features: embendings+num   -----------------
        imp_emb = self.ref_emb(impr[:,:,0])

        imp_emb_feat = [emb(imp_cat[:,:,i]) for i,emb in enumerate(self.imp_cat_embs)]#+[self.act_emb(imp_cat[:,:,3])]
        imp_emb_feat = torch.cat(imp_emb_feat, dim = 2)

        #4.1 impressions propr. embendings  ---------------

        mask = (prop_list!=0)
        prop_feat = self.prop_emb(prop_list)
        prop_feat_av = self.prop_att(prop_feat,imp_emb,mask)

        #4.1.2 concat all impressions features ---------------
        impr_feat = torch.cat((imp_emb,impr_num,self.pos_emb(impr[:,:,1]),prop_feat_av,imp_emb_feat), 2) #

        #4.2 - impressions-seq attention x impressions features ---------
        u1 = self.seq_att2(seq_h,impr_feat,mask_seq[:,:sl])

        #4.3 - seq-seq attention x impressions features --------------------
        u2 = self.combine_head2(impr_feat,att_seq_feat)

        #4.4 - session x impressions features -------------------------------
        u3 = self.combine_head3(impr_feat,sess_feat)

        #4.5 - filters x impressions features -------------------------------
        u4 = self.combine_head4(impr_feat,ses_filt_feat)

        #5 - simple impressions features layer----------------------------------
        u5 = torch.tanh(self.W1(impr_feat))

        #6  - self-attention impressions layer------------------------------
        mask = (impr[:,:,0]!=0)
        u6 = self.self_att(impr_feat,impr_feat,mask) #

        #-------------------------------------------------------------
        x = torch.cat((u1,u2,u3,u4,u5,u6),2)

        #7 - full conected layer for final output

        x = self.fc_head(x)
        out = F.log_softmax(x,dim=-1)
        return out




class myDataset(Dataset):

    def __init__(self,df,total_length, stat_dic):

        self.data = df
        self.total_length = total_length
        self.stat_dic = stat_dic
        self.feat_list =  ['pi_user_id',
                         'pi_session_id',
                         'pa_user_id_clickout_item',
                         'pa_user_id_interaction_item_image',
                         'pa_user_id_interaction_item_info',
                         'pa_user_id_search_for_item',
                         'pa_user_id_interaction_item_deals',
                         'pa_user_id_interaction_item_rating',
                         'pa_user_id_all_interactions',
                         'pa_session_id_clickout_item',
                         'pa_session_id_interaction_item_image',
                         'pa_session_id_interaction_item_info',
                         'pa_session_id_search_for_item',
                         'pa_session_id_interaction_item_deals',
                         'pa_session_id_interaction_item_rating',
                         'pa_session_id_all_interactions']

        self.CTR_list = ['CTR_user_id','CTR_session_id']

    def __len__(self):
        return self.data.shape[0]

    def ses_f_norm(self,x, nan_val = -1, max_val = None):
        if x != -1:
            x = min(max_val, x)
        else:
            x = nan_val
        x /= max_val
        return x

    def seq_f_norm(self,x_list, nan_val = -1, max_val = None):
        for k,x in enumerate(x_list):
            if x_list[k] != -1:
                x_list[k] = min(max_val, x)
            else:
                x_list[k] = nan_val
            x_list[k] /= max_val
        return x_list

    def complete(self,impressions):
        out = np.zeros((25,2), dtype="int32")  # the "template"
        out[:len(impressions),:] = impressions
        return out#.flatten()

    def complete_prop(self,prop_list):
        out = np.zeros(112, dtype="int32")  # the "template"
        out[:len(prop_list)] = prop_list
        return out

    def __getitem__(self, index):

        row = self.data.iloc[index].copy()

        #------------------------seq features---------------------
        #cat
        session = torch.from_numpy(np.array(eval(row['seq_cat']), dtype="int32"))

        pad_seq_cat = torch.zeros((self.total_length,4))
        if session.shape[0] > self.total_length: pad_seq_cat[:] = session[(session.shape[0]-self.total_length):]
        else: pad_seq_cat[:session.shape[0]] = session
        #num

        ses_num1 = torch.from_numpy(np.array(eval(row['seq_bool']))).double()

        ses_num2 = self.seq_f_norm(torch.from_numpy(np.array(eval(row['seq_t1_corr']))).double(), nan_val = 850, max_val = 800)
        ses_num3 = self.seq_f_norm(torch.from_numpy(np.array(eval(row['seq_t2_corr']))).double(), nan_val = 850, max_val = 800)
        ses_num4 = self.seq_f_norm(torch.from_numpy(np.array(eval(row['seq_t3_corr']))).double(), nan_val = 850, max_val = 800)

        ses_num5 = torch.from_numpy((np.array(eval(row['seq_time_new']))-self.stat_dic['seq_time_new']['means'])/self.stat_dic['seq_time_new']['stds'])

        ses_num = torch.cat((ses_num1,ses_num2,ses_num3,ses_num4,ses_num5),dim=-1)

        pad_seq_num = torch.zeros((self.total_length,4+8))
        if session.shape[0] > self.total_length: pad_seq_num[:] = ses_num[(ses_num.shape[0]-self.total_length):]
        else: pad_seq_num[:ses_num.shape[0]] = ses_num

        #------------------------impressions features---------------------
        impressions = np.array(eval(row['imp_list']), dtype="int32")
        prices = np.array(eval(row['prices_list']), dtype="float")
        imp_n1 = (torch.from_numpy(prices).unsqueeze(1)-self.stat_dic['prices']['means'])/self.stat_dic['prices']['stds']
        imp_n1b = (torch.from_numpy(prices).unsqueeze(1))/np.max(prices)
        imp_n1c = (torch.from_numpy(np.log(prices+1)).unsqueeze(1)-self.stat_dic['prices_log']['means'])/self.stat_dic['prices_log']['stds']

        imp_n2 = torch.from_numpy(np.array(eval(row['impf_stars']), dtype="float")).unsqueeze(1)
        imp_n3 = torch.from_numpy(np.array(eval(row['impf_from_stars']), dtype="float")).unsqueeze(1)
        imp_n4 = torch.from_numpy(np.array(eval(row['impf_rating']), dtype="float")).unsqueeze(1)
        imp_n5 = torch.from_numpy(np.array(eval(row['impf_n_prop']), dtype="float")).unsqueeze(1)
        imp_n6 = torch.from_numpy(np.array(eval(row['impf_is_from_stars_nan']), dtype="float")).unsqueeze(1)
        imp_n7 = torch.from_numpy(np.array(eval(row['impf_is_rating_nan']), dtype="float")).unsqueeze(1)
        imp_n8 = torch.from_numpy(np.array(eval(row['impf_is_stars_nan']), dtype="float")).unsqueeze(1)

        imp_n10 = torch.cat( [torch.from_numpy(np.array(eval(row[feat]), dtype="float")).unsqueeze(1) for feat in self.feat_list],dim=-1)
        imp_n10 =  imp_n10/(1+imp_n10.max(keepdim=True,dim=-1)[0])
        imp_n10_bool = (imp_n10>0).double() # booleanas

        imp_n11 = torch.cat([torch.from_numpy(np.array(eval(row[feat]), dtype="float")).unsqueeze(1) for feat in self.CTR_list],dim=-1)

        imp_n12 = torch.from_numpy(((np.array(eval(row['impressions_list_timesincelastaction']), dtype="float")-self.stat_dic['impressions_list_timesincelastaction']['means'])
)/self.stat_dic['impressions_list_timesincelastaction']['stds']).unsqueeze(1)


        imp_n14 = torch.from_numpy(((np.array(eval(row['past_dwell_with_items_session_id']), dtype="float")-self.stat_dic['dwell_times']['means'])
)/self.stat_dic['dwell_times']['stds']).unsqueeze(1)


        imp_num = torch.cat((imp_n1,imp_n1b,imp_n1c,imp_n2,imp_n3,imp_n4,imp_n5,imp_n6,imp_n7,imp_n8,imp_n10,imp_n10_bool,imp_n11,imp_n12,imp_n14),dim=-1)

        imp_c1 = torch.from_numpy(np.array(eval(row['impf_stars_enc']), dtype="int32")).unsqueeze(1)
        imp_c2 = torch.from_numpy(np.array(eval(row['impf_from_stars_enc']), dtype="int32")).unsqueeze(1)
        imp_c3 = torch.from_numpy(np.array(eval(row['impf_rating_enc']), dtype="int32")).unsqueeze(1)

        imp_cat = torch.cat((imp_c1,imp_c2,imp_c3),dim=-1).long()

        impressions = np.concatenate((impressions,np.arange(1,impressions.shape[0]+1)),axis=0).reshape(2,-1).T
        if len(impressions) < 25:
            impressions = self.complete(impressions)
        impressions = torch.from_numpy(impressions).long()

        pad_imp_num = torch.zeros((25,imp_num.shape[1])).double()
        if imp_num.shape[0] < 25: pad_imp_num[:imp_num.shape[0]] = imp_num
        else: pad_imp_num = imp_num

        pad_imp_cat = torch.zeros((25,imp_cat.shape[1])).long()
        if imp_cat.shape[0] < 25: pad_imp_cat[:imp_cat.shape[0]] = imp_cat
        else: pad_imp_cat = imp_cat

        #---------------------- impressions prop. features--------------------
        prop_list = torch.from_numpy(np.array([self.complete_prop(eval(pl)) for pl in eval(row['imp_item_prop_list0'])]).astype("int32")).long()

        pad_imp_prop = torch.zeros((25,prop_list.shape[1])).long()
        if prop_list.shape[0] < 25: pad_imp_prop[:prop_list.shape[0]] = prop_list
        else: pad_imp_prop = prop_list

        #-----------------------------------
        leng = np.array(min(session.shape[0],self.total_length)).astype("int32")
        if leng == 0:print('diff')

        targ = row['targ'].astype("int32")

        #------------------------session features---------------------
        ses_filt = np.array(eval(row['current_filters_enc']), dtype="int32")
        if ses_filt.shape[0]==0:
            ses_filt = np.array([0], dtype="int32")

        sess_filt = torch.from_numpy(ses_filt).long()

        filt_len = 18
        pad_filt_enc = torch.zeros(filt_len).long()
        if sess_filt.shape[0] < filt_len: pad_filt_enc[:sess_filt.shape[0]] = sess_filt
        else: pad_filt_enc = sess_filt

        sess_cats = np.array([row['prev_act'],row['prev_ref'],row['platf_enc'],row['device_enc'],row['last_activated_filter_enc']], dtype="int32")

        time1b = (row['time_since_prev_any_all_action_corrected']-self.stat_dic['time_since_prev_any_all_action_corrected']['mean'])/self.stat_dic['time_since_prev_any_all_action_corrected']['std']
        time1b_log = (np.log(1.9+row['time_since_prev_any_all_action_corrected'])-self.stat_dic['time_since_prev_any_all_action_corrected_log']['mean'])/self.stat_dic['time_since_prev_any_all_action_corrected_log']['std']

        time1 = self.ses_f_norm(row['time_since_prev_any_all_action_corrected'], nan_val = 850, max_val = 800)
        time2 = self.ses_f_norm(row['time_since_prev_any_item_action_corrected'], nan_val = 850, max_val = 800)
        time3 = self.ses_f_norm(row['time_since_prev_clickout_item_corrected'], nan_val = 850, max_val = 800)

        sess_num = np.array([time1b,time1b_log,time1,time2,time3,row['is_last_step'],row['is_platform_equal_country']])

        return pad_seq_cat,pad_seq_num, sess_cats, sess_num, pad_filt_enc, impressions, pad_imp_num, pad_imp_cat, pad_imp_prop, leng, targ


# # Train - Utils

def lr_scheduler(optimizer,  lr=0.1):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model_val(model, dataloaders, n_epochs, batch_size=128, lr=0.001):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= 1e-5)

    criterion = nn.NLLLoss() # se este escolhermos este temos de alterar na rede a ultima layers

    lrs = [0.001,0.001,0.00075,0.0005,0.0005]

    step = 0
    val_cost_0 = 0

    for epoch in tqdm(range(n_epochs), desc='1st loop'):
        tqdm.write("runnig epoch:   %2.0f" %(epoch+1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                lr_scheduler(optimizer,  lrs[epoch])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_cost = 0.0

            # Iterate over data.
            for seq_cat,seq_num, ses_cat, ses_num, filt_enc, imp_inp, imp_num, imp_cat, imp_prop, leng, labels in tqdm(dataloaders[phase]):

                seq_cat = seq_cat.long().to(device)
                ses_cat = ses_cat.long().to(device)
                seq_num = seq_num.float().to(device)
                filt_enc = filt_enc.long().to(device)

                ses_num = ses_num.float().to(device)
                imp_inp = imp_inp.long().to(device)
                imp_num = imp_num.float().to(device)
                imp_cat = imp_cat.long().to(device)
                imp_prop = imp_prop.long().to(device)

                leng = leng.long().to(device)

                labels = labels.long().to(device)

                sorted_len, indices = leng.sort(descending=True)
                seq_cat = seq_cat[indices].squeeze(1)
                imp_prop = imp_prop[indices]
                ses_cat = ses_cat[indices]
                seq_num = seq_num[indices]
                ses_num = ses_num[indices]
                filt_enc = filt_enc[indices]

                imp_inp = imp_inp[indices]
                imp_num = imp_num[indices]
                imp_cat = imp_cat[indices]

                labels = labels[indices]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(seq_cat,seq_num, ses_cat,ses_num,filt_enc, imp_inp,imp_num,imp_cat,imp_prop, sorted_len)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * seq_cat.size(0)

                sort_pred = [np.argsort(-1*c_pred.detach().cpu().numpy()).tolist() for c_pred in outputs]
                running_cost += sumReciprocalRank(sort_pred,list(labels.cpu().numpy()))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            val_cost = running_cost / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}, Mean Reciprocal Rank: {:.4f}'.format(phase, epoch_loss,val_cost))
            if phase=='val' and val_cost>val_cost_0:
                torch.save(model.state_dict(),MODELS_PATH+'/RecNet_Model_GRU_f04_v23dw_Val.pkl')
                val_cost_0 = val_cost
                best_epoch = epoch+1

    print('Best Mean Reciprocal Rank: {:.4f}, with {:d} epochs training'.format(val_cost_0,best_epoch))
    with torch.no_grad():  # operations inside don't track history
        model.load_state_dict(torch.load(MODELS_PATH+'/RecNet_Model_GRU_f04_v23dw_Val.pkl'))
        model.eval()
        out_pred = []

        # Iterate over data.
        for seq_cat,seq_num, ses_cat, ses_num, filt_enc, imp_inp, imp_num, imp_cat, imp_prop, leng, _ in tqdm(dataloaders['val']):

            seq_cat = seq_cat.long().to(device)
            ses_cat = ses_cat.long().to(device)
            ses_num = ses_num.float().to(device)
            seq_num = seq_num.float().to(device)
            filt_enc = filt_enc.long().to(device)

            imp_inp = imp_inp.long().to(device)
            imp_num = imp_num.float().to(device)
            imp_cat = imp_cat.long().to(device)
            imp_prop = imp_prop.long().to(device)
            leng = leng.long().to(device)

            sorted_len, indices = leng.sort(descending=True)
            seq_cat = seq_cat[indices].squeeze(1)
            ses_cat = ses_cat[indices]
            ses_num = ses_num[indices]
            seq_num = seq_num[indices]
            filt_enc = filt_enc[indices]

            imp_inp = imp_inp[indices]
            imp_num = imp_num[indices]
            imp_cat = imp_cat[indices]
            imp_prop = imp_prop[indices]

            outputs = model(seq_cat,seq_num, ses_cat,ses_num,filt_enc, imp_inp,imp_num,imp_cat,imp_prop, sorted_len)
            # reorganiza o tensor na ordem correta
            ordered_out_prob = torch.empty_like(outputs)
            ordered_out_prob[indices] = outputs
            out_pred.append(np.exp(ordered_out_prob.cpu().numpy()))

    return np.concatenate(out_pred)


def train_model(model, dataloaders, n_epochs, kf=0, batch_size=128, lr=0.001):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= 1e-5)

    criterion = nn.NLLLoss()

    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,4], gamma=0.5)
    lrs = [0.001,0.001,0.00075,0.0005,0.0005]

    step = 0
    for epoch in tqdm(range(n_epochs), desc='1st loop'):
        tqdm.write("runnig epoch:   %2.0f" %(epoch+1))

        phase = 'train'
        if phase == 'train':
                #scheduler.step()
            lr_scheduler(optimizer,  lrs[epoch])

            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_cost = 0.0

        # Iterate over data.

        for seq_cat,seq_num, ses_cat, ses_num, filt_enc, imp_inp, imp_num, imp_cat, imp_prop, leng, labels in tqdm(dataloaders[phase]):

            seq_cat = seq_cat.long().to(device)
            ses_cat = ses_cat.long().to(device)
            seq_num = seq_num.float().to(device)
            filt_enc = filt_enc.long().to(device)

            ses_num = ses_num.float().to(device)
            imp_inp = imp_inp.long().to(device)
            imp_num = imp_num.float().to(device)
            imp_cat = imp_cat.long().to(device)
            imp_prop = imp_prop.long().to(device)

            leng = leng.long().to(device)

            labels = labels.long().to(device)

            sorted_len, indices = leng.sort(descending=True)
            seq_cat = seq_cat[indices].squeeze(1)
            imp_prop = imp_prop[indices]
            ses_cat = ses_cat[indices]
            seq_num = seq_num[indices]
            ses_num = ses_num[indices]
            filt_enc = filt_enc[indices]

            imp_inp = imp_inp[indices]
            imp_num = imp_num[indices]
            imp_cat = imp_cat[indices]

            labels = labels[indices]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(seq_cat,seq_num, ses_cat,ses_num,filt_enc, imp_inp,imp_num,imp_cat,imp_prop, sorted_len)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)

                    optimizer.step()

            # statistics
            running_loss += loss.item() * seq_cat.size(0)

            sort_pred = [np.argsort(-1*c_pred.detach().cpu().numpy()).tolist() for c_pred in outputs]
            running_cost += sumReciprocalRank(sort_pred,list(labels.cpu().numpy()))

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        val_cost = running_cost / len(dataloaders[phase].dataset)
        print('{} Loss: {:.4f}, Mean Reciprocal Rank: {:.4f}'.format(phase, epoch_loss,val_cost))

    torch.save(model.state_dict(),MODELS_PATH+'/RecNet_Model_GRU_f04g_v23dw_'+str(kf)+'_'+str(epoch+1)+'.pkl')

    # return Test set predictions:

    with torch.no_grad():  # operations inside don't track history
        model.eval()
        out_pred = []

        # Iterate over data.
        for seq_cat,seq_num, ses_cat, ses_num, filt_enc, imp_inp, imp_num, imp_cat, imp_prop, leng, labels in tqdm(dataloaders['test']):

            seq_cat = seq_cat.long().to(device)
            ses_cat = ses_cat.long().to(device)
            seq_num = seq_num.float().to(device)
            filt_enc = filt_enc.long().to(device)

            ses_num = ses_num.float().to(device)
            imp_inp = imp_inp.long().to(device)
            imp_num = imp_num.float().to(device)
            imp_cat = imp_cat.long().to(device)
            imp_prop = imp_prop.long().to(device)

            leng = leng.long().to(device)

            labels = labels.long().to(device)

            sorted_len, indices = leng.sort(descending=True)
            seq_cat = seq_cat[indices].squeeze(1)
            imp_prop = imp_prop[indices]
            ses_cat = ses_cat[indices]
            seq_num = seq_num[indices]
            ses_num = ses_num[indices]
            filt_enc = filt_enc[indices]

            imp_inp = imp_inp[indices]
            imp_num = imp_num[indices]
            imp_cat = imp_cat[indices]

            outputs = model(seq_cat,seq_num, ses_cat,ses_num,filt_enc, imp_inp,imp_num,imp_cat,imp_prop, sorted_len)
            # reorganiza o tensor na ordem correta
            ordered_out_prob = torch.empty_like(outputs)
            ordered_out_prob[indices] = outputs
            out_pred.append(np.exp(ordered_out_prob.cpu().numpy()))

    return np.concatenate(out_pred)



def setup_args_parser():
    parser = argparse.ArgumentParser(description='Create cv features')
    parser.add_argument('--processed_data_dir_name', help='path to preprocessed data', default=DEFAULT_PREPROC_DIR_NAME)
    parser.add_argument('--features_dir_name', help='features directory name', default=DEFAULT_FEATURES_DIR_NAME)
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
    logger.info('Running 021_Run_Model.py')
    logger.info(100*'-')
    logger.info('processed data directory name: %s' % args.processed_data_dir_name)
    logger.info('features directory name: %s' % args.features_dir_name)


    # processed data path
    DATA_PATH = '../data/' + args.processed_data_dir_name + '/'
    #os.makedirs(DATA_PATH) if not os.path.exists(DATA_PATH) else None
    logger.info('processed data path: %s' % DATA_PATH)

    # features data path
    FEATURES_PATH = '../features/' + args.features_dir_name + '/'
    #os.makedirs(FEATURES_PATH) if not os.path.exists(FEATURES_PATH) else None
    logger.info('features path: %s' % FEATURES_PATH)
    # End of set up arguments


    # for reproducibility"
    random_seed = 2529
    np.random.seed(random_seed) # for reproducibility"
    random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # # Cross Validation

    X_tr  = pd.read_csv(FEATURES_PATH+'new_X_tr_f01.gz', compression='gzip')
    X_val = pd.read_csv(FEATURES_PATH+'new_X_val_f01.gz', compression='gzip')
    X_tr_f02  = pd.read_csv(FEATURES_PATH+'new_X_tr_f02.gz', compression='gzip')
    X_val_f02 = pd.read_csv(FEATURES_PATH+'new_X_val_f02.gz', compression='gzip')
    X_tr = X_tr.merge(X_tr_f02, on=['session_id', 'step'], how='left')
    X_val = X_val.merge(X_val_f02, on=['session_id', 'step'], how='left')

    df_ctr = pd.read_csv(FEATURES_PATH+'Xcv_CTR_ALL.csv', compression='gzip')
    df_dwell = pd.read_csv(FEATURES_PATH+'Dcv_past_dwell_with_items_session_id.gz', compression='gzip')

    X_tr = X_tr.merge(df_ctr, on=['session_id', 'step'], how='left')
    X_val = X_val.merge(df_ctr, on=['session_id', 'step'], how='left')
    X_tr = X_tr.merge(df_dwell, on=['session_id', 'step'], how='left')
    X_val = X_val.merge(df_dwell, on=['session_id', 'step'], how='left')

    n_epochs = 4
    batch_size = 128
    n_splits = 10
    total_length = 512
    prop_length = 112

    stats_val_dict = pickle.load(open(FEATURES_PATH+OUTPUT_NORMLIZATIONS_VAL, "rb" ))
    dwell_stats_val_dict = pickle.load(open(FEATURES_PATH+'Dwell_normalizations_val.pkl', "rb" ))
    stats_val_dict['dwell_times'] = dwell_stats_val_dict['dwell_times']
    stats_val_dict['dwell_times_log'] = dwell_stats_val_dict['dwell_times_log']

    datasets = {}
    datasets['val'] = myDataset(X_val, total_length,stats_val_dict)
    dataloaders = {}
    dataloaders['val'] = DataLoader(datasets['val'], batch_size=batch_size,
                                    shuffle=False, pin_memory=False, num_workers = 4)

    all_out_pred = np.zeros((X_val.shape[0],25))
    labels = X_val['targ'].astype("int32")

    for fold in range(n_splits):

        print('fold {}/{}'.format(fold+1, n_splits))
        np.random.seed(fold*random_seed) # for reproducibility"
        random.seed(fold*random_seed)

        torch.manual_seed(fold*random_seed)
        torch.cuda.manual_seed(fold*random_seed)

        datasets['train'] = myDataset(X_tr, total_length,stats_val_dict)
        dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size,
                                      shuffle=True, pin_memory=True, num_workers = 5)

        model = RecNet().to(device)

        out_pred = train_model_val(model, dataloaders, n_epochs, batch_size)

        sort_pred = [np.argsort(-1*c_pred).tolist() for c_pred in out_pred]
        fold_cost = MeanReciprocalRank(sort_pred,list(labels))

        all_out_pred += out_pred

        sort_pred = [np.argsort(-1*c_pred).tolist() for c_pred in all_out_pred/(fold+1)]
        cost_bag = MeanReciprocalRank(sort_pred,list(labels))

        print('Fold Mean Reciprocal Rank: {:.4f}, Bag Mean Reciprocal Rank: {:.4f}'.format(fold_cost,cost_bag))

    all_out_pred /= n_splits

    GR_COLS = ["user_id", "session_id_original", "timestamp", "step",'imp_list','targ']
    df_out = X_val[GR_COLS].copy()
    df_out.rename(columns={'session_id_original': 'session_id'}, inplace=True)
    df_out['pred'] = out_pred.tolist()

    print("Writing Validation Predictions...")
    df_out.to_csv('../predictions/Val_Pred_RecNet_Model_GRU_f04g_v23dw_bag.csv', index=False)


    # # Train
    X_train  = pd.read_csv(FEATURES_PATH+'new_X_train_f01.gz', compression='gzip')
    X_test = pd.read_csv(FEATURES_PATH+'new_X_test_f01.gz', compression='gzip')

    X_train_f02  = pd.read_csv(FEATURES_PATH+'new_X_train_f02.gz', compression='gzip')
    X_test_f02 = pd.read_csv(FEATURES_PATH+'new_X_test_f02.gz', compression='gzip')

    X_train = X_train.merge(X_train_f02, on=['session_id', 'step'], how='left')
    X_test = X_test.merge(X_test_f02, on=['session_id', 'step'], how='left')

    df_ctr = pd.read_csv(FEATURES_PATH+'X_CTR_ALL.csv', compression='gzip')
    df_dwell = pd.read_csv(FEATURES_PATH+'D_past_dwell_with_items_session_id.gz', compression='gzip')

    X_train = X_train.merge(df_ctr, on=['session_id', 'step'], how='left')
    X_test = X_test.merge(df_ctr, on=['session_id', 'step'], how='left')

    X_train = X_train.merge(df_dwell, on=['session_id', 'step'], how='left')
    X_test = X_test.merge(df_dwell, on=['session_id', 'step'], how='left')

    stats_dict = pickle.load(open(FEATURES_PATH+OUTPUT_NORMLIZATIONS_SUBM, "rb" ))
    dwell_stats_dict = pickle.load(open(FEATURES_PATH+'Dwell_normalizations_submission.pkl', "rb" ))
    stats_dict['dwell_times'] = dwell_stats_dict['dwell_times']
    stats_dict['dwell_times_log'] = dwell_stats_dict['dwell_times_log']

    datasets = {}
    datasets['test'] = myDataset(X_test, total_length,stats_dict)
    dataloaders = {}
    dataloaders['test'] = DataLoader(datasets['test'], batch_size=batch_size,
                                    shuffle=False, pin_memory=False, num_workers = 4)

    all_out_pred = np.zeros((X_test.shape[0],25))

    for fold in range(n_splits):

        print('fold {}/{}'.format(fold+1, n_splits))
        np.random.seed(fold*random_seed) # for reproducibility"
        random.seed(fold*random_seed)

        torch.manual_seed(fold*random_seed)
        torch.cuda.manual_seed(fold*random_seed)

        datasets['train'] = myDataset(X_train, total_length,stats_dict)
        dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size,
                                      shuffle=True, pin_memory=True, num_workers = 5)

        model = RecNet().to(device)

        out_pred = train_model(model, dataloaders, n_epochs, batch_size)

        all_out_pred += out_pred

    all_out_pred /= n_splits


    # # Submission

    GR_COLS = ["user_id", "session_id_original", "timestamp", "step",'imp_list']
    df_out = X_test[GR_COLS].copy()
    df_out.rename(columns={'session_id_original': 'session_id'}, inplace=True)
    df_out['pre'] = all_out_pred.tolist()

    print("Writing Validation Predictions...")
    df_out.to_csv('../predictions/test_Pred_RecNet_Model_GRU_f04g_v23dw_bag.csv', index=False)

    OUTPUT_ENCODING_DICT = 'enc_dicts_v02.pkl'

    enc_dict = pickle.load(open(DATA_PATH+OUTPUT_ENCODING_DICT, "rb" ))
    inv_dict = {value:key for key, value in enc_dict['reference'].items()}


    df_out['imp_list_new'] = df_out.imp_list.apply(lambda list_s: [inv_dict[v] for v in eval(list_s) if v])

    def order_recom(row):
        Y = row.pre[:len(row.imp_list_new)]
        X = row.imp_list_new
        return ' '.join([x for _,x in sorted(zip(Y,X),reverse=True)])

    df_out['item_recommendations'] = df_out.apply(order_recom, axis=1)

    df_out.drop(['imp_list','pre','imp_list_new'], axis=1, inplace=True)

    print("Writing submission...")
    df_out.to_csv('../submissions/Sub_RecNet_Model_GRU_f04g_v23dw_bag.csv', index=False)

if __name__ == "__main__":
    main()
