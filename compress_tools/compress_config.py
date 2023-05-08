#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/2 3:15
# @Author  : zfgao
# @File    : compress_config.py
import os
#config_dict = {'attention_probs_dropout_prob':0.25}
N = 30522
config_dict = {"compress": False,
               "compress_version" : 2, # version in BERTCompress_v4.py, eg.EmbeddingDecomV{}.format(compress_version)
               "teacher_name_or_path" : "/home/liupeiyu/bert_compress/transformers/examples/text-classification/tmp_compressemb/SST-2",
               "alpha" : 0.5,
               "loss_type" : "mse",
               "row" : 2000,
               "temperature" : 0.5,
               "loss_we" : "",
               "teacher" : False,
               "intermediate_size":3072,
               "mpo":True} #loss_type: string, option:"mse" for nn.MSELoss, "kd" for nn.KLDivLoss

base_dir = '/home/liupeiyu/bert_compress/transformers/examples/text-classification/tmp_file/'
high_pt = os.path.join(base_dir, 'high_wd_raw.npy')
low_pt = os.path.join(base_dir, 'low_wd_raw.npy')
emb1_pt = os.path.join(base_dir, 'emb1_kmeans.npy')
emb2_pt = os.path.join(base_dir, 'emb2_kmeans.npy')
emb1_N = os.path.join(base_dir, 'emb1_N.npy')
emb2_N = os.path.join(base_dir, 'emb2_N.npy')
we_bothN_pt = os.path.join(base_dir, 'we_bothN.npy')
# ---- old map: from word_id to cluster
# emb1_map_pt = os.path.join(base_dir, 'emb1_map.npy')
# emb2_map_pt = os.path.join(base_dir, 'emb2_map.npy')
# ---- new map:  from word_id to cluster(low freq) and word_id to word_id(high freq)
# emb1_map_pt = os.path.join(base_dir, 'emb1_map_high.npy')
# emb2_map_pt = os.path.join(base_dir, 'emb2_map_high.npy')
# ---- new map:  from word_id to cluster(high freq) and word_id to word_id(low freq)
emb1_map_pt = os.path.join(base_dir, 'emb1_map_low.npy')
emb2_map_pt = os.path.join(base_dir, 'emb2_map_low.npy')
# emb after BertEmbeddingTrain
emb1_train_pt = '/home/liupeiyu/bert_compress/data/embtrain_left.npy'
emb2_train_pt = '/home/liupeiyu/bert_compress/data/embtrain_right.npy'

# mpo setting
mpo=True
