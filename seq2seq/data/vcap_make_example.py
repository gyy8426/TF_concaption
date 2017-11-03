import tensorflow as tf
import os
import numpy as np 
import cPickle as pkl
from tqdm import tqdm #progress bar
'''''' 
def make_example(feat,cap_ids,cap_ids_token):
    example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
            "video/data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[feat]))
            #"video/data":tf.train.Features(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=video_filed[i])) for i in range(len(video_filed))]),
            #"video/data":tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=video_filed[i])) for i in range(len(video_filed))])
        }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
            #"video":tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=feat[i])) for i in range(len(feat))]),
            "video/caption_ids":tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[cap_ids[i]])) for i in range(len(cap_ids))]),
            "video/caption":tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[cap_ids_token[i]])) for i in range(len(cap_ids_token)) ])
            }
        )
    )
    return example.SerializeToString()
    
def make_example_feat(feat,cap_ids,cap_ids_token):
    example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
            "video/data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[feat]))
            #"video/data":tf.train.Features(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=video_filed[i])) for i in range(len(video_filed))]),
            #"video/data":tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=video_filed[i])) for i in range(len(video_filed))])
        })
    )
    return example.SerializeToString()
dataset = "MSR-VTT"
feat_path = "/mnt/disk3/guoyuyu/datasets/"+dataset+"/features/Resnet/"
per_path = "/mnt/disk3/guoyuyu/datasets/"+dataset+"/predatas/"
out_path = "/mnt/disk3/guoyuyu/datasets/"+dataset+"/"    
dict_map = pkl.load(open(per_path+'worddict_small.pkl'))
def numpytotfrecorder(target_name, source_name, CAP, dataset="train"):
    writer = tf.python_io.TFRecordWriter(target_name)
    print("target_name: ",target_name)
    for id_i in tqdm(source_name):
        feat_i = np.load(feat_path+id_i+'.npy').astype('float32')
        cap_i =  CAP[id_i]
        cap_ids = []
        cap_token = []
        for cap_i_j in cap_i:
            #cap_ids_j.append(int(cap_i_j['cap_id']))
            if dataset == "MSR-VTT":
                token_j = cap_i_j['tokenized'].encode("utf-8")
            if dataset == "MSVD":
                token_j = cap_i_j['tokenized']                
            rval = token_j.split(' ')
            words = [w for w in rval if w != '']
            words_id = [dict_map[w]
                         if w in dict_map else 1 for w in words]
            if dataset == "train":
                tem = make_example(feat_i.tostring(), words_id, words)
                writer.write(tem)
        if dataset == "test" or dataset == "valid":
            tem = make_example_feat(feat_i.tostring(), words_id, words)
            writer.write(tem)
    writer.close()
    return None
if dataset == "MSVD":
# MSVD dataset 
    train_ids = ['vid%s'%i for i in range(1,1201)]
    valid_ids = ['vid%s'%i for i in range(1201,1301)]
    test_ids = ['vid%s'%i for i in range(1301,1971)]

#MSR-VTT dataset
if dataset == "MSR-VTT":
    train_ids = ['video%s'%i for i in range(0,6513)]
    valid_ids = ['video%s'%i for i in range(6513,7910)]
    test_ids = ['video%s'%i for i in range(7910,10000)]
CAP = pkl.load(open(per_path + 'CAP.pkl'))
'''
filename=out_path + dataset +"_train_feat_ResNet152_pool5_2048_pair.tfrecords"
_ = numpytotfrecorder(filename,train_ids,CAP)
''' 
filename=out_path + dataset +"_valid_feat_ResNet152_pool5_2048_feat.tfrecords"
_ = numpytotfrecorder(filename,valid_ids,CAP,'valid')
filename=out_path + dataset +"_test_feat_ResNet152_pool5_2048_feat.tfrecords"
_ = numpytotfrecorder(filename,test_ids,CAP,'test')

'''
filename = "temp.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for train_id_i in train_ids[:3]:
    feat_i = np.load(feat_path+train_id_i+'.npy').astype('float32')
    cap_i =  CAP[train_id_i]
    cap_ids = []
    cap_token = []
    for cap_i_j in cap_i:
        cap_ids.append(int(cap_i_j['cap_id']))
        cap_token.append([cap_i_j['tokenized']])
    tem = make_example(feat_i.tostring(),cap_ids,cap_token)
    writer.write(tem)
writer.close()   
'''

