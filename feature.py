# -*- coding: utf-8 -*-
# Copyright 2017 The Xiaoyu Fang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from rawdata import RawData, read_sample_data, myvalue,read_sample_data_dic
from dataset import DataSet
from chart import extract_feature,new_extract_feature
import numpy




value = -0.2
values = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4]

_testcode = '002594'
# testcodes = ['600176','002594','000725','600582','600050','600036','002456','002415']
testcodes = [_testcode]
trance_codes = ['sh','sz','hs300']
# trance_codes = ['sh','hs300']
# trance_codes = ['sz','sz50']

allcodes = ['600176','002594','000725','600582','600050','600036','002456','002415']


for i in range(len(trance_codes)):
    trance_codes[i] = trance_codes[i] +".csv"
depend_features = len(trance_codes)

#feature值是一定的,和talib有关
input_shape_mine = [70, (7+1+1+11-7+7-6-6+6)*len(trance_codes)]  # [length of time series, length of feature]


dirpath = "dataset/myvalue/"

days_for_test = 100

def extract_from_file(filepath, output_prefix):
    window = input_shape_mine[0]
    fp = open("%s_feature.%s" % (output_prefix, window), "w")
    lp = open("%s_label.%s" % (output_prefix, window), "w")
    fpt = open("%s_feature.test.%s" % (output_prefix, window), "w")
    lpt = open("%s_label.test.%s" % (output_prefix, window), "w")

    selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"]

    dataset_dir = "./"+filepath
    # trance_codes = os.listdir(dataset_dir)
    moving_features = []
    moving_labels = []

    filepath = dataset_dir + "/" + _testcode +'.csv'
    raw_data_code = read_sample_data(filepath)
    moving_feature, moving_label = extract_feature(raw_data=raw_data_code, selector=selector, window=input_shape_mine[0],with_label=True, flatten=True)
    moving_labels = moving_label
    moving_features.append(moving_feature)

    for filename in trance_codes:
        if filename == _testcode +'.csv':
            continue
        raw_data=[]
        print("processing file: " + filename)
        filepath = dataset_dir + "/" + filename
        raw_data_dic = read_sample_data_dic(filepath)
        temparr = []
        for i in raw_data_code:
            item = raw_data_dic[i.date]
            temparr.append(item)
        raw_data=temparr          
        moving_feature, moving_label = extract_feature(raw_data=raw_data, selector=selector, window=input_shape_mine[0],
                                                         with_label=True, flatten=True)
        print("feature extraction done, start writing to file...")


        moving_features.append(moving_feature)
    for i in range(len(moving_features)):
        moving_features[i] = moving_features[i].tolist()
    combine_features = []
    for i in range(moving_labels.shape[0]):
        temparr = []
        for j in range(input_shape_mine[0]):
            for k in range(len(trance_codes)):
                temparr += moving_features[k][i][j*input_shape_mine[1]/len(trance_codes):(j+1)*input_shape_mine[1]/len(trance_codes)]
        combine_features.append(temparr)

    moving_features = combine_features
    train_end_test_begin = len(combine_features) - days_for_test

    if train_end_test_begin < 0:
        train_end_test_begin = 0
    for i in range(0, train_end_test_begin):
        for item in moving_features[i]:
            fp.write("%s\t" % item)
        fp.write("\n")
    for i in range(0, train_end_test_begin):
        lp.write("%s\n" % moving_labels[i])
    # test set
    for i in range(train_end_test_begin, len(moving_features)):
        for item in moving_features[i]:
            fpt.write("%s\t" % item)
        fpt.write("\n")
    for i in range(train_end_test_begin, len(moving_labels)):
        lpt.write("%s\n" % moving_labels[i])

    fp.close()
    lp.close()
    fpt.close()
    lpt.close()


def test(rvalue,r_testcode,rdays_for_test):
    global value,_testcode,days_for_test,testcodes
    value=rvalue
    _testcode=r_testcode
    days_for_test=rdays_for_test
    testcodes=[_testcode]

# if __name__ == '__main__':
def featuremain():
    
    for testcode in testcodes:
    
        print(days_for_test)
        window = input_shape_mine[0]
        fp = open("ultimate_feature.%s" % window, "w")
        lp = open("ultimate_label.%s" % window, "w")
        fpt = open("ultimate_feature.test.%s" % window, "w")
        lpt = open("ultimate_label.test.%s" % window, "w")

        selector = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"]
        dataset_dir = "./"+dirpath
        # trance_codes = os.listdir(dataset_dir)
        moving_features = []
        moving_labels = []

        filepath = dataset_dir + "/" + testcode +'.csv'
        raw_data_code = read_sample_data(filepath)
        moving_feature, moving_label = extract_feature(raw_data=raw_data_code, selector=selector, window=input_shape_mine[0],with_label=True, flatten=True)
        moving_labels = moving_label
        moving_features.append(moving_feature)

        for filename in trance_codes:
            if filename == testcode +'.csv':
                continue
            raw_data=[]
            print("processing file: " + filename)
            filepath = dataset_dir + "/" + filename
            raw_data_dic = read_sample_data_dic(filepath)
            temparr = []
            for i in raw_data_code:
                item = raw_data_dic[i.date]
                temparr.append(item)
            raw_data=temparr          
            moving_feature, moving_label = new_extract_feature(raw_data=raw_data, selector=selector, window=input_shape_mine[0],
                                                         with_label=True, flatten=True)
            print("feature extraction done, start writing to file...")


            moving_features.append(moving_feature)
        for i in range(len(moving_features)):
            moving_features[i] = moving_features[i].tolist()
        combine_features = []
        for i in range(moving_labels.shape[0]):
            temparr = []
            for j in range(input_shape_mine[0]):
                for k in range(len(trance_codes)):
                    temparr += moving_features[k][i][j*input_shape_mine[1]/len(trance_codes):(j+1)*input_shape_mine[1]/len(trance_codes)]
            combine_features.append(temparr)

        moving_features = combine_features
        train_end_test_begin = len(combine_features) - days_for_test

        if train_end_test_begin < 0:
            train_end_test_begin = 0
        for i in range(0, train_end_test_begin):
            for item in moving_features[i]:
                fp.write("%s\t" % item)
            fp.write("\n")
        for i in range(0, train_end_test_begin):
            lp.write("%s\n" % moving_labels[i])
        # test set
        for i in range(train_end_test_begin, len(moving_features)):
            for item in moving_features[i]:
                fpt.write("%s\t" % item)
            fpt.write("\n")
        for i in range(train_end_test_begin, len(moving_labels)):
            lpt.write("%s\n" % moving_labels[i])


        fp.close()
        lp.close()
        fpt.close()
        lpt.close()
