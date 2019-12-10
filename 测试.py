# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:31:56 2019

@author: fan
"""

import csv
import heapq#找出List中最大/最小的N个数及索引
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.spatial import distance
from pandas import Series
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


if __name__ == '__main__':
    
    mark = Mark()
    score = []#存储numpy数组的二维列表
    for index in range(len(file_name)):
        score.append(mark.processing(file_name[index],file_name[index],index))
    each_outlier_num[16] = 427
    each_cleaned_num = list(map(lambda x: x[0]-x[1], zip(each_origin_num, each_outlier_num)))
    #print(score[0])
    #print(score)
    #以上部分为预处理部分，形成矩阵
    
    choise = Chose()
    file_num = input()
    #choise.switch_case_file(file_num)
    p = Preprocessing()
    data = p.file_load(file_path_origin_data+choise.switch_case_file(int(file_num)))
    data = p.maxminnorm(data)
    finalData =np.array(p.PCA(data,2))#对数据进行降维
    
    data_out = p.file_load(file_path_outlier_data+choise.switch_case_file(int(file_num)))
    data_out = p.maxminnorm(data_out)
    p.picture(data,data_out)
    #以上部分读取待处理数据并进行归一化和降维

    #Normal_distribution(finalData,1.50,int(file_num))
    #以上部分采用3sigma原则处理数据

    #Neighbor(finalData,int(file_num))
    #以上部分采用基于临近的原则处理数据
    
    #Kmeans(finalData,int(file_num))
    #以上部分采用基于聚类的原则处理数据
    
    #LOF(finalData,int(file_num))
    #以上部分基于密度的异常检测算法,该算法会给数据集中的每个点计算一个离群因子LOF，
    #通过判断LOF是否接近于1来判定是否是离群因子。若LOF远大于1，则认为是离群因子，接近于1，则是正常点。
    
    p1 = Preprocessing()
    data_clean = p1.file_load(file_path_cleaned_data+choise.switch_case_file(int(file_num)))
    data_clean = p1.maxminnorm(data_clean)
    finalDataClean =np.array(p1.PCA(data_clean,2))#对数据进行降维
    
    #OneClassSVM(finalDataClean,finalData,int(file_num))
    #一分类，使用清洁数据进行训练，无监督异常值检测
    
    IForest(data_clean,data,finalData,int(file_num))
    f.close()
    #孤立森林，基于密度
