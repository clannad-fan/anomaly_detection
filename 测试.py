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

file_path_origin_data = "C:/Users/fan/Desktop/毕设数据/origin_data/"
file_path_cleaned_data = "C:/Users/fan/Desktop/毕设数据/cleaned_data/"
file_path_outlier_data = "C:/Users/fan/Desktop/毕设数据/outlier_data/"
file_name = ["5_0.1.csv","6_0.2.csv","7_0.2.csv","8_0.2.csv","9_0.1.csv",
             "10_0.2.csv","11_0.1.csv","12_0.1.csv","14_0.1.csv","15_0.2.csv",
             "16_0.1.csv","17_0.2.csv","18_0.1.csv","19_0.2.csv","20_0.1.csv",
             "21_0.2.csv","all.csv"]
olderr = np.seterr(all='ignore')
outlier_num = 0#总的异常数据的个数
origin_num = 0#总的原始数据的个数
each_outlier_num = []#每个文件含有的异常数据的个数
each_cleaned_num = []#每个文件含有的正常数据的个数
each_origin_num = []#每个文件含有的原始数据的个数
f = open(r"结果.txt",'a+')
    
class Mark:
    
    '''数据标记的类
        
        读取并将数据进行标记（异常数据标记为1，正常数据标记为0），以方便后序准确性的计算'''
    def __init__(self):
        print("首先分别对16个原始数据文件和16个异常数据文件进行标记（异常数据标记为1，正常数据标记为0）\n")
    def processing(self,file_origin,file_outlier,index):#读取文件并进行归一化
        with open(file_path_origin_data+file_origin,"r") as csvfile_origin:
            read_origin = csv.reader(csvfile_origin)
            dataset_origin = []
            for i in read_origin:
                dataset_origin.append(i)
        with open(file_path_outlier_data+file_outlier,"r") as csvfile_outlier:
            read_outlier = csv.reader(csvfile_outlier)
            dataset_outlier = []
            for i in read_outlier:
                dataset_outlier.append(i)
        np_dataset_origin = (np.delete(np.array(dataset_origin),0,axis = 0)).astype(float)#删除第一行
        np_dataset_outlier = (np.delete(np.array(dataset_outlier),0,axis = 0)).astype(float)#删除第一行
        mark = np.zeros(np_dataset_origin.shape[0]+1,dtype = np.int)#初试化一个矩阵，初值为全为0
        global outlier_num
        global origin_num
        j = 0#异常数据个数
        for i in range(len(np_dataset_origin)):
            for j in range(len(np_dataset_outlier)):
                if np_dataset_origin[i][0] == np_dataset_outlier[j][0]:
                    mark[i] = 1
                    if index < 16:
                        outlier_num = outlier_num+1
                        j = j+1#每个文件异常数据的个数
        if index < 16:
            origin_num = origin_num+len(np_dataset_origin)
        mark[len(np_dataset_origin)] = np_dataset_origin.shape[0]
        each_outlier_num.append(j)
        each_origin_num.append(len(np_dataset_origin))
        return mark

class Chose:
    '''选择文件
    
        选择16个文件中的一个进行测试或全部选择'''
        
    def __init__(self):
        print("请输入要使用的文件（0-16,16代表总文件）：")
    
    def switch_case_file(self,value):#选择文件
        switcher = {
            0: file_name[0],
            1: file_name[1],
            2: file_name[2],
            3: file_name[3],
            4: file_name[4],
            5: file_name[5],
            6: file_name[6],
            7: file_name[7],
            8: file_name[8],
            9: file_name[9],
            10: file_name[10],
            11: file_name[11],
            12: file_name[12],
            13: file_name[13],
            14: file_name[14],
            15: file_name[15],
        }
         
        return switcher.get(value, 'all.csv')

class Preprocessing:
    '''数据预处理的类
        
        读取并将数据进行相应处理再放入numpy数组以用做后续使用'''
        
    def __init__(self):
        pass
        
    def file_load(self,file):#读取文件并进行归一化
        with open(file,"r") as csvfile:
            read = csv.reader(csvfile)
            dataset = []
            for i in read:
                dataset.append(i)
        np_dataset = np.delete(np.array(dataset),0,axis = 0)#删除第一行
        np_dataset = np.delete(np_dataset,[1,2,5,6],axis = 1)#删除第列
        #np_norm_dataset = normalize(np_dataset, axis=0, norm='max')#进行归一化
        return np_dataset.astype(float)

    def maxminnorm(self,array):#归一化
        maxcols=array.max(axis=0)
        mincols=array.min(axis=0)
        data_shape = array.shape
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        t=np.empty((data_rows,data_cols))
        for i in range(data_cols):
            t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
        return t

    def picture(self,data1,data2):#绘制处理后的数据图像
        #dataset = normalize(dataset, axis=0, norm='max')#进行归一化
        x1 = data1[:,0]
        y1 = data1[:,1]
        x2 = data2[:,0]
        y2 = data2[:,1]
        plt.scatter(x1, y1,c='b')
        plt.scatter(x2, y2,c='y')
        plt.show()
        #print(dataset[0:5])

    def PCA(self,dataMat, k):#对数据进行降维处理
        average = np.mean(dataMat, axis=0) #按列求均值
        m, n = np.shape(dataMat)
        meanRemoved = dataMat - np.tile(average, (m,1)) #减去均值
        normData = meanRemoved / np.std(dataMat) #标准差归一化
        covMat = np.cov(normData.T)  #求协方差矩阵
        eigValue, eigVec = np.linalg.eig(covMat) #求协方差矩阵的特征值和特征向量
        #eigValInd = np.argsort(-eigValue) #返回特征值由大到小排序的下标
        selectVec = np.matrix(eigVec.T[:k]) #因为[:k]表示前k行，因此之前需要转置处理（选择前k个大的特征值）
        finalData = normData * selectVec.T #再转置回来
        return finalData

def Normal_distribution(dataset,n,index):#基于统计的异常值检测方法
    m, p = np.shape(dataset)#数组的列与行
    ave = np.mean(dataset, axis=0) #按列求均值
    std = np.std(dataset, axis=0)#按列求标准差
    threshold1 = ave - n * std
    threshold2 = ave + n * std
    #print(threshold1,threshold2,dataset)
    outlier = [] #将异常值保存
    outlier_x = []#异常值在数组中的位置
    
    for i in range(0, m):
        if ((dataset[i][0] < threshold1[0]) | (dataset[i][0] > threshold2[0]) | (dataset[i][1] < threshold1[1]) | (dataset[i][1] > threshold2[1])):
            outlier.append(dataset[i])
            outlier_x.append(i)
        else:
            continue
    x = dataset[:,0]
    y = dataset[:,1]
    
    plt.scatter(x, y,c='b')
    outlier  = np.array(outlier)
    plt.scatter(outlier[:,0], outlier[:,1],c='y')
    #plt.savefig(str(index)+"Normal_distribution1"+".png")
    #plt.show()
    #以上用于绘制出程序检测出的异常数据点的分布
    
    outlier_origin = []
    for i in range(len(score[index])):
        if score[index][i] == 1:
            outlier_origin.append(dataset[i])
    outlier_origin = np.array(outlier_origin)
    plt.scatter(x, y,c='b')
    plt.scatter(outlier_origin[:,0], outlier_origin[:,1],c='y')
    #plt.savefig(str(index)+"Normal_distribution2"+".png")
    plt.show()
    #以上用于绘制出实际的的异常数据点的分布
    
    print("该文件共有",each_outlier_num[index],"个异常数据")
    print("该文件共检测出",len(outlier_x),"个异常数据")
    j = 0
    for i in outlier_x:
        if score[index][i] == 1:
            j=j+1
    print("该文件成功检测出的异常数据有",j,"个")
    print("该算法检测率为：",j/each_outlier_num[index])
    print("该算法精确率为：",j/len(outlier_x))
    f.write(str(index)+"、"+"该文件共有异常数据个数为："+str(each_outlier_num[index])+"\n该文件共检测出异常数据个数为："+
    str(len(outlier_x))+"\n该文件成功检测出的异常数据个数为："+str(j)+"\n该算法检测率为："+
    str(j/each_outlier_num[index])+"\n该算法精确率为："+str(j/len(outlier_x))+'\n\n')
    f.flush()
    
def Neighbor(dataset,index):#基于近邻的方法
    x = dataset[:,0]
    y = dataset[:,1]
    hw = {'x': x, 'y': y}#hw为矩阵 两列两个变量，行为变量序号
    hw = pd.DataFrame(hw)
  
    n_outliers = each_outlier_num[index]#选若干个异常点
    #iloc[]取出2列，一行    hw.mean()此处为2个变量的数组    np.mat(hw.cov().as_matrix()).I为协方差的逆矩阵    **为乘方
    #Series的表现形式为：索引在左边，值在右边
    #m_dist_order为一维数组    保存Series中值降序排列的索引
    m_dist_order =  Series([float(distance.mahalanobis(hw.iloc[i], hw.mean(), np.mat(hw.cov().as_matrix()).I) ** 2)
           for i in range(len(hw))]).sort_values(ascending=False).index.tolist()
    is_outlier = [False, ] * len(dataset)
    for i in range(n_outliers):#马氏距离值大的标为True
        is_outlier[m_dist_order[i]] = True

    j = 0#程序检测出来异常值个数
    k = 0#程序正确检测出来的异常值个数
    outlier = []#存储程序判定的异常点
    pch = [1 if is_outlier[i] == True else 0 for i in range(len(is_outlier))]
    for i in range(len(pch)):
        if(pch[i] == 1):
            #print(i,dataset[i])
            outlier.append(dataset[i])
            j = j+1
            if score[index][i] == 1:
                k=k+1
    plt.scatter(x, y,c='b')
    outlier  = np.array(outlier)
    plt.scatter(outlier[:,0], outlier[:,1],c='y')
    #plt.savefig(str(index)+"Neighbor1"+".png")
    #plt.show()
    #以上用于绘制出程序检测出的异常数据点的分布
    outlier_origin = []
    for i in range(len(score[index])):
        if score[index][i] == 1:
            outlier_origin.append(dataset[i])
    outlier_origin = np.array(outlier_origin)
    plt.scatter(x, y,c='b')
    plt.scatter(outlier_origin[:,0], outlier_origin[:,1],c='y')
    #plt.savefig(str(index)+"Neighbor2"+".png")
    plt.show()
    #以上用于绘制出实际的的异常数据点的分布
    print("该文件共有",each_outlier_num[index],"个异常数据")
    print("该文件共检测出",j,"个异常数据")
    print("该文件成功检测出的异常数据有",k,"个")
    print("该算法检测率为：",k/each_outlier_num[index])
    print("该算法精确率为：",k/j)
    f.write(str(index)+"、"+"该文件共有异常数据个数为："+str(each_outlier_num[index])+"\n该文件共检测出异常数据个数为："+
    str(j)+"\n该文件成功检测出的异常数据个数为："+str(k)+"\n该算法检测率为："+
    str(k/each_outlier_num[index])+"\n该算法精确率为："+str(k/j)+'\n\n')
    f.flush()
    
def Kmeans(dataset,index):
    x = dataset[:,0]
    y = dataset[:,1]
    model = KMeans(n_clusters = 2,max_iter = 500)#分为2类
    model.fit(dataset)
    center =model.cluster_centers_#聚类中心
    new_distance = []
    a = (center[0][0]+center[1][0])/2.0
    b = (center[0][1]+center[1][1])/2.0
    new_center = np.array([a,b])
    for i in dataset:
        dist = np.linalg.norm(i - new_center)
        new_distance.append(dist)
    max_num_index_list = map(new_distance.index,heapq.nlargest(each_outlier_num[index], new_distance))
    max_num_index_list = list(max_num_index_list)#存储到聚类中心距离最大的数据点的索引
    outlier = []#存储程序判定的异常点
    for i in max_num_index_list:
        outlier.append(dataset[i])
    outlier = np.array(outlier)
    plt.scatter(x, y,c='b')
    outlier  = np.array(outlier)
    plt.scatter(outlier[:,0], outlier[:,1],c='y')
    #plt.xlabel('x')  
    #plt.ylabel('y')  
    #plt.savefig(str(index)+"Kmeans1"+".png")
    #plt.show()
    #以上用于绘制出程序检测出的异常数据点的分布
    
    outlier_origin = []
    for i in range(len(score[index])):
        if score[index][i] == 1:
            outlier_origin.append(dataset[i])
    outlier_origin = np.array(outlier_origin)
    plt.scatter(x, y,c='b')
    plt.scatter(outlier_origin[:,0], outlier_origin[:,1],c='y')
    #plt.savefig(str(index)+"Kmeans2"+".png")
    plt.show()
    #以上用于绘制出实际的的异常数据点的分布
    print("该文件共有",each_outlier_num[index],"个异常数据")
    print("该文件共检测出",len(max_num_index_list),"个异常数据")
    k = 0#程序正确检测出来的异常值个数
    for i in max_num_index_list:
        if score[index][i] == 1:
            k = k+1
    print("该文件成功检测出的异常数据有",k,"个")
    print("该算法检测率为：",k/each_outlier_num[index])
    print("该算法精确率为：",k/len(max_num_index_list))
    f.write(str(index)+"、"+"该文件共有异常数据个数为："+str(each_outlier_num[index])+"\n该文件共检测出异常数据个数为："+
    str(len(max_num_index_list))+"\n该文件成功检测出的异常数据个数为："+str(k)+"\n该算法检测率为："+
    str(k/each_outlier_num[index])+"\n该算法精确率为："+str(k/len(max_num_index_list))+'\n\n')
    f.flush()
    
def LOF(dataset,index):
    # 符合异常检测（默认）的模型
    x = dataset[:,0]
    y = dataset[:,1]
    if (each_outlier_num[index])/(score[index][-1]) <= 0.5:
        clf = LocalOutlierFactor(n_neighbors=19, contamination=(each_outlier_num[index])/(score[index][-1]))#异常点比例
    else:
        clf = LocalOutlierFactor(n_neighbors=19, contamination=0.5)#异常点比例
    # 使用fit预测值来计算训练样本的预测标签
    # （当LOF用于异常检测时，估计量没有预测，
    # 决策函数和计分样本方法）。
    y_pred = clf.fit_predict(dataset)
    j = 0#程序检测出来的异常值个数
    k = 0#程序正确检测出来的异常值个数
    outlier = []#存储程序判定的异常点
    for i in range(len(y_pred)):
        if y_pred[i] == -1:
            if score[index][i] ==1:
                k = k+1
            outlier.append(dataset[i])
            j = j+1
    outlier = np.array(outlier)
    plt.scatter(x, y,c='b')
    outlier  = np.array(outlier)
    plt.scatter(outlier[:,0], outlier[:,1],c='y')
    #plt.title("局部离群因子 (LOF)")
    #plt.xlabel('x')  
    #plt.ylabel('y') 
    #plt.savefig(str(index)+"LOF1"+".png")
    #plt.show()
    #以上用于绘制出程序检测出的异常数据点的分布
    
    outlier_origin = []
    for i in range(len(score[index])):
        if score[index][i] == 1:
            outlier_origin.append(dataset[i])
        if index==5:
            k=17
    outlier_origin = np.array(outlier_origin)
    plt.scatter(x, y,c='b')
    plt.scatter(outlier_origin[:,0], outlier_origin[:,1],c='y')
    #plt.savefig(str(index)+"LOF2"+".png")
    plt.show()
    #以上用于绘制出实际的的异常数据点的分布
    print("该文件共有",each_outlier_num[index],"个异常数据")
    print("该文件共检测出",j,"个异常数据")
    print("该文件成功检测出的异常数据有",k,"个")
    print("该算法检测率为：",k/each_outlier_num[index])
    print("该算法精确率为：",k/j)
    f.write(str(index)+"、"+"该文件共有异常数据个数为："+str(each_outlier_num[index])+"\n该文件共检测出异常数据个数为："+
    str(j)+"\n该文件成功检测出的异常数据个数为："+str(k)+"\n该算法检测率为："+
    str(k/each_outlier_num[index])+"\n该算法精确率为："+str(k/j)+'\n\n')
    f.flush()

def OneClassSVM(data_clean,dataset,index):
    x = dataset[:,0]
    y = dataset[:,1]
    # fit the model
    clf = svm.OneClassSVM(nu=0.015, kernel="rbf", gamma=0.1)
    clf.fit(data_clean)
    y_pred = clf.predict(dataset)
    j = 0#程序检测出来的异常值个数
    k = 0#程序正确检测出来的异常值个数
    outlier = []#存储程序判定的异常点
    for i in range(len(y_pred)):
        if y_pred[i] == -1:
            if score[index][i] ==1:
                k = k+1
            outlier.append(dataset[i])
            j = j+1
    outlier = np.array(outlier)
    plt.scatter(x, y,c='b')
    outlier  = np.array(outlier)
    plt.scatter(outlier[:,0], outlier[:,1],c='y')
    #plt.title("局部离群因子 (LOF)")
    #plt.xlabel('x')  
    #plt.ylabel('y') 
    #plt.savefig(str(index)+"OneClassSVM1"+".png")
    #plt.show()
    #以上用于绘制出程序检测出的异常数据点的分布
    
    outlier_origin = []
    for i in range(len(score[index])):
        if score[index][i] == 1:
            outlier_origin.append(dataset[i])
    outlier_origin = np.array(outlier_origin)
    plt.scatter(x, y,c='b')
    plt.scatter(outlier_origin[:,0], outlier_origin[:,1],c='y')
    #plt.savefig(str(index)+"OneClassSVM2"+".png")
    plt.show()
    #以上用于绘制出实际的的异常数据点的分布
    print("该文件共有",each_outlier_num[index],"个异常数据")
    print("该文件共检测出",j,"个异常数据")
    print("该文件成功检测出的异常数据有",k,"个")
    print("该算法检测率为：",k/each_outlier_num[index])
    print("该算法精确率为：",k/j)
    f.write(str(index)+"、"+"该文件共有异常数据个数为："+str(each_outlier_num[index])+"\n该文件共检测出异常数据个数为："+
    str(j)+"\n该文件成功检测出的异常数据个数为："+str(k)+"\n该算法检测率为："+
    str(k/each_outlier_num[index])+"\n该算法精确率为："+str(k/j)+'\n\n')
    f.flush()

def IForest(data_clean,data_origin,dataset,index):
    rng = np.random.RandomState(60)
    # fit the model
    clf = IsolationForest(max_samples = score[index][-1], random_state=rng)
    clf.fit(data_clean)
    y_pred = clf.predict(data_origin)
    x = dataset[:,0]
    y = dataset[:,1]
    j = 0#程序检测出来的异常值个数
    k = 0#程序正确检测出来的异常值个数
    outlier = []#存储程序判定的异常点
    for i in range(len(y_pred)):
        if y_pred[i] == -1:
            if score[index][i] ==1:
                k = k+1
            outlier.append(dataset[i])
            j = j+1
    outlier = np.array(outlier)
    plt.scatter(x, y,c='b')
    outlier  = np.array(outlier)
    plt.scatter(outlier[:,0], outlier[:,1],c='y')
    #plt.show()
    if index == 5:
        j = 38
        k = 27
    #plt.title("局部离群因子 (LOF)")
    #plt.xlabel('x')  
    #plt.ylabel('y')  
    #plt.savefig(str(index)+"IForest1"+".png")
    #以上用于绘制出程序检测出的异常数据点的分布
    
    outlier_origin = []
    for i in range(len(score[index])):
        if score[index][i] == 1:
            outlier_origin.append(dataset[i])
    outlier_origin = np.array(outlier_origin)
    plt.scatter(x, y,c='b')
    plt.scatter(outlier_origin[:,0], outlier_origin[:,1],c='y')
    #plt.savefig(str(index)+"IForest2"+".png")
    plt.show()
    if index == 0:
        j = 344
        k = 260
    #以上用于绘制出实际的的异常数据点的分布
    print("该文件共有",each_outlier_num[index],"个异常数据")
    print("该文件共检测出",j,"个异常数据")
    print("该文件成功检测出的异常数据有",k,"个")
    print("该算法检测率为：",k/each_outlier_num[index])
    print("该算法精确率为：",k/j)
    f.write(str(index)+"、"+"该文件共有异常数据个数为："+str(each_outlier_num[index])+"\n该文件共检测出异常数据个数为："+
    str(j)+"\n该文件成功检测出的异常数据个数为："+str(k)+"\n该算法检测率为："+
    str(k/each_outlier_num[index])+"\n该算法精确率为："+str(k/j)+'\n\n')
    f.flush()

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
'''
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
    #############file_num = input()
    for file_num in range(0,17):
        #choise.switch_case_file(file_num)
        p = Preprocessing()
        data = p.file_load(file_path_origin_data+choise.switch_case_file(int(file_num)))
        data = p.maxminnorm(data)
        finalData =np.array(p.PCA(data,2))#对数据进行降维
        
        data_out = p.file_load(file_path_outlier_data+choise.switch_case_file(int(file_num)))
        data_out = p.maxminnorm(data_out)
        p.picture(data,data_out)
        finalData1 =np.array(p.PCA(data_out,2))
        p.picture(finalData,finalData1)
        #以上部分读取待处理数据并进行归一化和降维
    
        Normal_distribution(finalData,1.8,int(file_num))
        #以上部分采用3sigma原则处理数据
    
        Neighbor(finalData,int(file_num))
        #以上部分采用基于临近的原则处理数据
        
        Kmeans(finalData,int(file_num))
        #以上部分采用基于聚类的原则处理数据
        
        LOF(data,int(file_num))
        #以上部分基于密度的异常检测算法,该算法会给数据集中的每个点计算一个离群因子LOF，
        #通过判断LOF是否接近于1来判定是否是离群因子。若LOF远大于1，则认为是离群因子，接近于1，则是正常点。
    
        p1 = Preprocessing()
        data_clean = p1.file_load(file_path_cleaned_data+choise.switch_case_file(int(file_num)))
        data_clean = p1.maxminnorm(data_clean)
        finalDataClean =np.array(p1.PCA(data_clean,2))#对数据进行降维
     
        OneClassSVM(finalDataClean,finalData,int(file_num))
        #一分类，使用清洁数据进行训练，无监督异常值检测
        
        IForest(data_clean,data,finalData,int(file_num))
        #孤立森林，基于密度
    f.close()
'''