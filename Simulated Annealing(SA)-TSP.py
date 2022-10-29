# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:44:11 2022

@author: 18194
"""

import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time

city_location = pd.read_excel(r"D:\python\Intelligent Optimizaition Algorithm\dataset\Oliver30.xlsx")

def set_distance(df):
    """
    输入一个dataframe，得到各点间的距离
    --------------
    parameter:
        df:dataframe
    --------------
    return:
        distance_matrix:list
    """
    city_num = len(list(df.iloc[:,1]))
    city_matrix = []
    for i in range(0,city_num):
        city_matrix.append(list(df.iloc[i,0:]))
    distance_matrix = []
    for i in range(0,city_num):
        row_distance = []#i城市到各城市的距离
        for j in range(0,city_num):
            #欧几里得距离
            x = abs(city_matrix[i][1] - city_matrix[j][1])
            y = abs(city_matrix[i][2] - city_matrix[j][2])
            distance = math.sqrt(x**2+y**2)
            row_distance.append(distance)
        distance_matrix.append(row_distance)
    return city_num,city_matrix,distance_matrix
city_num,city_matrix,dis = set_distance(city_location)
city_list = [i for i in range(0,city_num)]

class SA():#Simulated annealing
    """
    模拟退火算法
    """
    def __init__(self,city_list,distance_matrix,city_num,city_matrix,time=100,T0=1000000,Tend=0.01,alpha=0.99):
        self.time = time      #内循环迭代次数
        self.alpha = alpha    #降温系数
        self.T0 = T0          #初始温度
        self.Tend = Tend      #温度终值为
        self.T = T0           #当前温度
        self.city_list = city_list
        self.distance_matrix = distance_matrix
        self.city_num = city_num
        self.city_matrix = city_matrix
        
    def initialization(self):
        """
        #除了随机生成，还可以直接生成100条然后用贪婪法生成初始路径
        默认从城市1出发
        输入所有城市的列表，返回一个随机初始化的城市路径列表
        return list
        """
        self.city_list.append(0)
        temp_list = self.city_list[1:-1]
        np.random.shuffle(temp_list)
        initialization_list = self.city_list[:1] + temp_list + self.city_list[-1:]
        return initialization_list
    
    def get_length(self,path_list):
        """
        input list
        获取该路径的总长度
        输入一个包含途经城市的列表，返回总长度
        return:
            int总长度
        """ 
        length = 0
        for i in range(0,len(path_list)-1):
            j = i+1
            length += self.distance_matrix[path_list[i]][path_list[j]]
        return length
        
    def generate_new(self,time,list):
        """
        input:
            list:list,老路径
            time:int,当前循环次数
        生成新路径，如果循环次数为偶数次用二变换法，奇数次用三变换法
        二变换法：随机选择两个数相交换
        三变换法：随机选择三个数进行随机交换
        也可以选择其他生成方法
        return:
            list新路径
        """
        list_len = len(list)
        if time%2 == 0:
            #二变换法
            temp = random.sample(range(2,list_len - 1), 2)#生成两个互不相同的样本
            temp1,temp2 = temp[0],temp[1]
            t1_index,t2_index = list.index(temp1),list.index(temp2)
            list[t1_index] = temp2
            list[t2_index] = temp1            
        else:
            #三变换法
            temp = random.sample(range(2,list_len - 1),3)#生成三个互不相同的样本
            temp1,temp2,temp3 = temp[0],temp[1],temp[2]
            t1_index = list.index(temp1)
            t2_index = list.index(temp2)
            t3_index = list.index(temp3)
            list[t1_index] = temp3
            list[t2_index] = temp1
            list[t3_index] = temp2 
        return list
    
    def generate_new2(self,parent1_path,parent2_path):
        #尝试使用遗传算法中的顺序交叉法
        path_len = self.city_num+1
        gen_start = random.randint(1,path_len-2)
        gen_end = random.randint(gen_start+1,path_len-1)#起始终止基因
        
        parent2_seg = parent2_path[gen_start:gen_end]#片段
        child = []#子代
        #生成子代
        gene = 0
        while gene < path_len:
            if gene == gen_start:
                for j in range(0,len(parent2_seg)):
                    child.append(parent2_seg[j])
                if parent1_path[gene] not in parent2_seg:
                    child.append(parent1_path[gene])
            elif parent1_path[gene] not in parent2_seg:
                child.append(parent1_path[gene])
            gene += 1
        
        return child
    
    def criterion(self,old_length,new_length,T):
        """
        input:
            old_length:int,前一个状态下的总长度
            new_length:int,生成新状态后的总长度
            T：int,当前温度
        通过metropolis准则判断是否接受当前状态
        return:
            bool
        """
        delta_f = new_length - old_length
        if delta_f <= 0:
            return True
        else:
            probability = math.exp(-delta_f/T)
            if probability > random.random():
                return True
            else:
                return False
         
    def run(self):
        start_time = time.time()
        best_path = self.initialization()
        best_length = self.get_length(best_path)
        current_path = self.generate_new(1, best_path)
        current_length = self.get_length(current_path)
        length_matrix = []
        while True:
            if self.T < self.Tend:
                break
            for i in range(0,self.time):
                if i == 0:
                    new_path = self.generate_new(i,current_path)
                else:
                    new_path =self.generate_new2(current_path,best_path)
                new_length = self.get_length(new_path)
                if self.criterion(current_length,new_length,self.T):
                    current_path = new_path
                    current_length = new_length
                if current_length < best_length:
                    best_length = current_length
                    best_path = current_path
            length_matrix.append(current_length)
                    
            self.T = self.T * self.alpha
        end_time = time.time()
        print("-----模拟退火算法-----")    
        print("最佳路径：")
        print(best_path)
        print("最短距离：",best_length)
        print("SA time:",end_time - start_time)
        print("---------------------")
        self.visible(length_matrix)
        self.picture(best_path)
        self.city_list.pop()
        
    def visible(self,length_matrix):
        plt.plot(np.array(length_matrix))
        plt.ylabel("lengths")
        plt.xlabel("t")
        plt.show()
        
    def picture(self,path):
        X = []
        Y = []
        for index in path:
            X.append(self.city_matrix[index][1])
            Y.append(self.city_matrix[index][2])
        plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(1)
        plt.plot(X, Y, '-o')
        for i in range(len(X)):
            plt.text(X[i] + 0.05, Y[i] + 0.05, str(path[i]), color='red')
        plt.xlabel('横坐标')
        plt.ylabel('纵坐标')
        plt.title('轨迹图')
        plt.show()
        
sa = SA(city_list,dis,city_num,city_matrix)
sa.run()