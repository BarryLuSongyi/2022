# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:22:13 2022

@author: 18194
"""

import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

data = pd.read_excel(r"D:\python\Intelligent Optimizaition Algorithm\dataset\Oliver30.xlsx")
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
city_num,city_matrix,distance_matrix = set_distance(data)

class PSO():
    """
    粒子群算法
    """
    def __init__(self,city_num,distance_matrix,city_matrix,n=100,c1=0.4,c2=0.4,w=0.2,iter_num=1000):
        self.city_num = city_num
        self.distance_matrix = distance_matrix
        self.city_matrix = city_matrix
        self.n = n #粒子群规模
        self.c1 = c1#自我认知因子
        self.c2 = c2#社会认知因子
        self.w = w#惯性因子
        self.iter_num = iter_num#迭代次数
    
    class particle():
        """
        定义粒子类
        
        """
        def __init__(self,city_num,distance_matrix):
            self.current_location = []#目前位置
            self.current_length = -1#目前长度
            self.best_location = []#个人最好位置
            self.best_length = -1#个人最好距离
            self.city_num = city_num
            self.distance_matrix = distance_matrix
        
        def initialization(self):
            #生成粒子初始路线
            self.current_location = [i for i in range(0,self.city_num)]
            temp = self.current_location[1:]
            random.shuffle(temp)
            self.current_location[1:] = temp
            self.current_location.append(0)
            self.current_length = self.get_length(self.current_location)
            self.best_length = self.current_length
            self.best_location = self.current_location
            
        def get_length(self,path):
            #输入路径，获得总路线长度
            length = 0
            for i in range(0,len(path)-1):
                length += self.distance_matrix[path[i]][path[i+1]]
            return length
                   
    def update(self,particle,gbest_location):
        """
        对各粒子进行迭代
        利用遗传算法的交叉进行
        本粒子为父代1，按照轮盘赌选择父代2
        """
        pro_list = []#为轮盘赌表分别加入三个概率选择
        pro_list.append(self.w/(self.w+self.c1+self.c2))#自身逆序
        pro_list.append(self.c1/(self.w+self.c1+self.c2))#个人最佳
        pro_list.append(self.c2/(self.w+self.c1+self.c2))#群体最佳
        account_pro = 0
        randval = random.random()
        for i in range(0,len(pro_list)):
            account_pro += pro_list[i]
            if account_pro >= randval:
                break
        parent1 = particle.current_location
        if i == 0:
            parent2 = particle.current_location[::-1]
        elif i == 1:
            parent2 = particle.best_location
        elif i == 2:
            parent2 = gbest_location
        #使用顺序交叉，保证不会出现重复基因，并且不需要做冲突检验
        gen_start = random.randint(1,len(parent1)-2)
        gen_end = random.randint(gen_start+1, len(parent1)-1)
        parent2_seg = parent2[gen_start:gen_end]#得到parent2的插入片段
        son = []
        path_len = self.city_num+1
        gene = 0
        while gene < path_len:
            if gene == gen_start:
                for j in range(0,len(parent2_seg)):
                    son.append(parent2_seg[j])
                if parent1[gene] not in parent2_seg:
                    son.append(parent1[gene])
            elif parent1[gene] not in parent2_seg:
                son.append(parent1[gene])
            gene += 1
         
        return son
        
    def visible(self,length_list):
        plt.plot(np.array(length_list))
        plt.ylabel('length')
        plt.xlabel('times')
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
        
    def run(self):
        start_time = time.time()
        particle_list = [self.particle(city_num,distance_matrix) for i in range(0,self.n)]
        for i in particle_list:
            i.initialization()#初始化n数量的粒子群后给每个粒子随机初始化
        gbest_location = particle_list[0].current_location
        gbest_length = particle_list[0].current_length
        for i in particle_list:#得到初始粒子群最优位置和长度
            if i.current_length < gbest_length:
                gbest_length = i.current_length
                gbest_location = i.current_location
        #开始迭代    
        length_list = []
        for times in range(0,self.iter_num):
            for i in range(0,len(particle_list)):
                particle_list[i].current_location = self.update(particle_list[i],gbest_location)
                particle_list[i].current_length = particle_list[i].get_length(particle_list[i].current_location)
                if particle_list[i].current_length < particle_list[i].best_length:
                    particle_list[i].best_length = particle_list[i].current_length
                    particle_list[i].best_location = particle_list[i].current_location
                if particle_list[i].current_length < gbest_length:
                    gbest_length = particle_list[i].current_length
                    gbest_location = particle_list[i].current_location
                #length_list.append(particle_list[i].best_length)
                length_list.append(gbest_length)
            #print(gbest_length)
            #print(gbest_location)
        print(gbest_location)
        print(gbest_length)
        end_time = time.time()
        print("PSO time:",end_time - start_time)
        self.visible(length_list)
        self.picture(gbest_location)
pso = PSO(city_num,distance_matrix,city_matrix)
pso.run()      



