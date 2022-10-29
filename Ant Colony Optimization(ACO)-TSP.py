# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:30:55 2022

@author: 18194
"""

import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

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

class ACO():
    def __init__(self,city_num,distance_matrix,city_matrix,alpha=2,beta=2,Q=10,RHO=0.1,ant_num=100,generation=100,elite=10):
        self.city_num = city_num
        self.distance_matrix = distance_matrix
        self.city_matrix = city_matrix
        self.alpha = alpha#信息启发式因子
        self.beta = beta#期望启发式因子
        self.Q = Q#更新因子
        self.RHO = RHO#挥发因子
        self.ant_num = ant_num#蚂蚁数量
        self.generation = generation#迭代代数
        self.elite = elite#精英蚂蚁的数量，elite<ant_num
        
    class Ant():
        def __init__(self,city_num,distance_matrix):
            self.path = []
            self.length = -1
            self.city_num = city_num
            self.distance_matrix = distance_matrix
            
        
        def initialization(self):
            self.path = []
            self.path.append(0)
            self.length = 0
        
        def get_length(self,path):
            #输入路径，获得总路线长度
            length = 0
            for i in range(0,len(path)-1):
                length += self.distance_matrix[path[i]][path[i+1]]
            return length
            
        def select_city(self,pheromon_matrix,beta_matrix,alpha,beta):
            """
            输入蚂蚁一只
            返回下个城市一个
            """
            #蚂蚁选择下一个城市
            next_city = -1
            if len(self.path) == self.city_num:
                next_city = 0
            next_city_list = []#该蚂蚁可去的城市列表
            for i in range(0,self.city_num):
                if i not in self.path:
                    next_city_list.append(i)
            alpha_list = []#存放去到各城市的alpha
            beta_list = []#存放去到各城市的beta
            pro_list = []#总概率，用轮盘赌得出下个城市
            total_alpha_beta = 0#分母
            current_city = self.path[-1]
            len_city_list = len(next_city_list)
            
            for i in range(0,len_city_list):
                path_alpha,path_beta = (self.get_alpha_beta(pheromon_matrix,beta_matrix,current_city,next_city_list[i]))
                alpha_list.append(path_alpha**alpha)
                beta_list.append(path_beta**beta)
            for j in range(0,len_city_list):
                total_alpha_beta += alpha_list[j]*beta_list[j]
            for k in range(0,len_city_list):
                pro_list.append((alpha_list[k]*beta_list[k])/total_alpha_beta)
                
            #轮盘赌
            rand_val = random.random()
            accounting_pro = 0
            for index in range(0,len_city_list):
                accounting_pro += pro_list[index]
                if rand_val < accounting_pro:
                    next_city = next_city_list[index]
                    break
            self.path.append(next_city)
            
            
        def get_alpha_beta(self,pheromon_matrix,beta_matrix,start_city,end_city):
            #输入去留城市，得到其信息素量
            alpha = 0
            beta = 0
            if start_city == end_city:
                print("wrong cities with same index")
            elif start_city > end_city:
                alpha = pheromon_matrix[start_city][end_city]
                beta = beta_matrix[start_city][end_city]
            else:
                alpha = pheromon_matrix[end_city][start_city]
                beta = beta_matrix[end_city][start_city]
            return alpha,beta
    
    def get_top_ant(self,length_list):
        """
        输入距离列表，获得前elite小距离的蚂蚁的索引
        """
        temp_list = copy.deepcopy(length_list)
        temp_list.sort()
        min_n_ant_index = [length_list.index(one) for one in temp_list[::][:self.elite]]
        return min_n_ant_index
    
    def update(self,pheromon_matrix,beta_matrix,elite_ants_path):
        """
        输入旧信息表，得到新信息表
        利用前elite的路径来进行更新
        """
        for ant in range(0,self.elite):
            for i in range(0,self.city_num-1):
                j = i+1
                row = elite_ants_path[ant][i]
                column = elite_ants_path[ant][j]
                if column == row:
                    print("wrong cities with same index")
                elif row > column:
                    pheromon_matrix[row][column] += beta_matrix[row][column]
                else:
                    pheromon_matrix[column][row] += beta_matrix[column][row]
        return pheromon_matrix
    
    def fade(self,pheromon_matrix):
        for i in range(0,self.city_num):
            for j in range(0,i):
                pheromon_matrix[i][j] = (1-self.RHO)*pheromon_matrix[i][j]
        return pheromon_matrix
    
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
        #带精英策略的蚁周系统蚁群算法
        #重点解决信息素的不对称问题、选择问题、升级问题
        #pheromon_matrix中，行的索引总是比列的大（前大后小）
        print("running ACO...")
        start_time = time.time()
        pheromon_matrix= [[0 for column in range(0,self.city_num)] for row in range(0,self.city_num)]#需要做到信息素的更新全对称
        beta_matrix = [[0 for col in range(0,self.city_num)] for row in range(0,self.city_num)]
        for i in range(0,self.city_num):
            for j in range(0,i):
                pheromon_matrix[i][j] = 1
                beta_matrix[i][j] = self.Q/self.distance_matrix[i][j]
        
        g_best_length_list = []#初始化历史最佳
        h_best_length_list = []
        h_best_path = [i for i in range(0,self.city_num)]
        h_best_length = 0
        for i in range(0,len(h_best_path)-1):
            h_best_length += self.distance_matrix[h_best_path[i]][h_best_path[i+1]]
            
        
        gen = 0
        while gen < self.generation:
            ant_list = [self.Ant(self.city_num,self.distance_matrix) for i in range(0,self.ant_num)]
            for ant in ant_list:ant.initialization()#赋起点为0
            
            for city in range(0,self.city_num):#各蚂蚁均走完一圈
                for ant in range(0,self.ant_num):
                    ant_list[ant].select_city(pheromon_matrix, beta_matrix, self.alpha, self.beta)
            
            length_list = []#储藏蚂蚁们的路径长度
            for ant in range(0,self.ant_num):
                ant_list[ant].length = ant_list[ant].get_length(ant_list[ant].path)
                length_list.append(ant_list[ant].length)
            elite_ants = self.get_top_ant(length_list)#获得精英蚂蚁列表
            elite_ants_path = []
            for ant in range(0,self.elite):
                elite_ants_path.append(ant_list[elite_ants[ant]].path)
                
            #此处update，使用带精英策略的蚁周系统蚁群算法
            pheromon_matrix = self.fade(pheromon_matrix)#先fade一下
            pheromon_matrix = self.update(pheromon_matrix,beta_matrix,elite_ants_path)
            
            g_best_length = ant_list[elite_ants[0]].length#更新代最佳和历史最佳
            g_best_path = ant_list[elite_ants[0]].path
            g_best_length_list.append(g_best_length)
            if g_best_length < h_best_length:
                h_best_length = g_best_length
                h_best_path = g_best_path
            h_best_length_list.append(h_best_length)
            gen += 1
        end_time = time.time()
        print(h_best_path)
        print(h_best_length)
        print("ACO time:",end_time - start_time)
        self.visible(g_best_length_list)
        self.picture(h_best_path)

aco = ACO(city_num,distance_matrix,city_matrix)
aco.run()
        