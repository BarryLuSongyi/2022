# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 22:05:16 2022

@author: 18194
"""

import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
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

class GA():#Genetic Algorithm
    def __init__(self,city_num,distance_matrix,city_matrix,generation=1000,colony=100,survive_rate=0.5,mutation_rate=0.5):
        self.city_num = city_num#城市数量
        self.distance_matrix = distance_matrix#距离矩阵
        self.city_matrix = city_matrix
        self.generation = generation#代数
        self.colony = colony#种群规模   
        self.survivor = int(colony * survive_rate)#每代存活规模
        self.mutation_rate = mutation_rate
        
    class Individual():
        def __init__(self,city_num,distance_matrix):
            self.city_num = city_num#城市数量
            self.distance_matrix = distance_matrix#距离矩阵
            self.path = []#个体路线
            self.length = -1#个体距离
        
        def initialization(self):
            #生成个体初始路线和距离
            self.path = [i for i in range(0,self.city_num)]
            temp = self.path[1:]
            random.shuffle(temp)
            self.path[1:] = temp
            self.path.append(0)
            self.length = self.get_length(self.path)
            
        def get_length(self,path):
            #输入路径，获得总路线长度
            length = 0
            for i in range(0,len(path)-1):
                length += self.distance_matrix[path[i]][path[i+1]]
            return length
        
        def mutation(self):
            """
            变异函数，有一定几率进行变异
            部分逆序变异
            """
            rand1 = random.randint(1,self.city_num)
            rand2 = random.randint(rand1,self.city_num)
            temp = self.path[rand1:rand2][::-1]
            new_path = self.path[:rand1] + temp + self.path[rand2:]
            if self.get_length(new_path) < self.length:#只有小于原距离才变异
                self.path = new_path
                self.length = self.get_length(self.path)
    
    def cross(self,parent1_path,parent2_path):
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
    
    def select(self,individual_list):
        """
        选择函数，利用轮盘赌算法
        """
        length_list = []
        total_length = 0
        for i in individual_list:
            length_list.append(i.length)
        max_length = max(length_list)
        for i in range(0,self.colony):
            total_length += max_length - length_list[i]
        pro_list = [((max_length - length_list[i])/total_length) for i in range(0,self.colony)]#各父代概率表
        accounting_pro = 0
        rand_val = random.random()
        for parent in range(0,self.colony):
            accounting_pro += pro_list[parent]
            if rand_val < accounting_pro:
                break
        return parent
    
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
        individual_list = [self.Individual(self.city_num,self.distance_matrix) for i in range(0,self.colony)]
        for i in individual_list:#初始化各个体
            i.initialization()
        c_best_path = individual_list[0].path#种群最佳路径
        c_best_length = individual_list[0].length#种群最佳距离
        for i in individual_list:#初始化第一代最佳
            if i.length < c_best_length:
                c_best_length = i.length
                c_best_path = i.path
                
        gen = 0
        length_list = []
        while gen < self.generation:
            #所有的个体都与历史最佳杂交？
            g_best_path = individual_list[0].path#本代最佳路径
            g_best_length = individual_list[0].length#本代最佳距离
            
            next_gen_parent = []
            while len(next_gen_parent) < self.survivor:#自然选择
                selected_parent = self.select(individual_list)
                if selected_parent not in next_gen_parent:
                    next_gen_parent.append(selected_parent)
                    
            for i in range(0,self.survivor):#与每代最优杂交
                #child_path = self.cross(individual_list[next_gen_parent[i]].path, g_best_path)
                child_path = self.cross(individual_list[next_gen_parent[i]].path,c_best_path)
                individual_list[next_gen_parent[i]].path = child_path
                individual_list[next_gen_parent[i]].length = individual_list[next_gen_parent[i]].get_length(child_path)
                
            for j in range(0,self.colony):
                if j not in next_gen_parent:
                    individual_list[j].initialization()#对于没被选中的，直接重组
                    
            for individual in range(0,self.colony):#突变
                rand_val = random.random()
                if rand_val < self.mutation_rate:
                    individual_list[individual].mutation()
            
            for individual in individual_list:#记录
                if individual.length < g_best_length:
                    g_best_length = individual.length
                    g_best_path = individual.path
                if individual.length < c_best_length:
                    c_best_length = individual.length
                    c_best_path = individual.path
                              
            length_list.append(c_best_length)
            gen += 1
            
        print(c_best_path)
        print(c_best_length)
        
        end_time = time.time()
        print("GA time:",end_time - start_time)
        self.visible(length_list)
        self.picture(c_best_path)

ga = GA(city_num,distance_matrix,city_matrix)
ga.run()