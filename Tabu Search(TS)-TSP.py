# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:25:17 2023

@author: 18194
"""

import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

data = pd.read_excel(r"D:\python\TSPdataset\eil76.xlsx")
def set_distance(df):
    cityNum = len(list(df.iloc[:,1]))
    cityMatrix = []
    for i in range(0,cityNum):
        cityMatrix.append(list(df.iloc[i,0:]))
    distanceMatrix = []
    for i in range(0,cityNum):
        rowDistance = []#i城市到各城市的距离
        for j in range(0,cityNum):
            #欧几里得距离
            x = abs(cityMatrix[i][1] - cityMatrix[j][1])
            y = abs(cityMatrix[i][2] - cityMatrix[j][2])
            distance = math.sqrt(x**2+y**2)
            rowDistance.append(distance)
        distanceMatrix.append(rowDistance)
    return cityNum,cityMatrix,distanceMatrix
cityNum,cityMatrix,distanceMatrix = set_distance(data)

class TS():
    def __init__(self,tabuSize=10,candidateNum=40,generation=5000):
        self.path = []
        self.fitness = -1
        self.bestPath = []
        self.bestFitness = -1
        self.tabuSize = tabuSize
        self.candidateNum = candidateNum
        self.generation = generation
        
    
    def initialization1(self):
        #对于更乱的图起到了更好的效果
        self.path = [i for i in range(0,cityNum)]
        temp = self.path[1:]
        random.shuffle(temp)
        self.path[1:] = temp
        self.path.append(0)
        self.length = self.get_fitness(self.path)
    
    def initialization(self):
        #最近邻法
        farestCity = distanceMatrix[0].index(max(distanceMatrix[0]))
        path = [farestCity]
        while len(path) < cityNum-1:
            distanceList = distanceMatrix[path[-1]]
            nearestCity = path[-1]
            minDistance = float('inf')
            for distance in distanceList[1:]:
                if (distance < minDistance) and (distanceList.index(distance) not in path):
                    minDistance = distance
                    nearestCity = distanceList.index(distance)
            path.append(nearestCity)
        path.insert(0, 0)
        path.append(0)
        self.path = path

    def get_fitness(self,path):
        fitness = 0
        for i in range(0,len(path)-1):
            fitness += distanceMatrix[path[i]][path[i+1]]
        return fitness
    
    def two_swap(self,path):
        temp = copy.deepcopy(path)
        randVal1 = random.randint(1,cityNum-2)
        randVal2 = random.randint(randVal1+1,cityNum-1)
        tempVal = temp[randVal1]
        temp[randVal1] = temp[randVal2]
        temp[randVal2] = tempVal
        
        newPath,newLength = temp,self.get_fitness(temp)
        return newPath,newLength,randVal1,randVal2
    
    def two_opt(self,path):
        temp = copy.deepcopy(path)
        randVal1 = random.randint(1,cityNum-2)
        randVal2 = random.randint(randVal1+1,cityNum-1)
        seg = path[randVal1:randVal2][::-1]
        temp = temp[:randVal1] + seg + temp[randVal2:]
        
        newPath,newLength = temp,self.get_fitness(temp)
        return newPath,newLength,randVal1,randVal2
           
    def visible(self,length_list):
        plt.plot(np.array(length_list))
        plt.ylabel('length')
        plt.xlabel('times')
        plt.show() 
    
    def picture(self,path):
        X = []
        Y = []
        for index in path:
            X.append(cityMatrix[index][1])
            Y.append(cityMatrix[index][2])
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
        startTime = time.time()
        
        #初始化
        self.initialization1()
        self.fitness = self.get_fitness(self.path)
        self.bestPath = self.path
        self.bestFitness = self.fitness
        
        
        tabuList = []#禁忌表，记录由哪个领域算子及动作所生成的
        fitnessList = []
        while (len(fitnessList) < self.generation) or (fitnessList[-1] != fitnessList[-100]):
            #起码迭代100次，然后五次没更新就可以停止
            candidate = []
            operator = []
            candidatePath = []
            candidateFitness = float('inf')
            
            while len(candidate) < self.candidateNum:
                #newPath,newFitness,randVal1,randVal2 = self.two_swap(self.path)
                newPath,newFitness,randVal1,randVal2 = self.two_opt(self.path)
                
                if (newPath,newFitness) in candidate:
                    continue
                if (randVal1,randVal2) in tabuList:
                    if newFitness < self.bestFitness:#特赦
                        candidate.append((newPath,newFitness))
                        operator.append((randVal1,randVal2))
                        continue
                    else:
                        continue
                candidate.append((newPath,newFitness))
                operator.append((randVal1,randVal2))
            
            tabuIndex = -1
            for (path,fitness) in candidate:
                if fitness < candidateFitness:
                    candidatePath = path
                    candidateFitness = fitness
                    tabuIndex = candidate.index((path,fitness))
                    
            tabuList.append(operator[tabuIndex])
            if len(tabuList) > self.tabuSize:
                tabuList.pop(0)
            
            self.path = candidatePath
            self.fitness = candidateFitness
            if candidateFitness < self.bestFitness:
                self.bestPath = candidatePath
                self.bestFitness = candidateFitness
                
            fitnessList.append(self.bestFitness)  
        
        print(self.bestPath)
        print(self.bestFitness)
        self.visible(fitnessList)
        self.picture(self.bestPath)
            
        
        endTime = time.time()
        print("TS time:",endTime - startTime)
        
    
if __name__ == "__main__":
    print("ts running...")
    ts = TS()
    ts.run()
    