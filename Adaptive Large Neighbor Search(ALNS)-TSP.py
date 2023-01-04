# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:30:53 2022

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

class ALNS():
    def __init__(self,w1=0.4,w2=0.3,w3=0.2,w4=0.1,miu=0.8,T0=100000000,Tend=0.01,alpha=0.99):
        self.w1 = w1#目前全局最优解
        self.w2 = w2#比前一个好
        self.w3 = w3#被接受
        self.w4 = w4#被拒绝
        self.miu = miu
        self.alpha = alpha    #降温系数
        self.T0 = T0          #初始温度
        self.Tend = Tend      #温度终值
        self.path = []
        self.length = -1
        
    def initialization(self):
        self.path = [i for i in range(0,cityNum)]
        temp = self.path[1:]
        random.shuffle(temp)
        self.path[1:] = temp
        self.path.append(0)
        self.length = self.get_length(self.path)
    
    def get_length(self,path):
        length = 0
        for i in range(0,len(path)-1):
            length += distanceMatrix[path[i]][path[i+1]]
        return length
    
    def DRandom(self,path,destroyNum):
        if destroyNum >= cityNum or destroyNum < 0 or type(destroyNum) != int:
            print("out of index!")
            return path,[]
        #print("this is random destroy")
        randVal = random.sample(range(1,cityNum),destroyNum)
        popedValue = []
        for i in range(0,destroyNum):
            popedValue.append(randVal[i])
            path.remove(randVal[i])
        return path,popedValue
        
    def DGreedy(self,path,destroyNum):
        if destroyNum >= cityNum or destroyNum < 0 or type(destroyNum) != int:
            print("out of index!")
            return path,[]
        #print("this is greedy destroy")
        valueList = [0]
        lastValue = distanceMatrix[path[0]][path[1]]
        for i in range(1,cityNum):
            nextValue = distanceMatrix[path[i]][path[i+1]]
            totalValue = lastValue + nextValue
            valueList.append(totalValue)
            lastValue = nextValue
        
        tempList = copy.deepcopy(valueList)
        tempList.sort()
        
        popedValue = []
        for key,val in enumerate(valueList):
            if len(popedValue) >= destroyNum:
                break
            else:
                if val in tempList[::-1][:destroyNum]:
                    popedValue.append(path[key])
            
        for i in range(0,destroyNum):
            path.remove(popedValue[i])
        
        return path,popedValue
    
    def RRandom(self,path,popedValue):
        for city in popedValue:
            location = random.randint(1,len(path)-1)
            path.insert(location,city)
        return path
    
    def RGreedy(self,path,popedValue):
        #print("this is greedy repair")
        for city in popedValue:
            temp = copy.deepcopy(path)
            location = random.randint(1,len(path)-1)
            temp.insert(location,city)
            newLength = self.get_length(temp)
            
            temp.remove(city)
            for i in range(1,len(path)):
                temp.insert(i,city)
                if self.get_length(temp) < newLength:
                    location = i
                    newLength = self.get_length(temp)
                temp.remove(city)
            path.insert(location,city)
        return path       
                
    def RRegret(self):
        print("this is regret repair")
        
    
    def metropolis(self,path,T,bestLength):
        newLength = self.get_length(path)
        deltaF = newLength - self.length
        if deltaF <= 0:
            if newLength < bestLength:
                return path,newLength,self.w1
            else:
                return path,newLength,self.w2
        else:
            probability = math.exp(-deltaF/T)
            if probability > random.random():
                return path,newLength,self.w3
            else:
                return self.path,self.length,self.w4
    
    def select(self,weightList):
        totalPro = sum(weightList)
        proList = []
        for i in range(0,len(weightList)):
            proList.append(weightList[i]/totalPro)
        randVal = random.random()
        accountingPro = 0
        for j in range(0,len(weightList)):
            accountingPro += proList[j]
            if randVal < accountingPro:
                break
        return j
        
    def renew(self,destroy,destroyWeight,repair,repairWeight,weight):
        for i in range(0,len(destroyWeight)):
            if i == destroy:
                destroyWeight[i] = self.miu * destroyWeight[i] + (1-self.miu) * weight
            else:
                destroyWeight[i] = self.miu * destroyWeight[i]
        for j in range(0,len(repairWeight)):
            if j == repair:
                repairWeight[j] = self.miu * repairWeight[j] + (1-self.miu) * weight
            else:
                repairWeight[j] = self.miu * repairWeight[j]
                
        return destroyWeight,repairWeight

    def visible(self,length_list):
        plt.plot(np.array(length_list))
        plt.ylabel('length')
        plt.xlabel('times')
        plt.show() 

    def run(self):
        print("ALNS running...")
        
        self.initialization()
        T = self.T0
        bestPath = self.path
        bestLength = self.length
        bestLengthList = [bestLength]
        
        destroyWeight = [1,1]
        repairWeight = [1,1]
        
        while T > self.Tend:
            #print(self.path)
            
            destroy = self.select(destroyWeight)
            repair = self.select(repairWeight)
            #drNum = random.randint(1,10)
            drNum = 5
            
            if destroy == 0:
                destroyedPath,popedValue = self.DRandom(self.path,drNum)
            elif destroy == 1:
                destroyedPath,popedValue = self.DGreedy(self.path,drNum)
                
            #print(popedValue)
            if repair == 0:    
                newPath = self.RRandom(destroyedPath,popedValue)
            elif repair == 1:
                newPath = self.RGreedy(destroyedPath,popedValue)
                
            #print(newPath)
            self.path,self.length,weight = self.metropolis(newPath,T,bestLength)
            if weight == self.w1:
                bestLengthList.append(self.length)
                bestPath = self.path
                bestLength = self.length
                
            destroyWeight,repairWeight = self.renew(destroy,destroyWeight,repair,repairWeight,weight)
            #bestLengthList.append(self.length)
            #print(self.path)
            
            T = T*self.alpha
        print(bestPath)
        print(bestLength)
        self.visible(bestLengthList)
        
if __name__ == "__main__":
    startTime = time.time()
    alns = ALNS()
    alns.run()
    endTime = time.time()
    print("ALNS time:",endTime-startTime)