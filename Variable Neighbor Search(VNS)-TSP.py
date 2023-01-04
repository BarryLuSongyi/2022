# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 12:46:22 2022

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



class VNS():
    def __init__(self,numIter=10000,neighborK=4,localL=500):
        self.numIter = numIter
        self.neighborK = neighborK
        self.localL = localL
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
    
    def two_swap(self,path):
        temp = copy.deepcopy(path)
        randVal1 = random.randint(1,cityNum-2)
        randVal2 = random.randint(randVal1+1,cityNum-1)
        tempVal = temp[randVal1]
        temp[randVal1] = temp[randVal2]
        temp[randVal2] = tempVal
        
        newPath,newLength = temp,self.get_length(temp)
        return newPath,newLength
           
    def three_swap(self,path):
        temp = copy.deepcopy(path)
        randVal = random.sample(range(1,cityNum),3)#生成三个互不相同的样本
        randVal1,randVal2,randVal3 = randVal[0],randVal[1],randVal[2]
        tempVal1,tempVal2 = temp[randVal1],temp[randVal2]
        temp[randVal1] = temp[randVal3]
        temp[randVal2] = tempVal1
        temp[randVal3] = tempVal2
        
        newPath,newLength = temp,self.get_length(temp)
        return newPath,newLength
        
    def two_opt(self,path):
        temp = copy.deepcopy(path)
        randVal1 = random.randint(1,cityNum-2)
        randVal2 = random.randint(randVal1+1,cityNum-1)
        seg = path[randVal1:randVal2][::-1]
        temp = temp[:randVal1] + seg + temp[randVal2:]
        
        newPath,newLength = temp,self.get_length(temp)
        return newPath,newLength
       
    def three_opt(self,path):
        temp = copy.deepcopy(path)
        randVal = random.sample(range(1,cityNum),3)#生成三个互不相同的样本
        randVal.sort()
        #print(randVal)
        randVal1,randVal2,randVal3 = randVal[0],randVal[1],randVal[2]
        
        #去掉三条边得到三个片段
        seg1,seg2,seg3,seg4 = temp[:randVal1],temp[randVal1:randVal2],temp[randVal2:randVal3],temp[randVal3:]
        #print(seg1,seg2,seg3,seg4)
        
        #去掉三条边后可能的顺序路线
        path1 = seg1 + seg3 + seg2 + seg4
        path2 = seg1 + seg3[::-1] + seg2[::-1] + seg4
        path3 = seg1 + seg2[::-1] + seg3 + seg4
        path4 = seg1 + seg2 + seg3[::-1] + seg4
        path5 = seg1 + seg2[::-1] + seg3[::-1] + seg4
        path6 = seg1 + seg3[::-1] + seg2 + seg4
        path7 = seg1 + seg3 + seg2[::-1] + seg4
        
        pathList = [path1,path2,path3,path4,path5,path6,path7]
        bestPath,bestLength = path1,self.get_length(path1)
        for path in pathList:
            pathLength = self.get_length(path)
            if pathLength < bestLength:
                bestPath = path
                bestLength = pathLength
        
        return bestPath,bestLength
        
    def VND(self,neighborPath,neighborLength,iterK,localSearchL):
        localPathList = []
        localLengthList = []
        for i in range(0,10):
            #randVal = random.randint(0,self.neighborK-1)
            randVal = 3
            if randVal == 0:
                localPath,localLength = self.three_opt(neighborPath)
            elif randVal == 1:
                localPath,localLength = self.two_opt(neighborPath)
            elif randVal == 2:
                localPath,localLength = self.three_swap(neighborPath)
            else:
                localPath,localLength = self.two_swap(neighborPath)
                
            localPathList.append(localPath)
            localLengthList.append(localLength)
        
        minLocalLength = min(localLengthList)
        minLocalPath = localPathList[localLengthList.index(minLocalLength)]
        if localSearchL == self.localL and minLocalLength < neighborLength:
            self.path = neighborPath
            self.length = neighborLength
            iterK = 0
        elif minLocalLength < neighborLength:
            self.VND(minLocalPath,minLocalLength,iterK,localSearchL+1)
        elif minLocalLength > neighborLength and localSearchL > 0:
            self.path = neighborPath
            self.length = neighborLength
            iterK = 0
        elif minLocalLength > neighborLength and localSearchL == 0:
            iterK += 1
        
    def visible(self,length_list):
        plt.plot(np.array(length_list))
        plt.ylabel('length')
        plt.xlabel('times')
        plt.show() 
    
   
    def run(self):
        self.initialization()
        #print(self.__dict__.items())
         
        iterNum = 0
        bestLengthList = [self.length]
        while iterNum < self.numIter:
            #print("第",iterNum,"代")
            iterK = 0
            while iterK < self.neighborK:#iterK表示目前到了哪个邻域
                #print("第",iterK,"邻域")
                if iterK == 0:
                    neighborPath,neighborLength = self.three_opt(self.path)
                elif iterK == 1:
                    neighborPath,neighborLength = self.two_opt(self.path)
                elif iterK == 2:
                    neighborPath,neighborLength = self.three_swap(self.path)
                else:
                    neighborPath,neighborLength = self.two_swap(self.path)
                    
                if neighborLength < self.length:
                    self.VND(neighborPath,neighborLength,iterK,0)
                else:
                    iterK += 1
                
            bestLengthList.append(self.length)
            iterNum += 1
        print(self.path)
        print(self.length)
        self.visible(bestLengthList)
        
    
    
if __name__ == '__main__':
    startTime = time.time()
    vns = VNS()
    vns.run()
    endTime = time.time()
    print("TS time:",endTime-startTime)