# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:56:17 2022

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

class Country():
    def __init__(self,level=False):
        self.level = level#地位，帝国为True，殖民地为False
        self.path = []#路径
        self.length = -1#使用路径长度作为适应度
        self.master = -1#表示所属帝国下标，若为-1则自身为帝国
        self.colony = []#表示所拥有的的殖民地下标，若为空数组则自身为殖民地
        self.colonyNum = -1#表示所拥有的的殖民地数量，若为-1则自身为殖民地
    
    def get_length(self,path):
        """
        This function calculats the length of a input route
        input:
            path:list
        return:
            length:int
        """
        length = 0
        for i in range(0,len(path)-1):
            length += distanceMatrix[path[i]][path[i+1]]
        return length

    def initialization(self):
        """
        This function initialize each country's initial route and length, and update them to their items.
        input:
            None
        return:
            None
        """
        self.path = [i for i in range(0,cityNum)]
        temp = self.path[1:]
        random.shuffle(temp)
        self.path[1:] = temp
        self.path.append(0)
        self.length = self.get_length(self.path)
        
    def revolution(self):
        rand1 = random.randint(1,cityNum)
        rand2 = random.randint(rand1,cityNum)
        temp = self.path[rand1:rand2][::-1]
        newPath = self.path[:rand1] + temp + self.path[rand2:]
        if self.get_length(newPath) < self.length:#只有小于原距离才变异
            self.path = newPath
            self.length = self.get_length(self.path)
    
class ICA():
    def __init__(self,numImperialist=100,numColony=1000,revolutionRate=0.7,numIter=1000,colEffect=0.2):
        self.numImperialist = numImperialist
        self.numColony = numColony
        self.numCountry = numImperialist + numColony
        self.revolutionRate = revolutionRate
        self.numIter = numIter
        self.colEffect = colEffect
        
    
    def set_imperialist(self,countryList):
        """
        This function is for initialize imperialist
        input:
            countryList:list
        return:
            costList:list
        """
        costList = [countryList[i].length for i in range(0,self.numCountry)]
        maxCost = max(costList)
        for i in range(0,self.numCountry):
            costList[i] = maxCost - costList[i]
        tempList = copy.deepcopy(costList)
        tempList.sort()
        maxCountryIndex = [costList.index(one) for one in tempList[::-1][:self.numImperialist]]
        for imp in maxCountryIndex:
            countryList[imp].level = True
        
        return costList,maxCountryIndex
        
    def set_colony(self,countryList,costList,maxCountryIndex):
        """
        This function initialize colony
        input:
            countryList:list
            costList:list
            maxCountryIndex:list
        return:
            None
        """
        totalPro = 0
        for imp in range(0,self.numImperialist):
            totalPro += costList[maxCountryIndex[imp]]
        for imp in range(0,self.numImperialist):
            countryList[maxCountryIndex[imp]].colonyNum = round((costList[maxCountryIndex[imp]]/totalPro)*self.numColony)
        
        countryIndex = 0
        for imp in range(0,self.numImperialist):
            counter = 0
            while counter <= countryList[maxCountryIndex[imp]].colonyNum:
                if countryIndex == self.numCountry:
                    break
                counter += 1
                if countryList[countryIndex].level == False:
                    countryList[countryIndex].master = maxCountryIndex[imp]
                    countryIndex += 1
                else:
                    countryIndex += 1
            
    def assimilation(self,colonyPath,masterPath):#有的殖民地迭代两次后不迭代了
        """
        This function use GA technique OX to assimilate colony to its master
        input:
            colonyPath:list
            masterPath:list
        return:
            child:list
        """
        pathLen = cityNum+1
        genStart = random.randint(1,pathLen-2)
        genEnd = random.randint(genStart+1,pathLen-1)#起始终止基因
        
        masterSeg = masterPath[genStart:genEnd]#片段
        child = []#子代
        #生成子代
        gene = 0
        while gene < pathLen:
            if gene == genStart:
                for j in range(0,len(masterSeg)):
                    child.append(masterSeg[j])
                if colonyPath[gene] not in masterSeg:
                    child.append(colonyPath[gene])
            elif colonyPath[gene] not in masterSeg:
                child.append(colonyPath[gene])
            gene += 1
        return child
    
    def rebellion(self,impIndex,colIndex,countryList):
        """
        This function switch the situation of master and colony if the colony better the imperialist.
        """
        impColonyList = countryList[impIndex].colony
        impColonyNum = countryList[impIndex].colonyNum
        
        countryList[impIndex].level = False
        countryList[colIndex].level = True
        
        countryList[colIndex].colonyNum = impColonyNum
        countryList[impIndex].colonyNum = -1
        
        countryList[colIndex].master = -1
        countryList[impIndex].master = colIndex
        
        if impColonyList != [] :    
            impColonyList.remove(colIndex)
        impColonyList.append(impIndex)
        countryList[colIndex].colony = impColonyList
        countryList[impIndex].colony = []
        
    def renew_cost(self,countryList):
        costList = [countryList[i].length for i in range(0,self.numCountry)]
        maxCost = max(costList)
        for i in range(0,self.numCountry):
            costList[i] = maxCost - costList[i]
        return costList
    
    def find_weakest_colony(self,weakestImp,countryList,costList):
        colCostList = []
        if countryList[weakestImp].colony != []:
            for col in countryList[weakestImp].colony:
                colCostList.append(costList[col])
            minCostColony = countryList[weakestImp].colony[colCostList.index(min(colCostList))]
        else:
            minCostColony = weakestImp
        return minCostColony
    
    def competition(self,weakestImp,weakestCol,strongestImp,countryList):
        countryList[weakestCol].master = strongestImp
        countryList[weakestImp].colony.remove(weakestCol)
        countryList[strongestImp].colony.append(weakestCol)
        countryList[weakestImp].colonyNum -= 1
        countryList[strongestImp].colonyNum += 1
    
    def fade(self,weakestImp,strongestImp,countryList,imperialistList):
        countryList[weakestImp].master = strongestImp
        countryList[strongestImp].colony.append(weakestImp)
        countryList[weakestImp].colonyNum = -1
        countryList[strongestImp].colonyNum += 1
        imperialistList.remove(weakestImp)
        
        
    def visible(self,length_list):
        plt.plot(np.array(length_list))
        plt.ylabel('length')
        plt.xlabel('times')
        plt.show()
    
    def run(self):
        countryList = [Country() for country in range(0,self.numCountry)]
        for country in countryList:
            country.initialization()
            
        costList,imperialistList = self.set_imperialist(countryList)#计算势力最大的numImperialist个国家的下标
        self.set_colony(countryList,costList,imperialistList)#为殖民地赋初值
        for country in countryList:#帝国所有殖民地赋初值
            if country.master != -1:
                countryList[country.master].colony.append(countryList.index(country))
                
        bestLength = countryList[0].length
        bestLengthList = [countryList[0].length]
        bestPath = countryList[0].path                
        #print(countryList[0].__dict__.items())
        year = 0
        while year < self.numIter:
            for country in countryList:#assimilation同化
                if country.level == False:
                    country.path = self.assimilation(country.path,countryList[country.master].path)
                    
            for imp in imperialistList:#如果殖民地势力比帝国大则反叛
                if countryList[imp].colony == []:continue
                colonyLengthList =[]
                for col in countryList[imp].colony:
                    colonyLengthList.append(countryList[col].length)
                minColonyLength = min(colonyLengthList)
                if minColonyLength < countryList[imp].length:
                    #print("empire ",imp," rebells")
                    minColonyIndex = countryList[imp].colony[colonyLengthList.index(minColonyLength)]
                    self.rebellion(imp,minColonyIndex,countryList)
                    imperialistList[imperialistList.index(imp)] = minColonyIndex
            
            for i in countryList:#revolution革命
                randVal = random.random()
                if randVal < self.revolutionRate:
                    i.revolution()
                    if i.level == False and i.length<countryList[i.master].length:
                        self.rebellion(countryList.index(i),i.master,countryList)
            
            
            imperialistPowerList = []#帝国竞争
            for imp in imperialistList:#计算帝国总势力
                imperialistPower = costList[imp]
                if countryList[imp].colonyNum == 0:
                    imperialistPowerList.append(imperialistPower)
                    continue
                else:
                    colonyPower = 0
                    for col in countryList[imp].colony:
                        colonyPower += costList[col]
                imperialistPower += self.colEffect * colonyPower / countryList[imp].colonyNum 
                imperialistPowerList.append(imperialistPower)
            weakestImp = imperialistList[imperialistPowerList.index(min(imperialistPowerList))]
            strongestImp = imperialistList[imperialistPowerList.index(max(imperialistPowerList))]
            weakestCol = self.find_weakest_colony(weakestImp,countryList,costList)
            if weakestCol not in imperialistList:
                self.competition(weakestImp,weakestCol,strongestImp,countryList)
            else:
                self.fade(weakestCol,strongestImp,countryList,imperialistList)
            
            
            #print(imperialistPowerList)
            costList = self.renew_cost(countryList)
            #print(countryList[imperialistList[0]].length)
            for i in countryList:
                if i.length < bestLength:
                    bestLength = i.length
                    bestPath = i.path
            bestLengthList.append(bestLength)
            year += 1
        self.visible(bestLengthList)
        print(bestPath)
        print(bestLength)

if __name__ == "__main__":
    print("ICA runing...")
    startTime = time.time()
    ica = ICA()
    ica.run()
    endTime = time.time()
    print("ICA time:",endTime - startTime)
    #main process
    