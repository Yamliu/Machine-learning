import numpy as np
from math import log
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 代表侮辱性词, 0 代表正常言论
    return postingList,classVec

#根据数据集创建词汇向量
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)
#将输入文档转换为向量
def document2Vect(vocabList,inputSet):
    Vect = len(vocabList)*[0]
    for word in inputSet:
        if word in vocabList:
            Vect[vocabList.index(word)]=1
        else:
            print('未找到该词')
    return Vect
#朴素贝叶斯分类器训练函数，返回两个概率向量和一个常数
def trainNB(trainSet,trainCate):
    numTrainSet = len(trainSet)
    numword = len(trainSet[0])
    pc1= sum(trainCate)/float(numTrainSet)
    p0Num = np.zeros(numword)
    p1Num = np.zeros(numword)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainSet):
        if trainCate[i]==1:
            p1Num += trainSet[i]
            p1Denom += sum(trainSet[i])
        else:
            p0Num += trainSet[i]
            p0Denom += sum(trainSet[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p1Vect,p0Vect,pc1

#朴素贝叶斯分类函数
def classifyNB(vect2classify,p1Vect,p0Vect,pc1):
    p1 = np.sum(vect2classify * p1Vect)+ pc1
    p0 = np.sum(vect2classify * p0Vect)+1-pc1
    if p1>p0:
        return 1
    else:
        return 0

d,v=loadDataSet()
vocabList = createVocabList(d)
trainMat = []
for i in d:
    trainMat.append(document2Vect(vocabList,i))
vect1 = document2Vect(vocabList,['my', 'cute', 'I', 'love', 'him'])
vect2 = document2Vect(vocabList,['stupid', 'dog', 'I', 'love', 'him'])
p1,p2,pc1 =trainNB(trainMat,v)
print(p1,p2,pc1)
print(vect1,vect2)
c1 =classifyNB(vect1,p1,p2,pc1)
c2 =classifyNB(vect2,p1,p2,pc1)
print(c1,c2)
