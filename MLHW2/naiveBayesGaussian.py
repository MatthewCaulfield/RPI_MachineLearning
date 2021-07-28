import csv
import math
import random
import operator
import numpy as np
import statistics
from collections import Counter
import pandas as pd


def naiveBayesGaussian(filename, num_splits, train_percent):
    statDic = {}

    for j in train_percent:
        statDic[j] = []

    data = loadData(filename)

    for column in range(len(data[0])):
        column2Float(data, column)
    # print(data[0])

    for i in range(num_splits):
        trainTest = splitData(data, 0.8)
        train = trainTest[0]
        test = trainTest[1]
        # print(train[0])
        # print(test)
        testClasses = [i[len(i) - 1] for i in test]
        for percentTrain in train_percent:
            #print(percentTrain)
            splitTrain = splitTrainData(train, percentTrain)
            # print(splitTrain)
            predicted = naiveBayes(splitTrain, test)
            acc = accuracy(testClasses, predicted)
            #print("current accuracy:", acc))
            statDic[percentTrain].append(acc)
    df = pd.DataFrame(statDic)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv("C:/Users/caulf/Desktop/MLHW2/output.csv")
   # print(statDic)

def seperateByClass(dataset):
    classDict = {}
    for dataPoint in dataset:
        currClass = dataPoint[-1]
        if currClass not in classDict:
            classDict[currClass] = [dataPoint[:-1]]
        else:
            classDict[currClass].append(dataPoint[:-1])
    return classDict

def mean(listOfValues):
    total = 0
    for num in listOfValues:
        total += num
    return total/len(listOfValues)

def variance(listOfValues, meanValue):
    total = 0
    for num in listOfValues:
       total +=  (num - meanValue)**2/(len(listOfValues)-1)
    if total == 0:
        total = 0.00001
    return total

def summarizeDataset(dataset):
    summaryData = []
    for col in range(len(dataset[0])):
        currCol = []
        for dataPoint in dataset:
            currCol.append(dataPoint[col])
        colMean = mean(currCol)
        colVar = variance(currCol, colMean)
        colStDev = colVar**0.5
        summaryData.append((colMean, colStDev, len(currCol)))
    return summaryData

def summarizeByClass(dataset):
    classDict = seperateByClass(dataset)
    summaryDict = {}
    for currClass in classDict:
        classData = classDict[currClass]
        classSummary = summarizeDataset(classData)
        summaryDict[currClass] = classSummary
    return summaryDict

def calcProb(value, mean, std_dev):
    probability = ((1 / math.sqrt(2 * math.pi * std_dev)) * math.exp(-((value-mean) ** 2) / (2 * std_dev ** 2 )))
    return probability

def calcClassProbs(summaries, instance):
    probDict = {}
    totalLen = 0
    for currClass in summaries:
        totalLen += summaries[currClass][0][-1]
    for currClass in summaries:
        priorProb = summaries[currClass][0][-1]/totalLen
        #print(priorProb)
        classProb = priorProb
        for i in range(len(summaries[currClass])):
            colMean = summaries[currClass][i][0]
            colStdDev = summaries[currClass][i][1]
            #print('mean', colMean, 'stdDev', colStdDev)
            colProb = calcProb(instance[i],colMean,colStdDev)
            classProb *= colProb
        probDict[currClass] = classProb
    return probDict

def predict(summaries, instance):
    probDict = calcClassProbs(summaries, instance)
    return max(probDict.items(), key=operator.itemgetter(1))[0]

def naiveBayes(train, test):
    classSummary = summarizeByClass(train)
    return[predict(classSummary, instance) for instance in test]


def splitData(data, trainRatio):
    class0 = [i if i[len(i)-1] == 0 else [] for i in data]
    class1 = [i if i[len(i)-1] == 1 else [] for i in data]

    while ([] in class0):
        class0.remove([])

    while ([] in class1):
        class1.remove([])

    random.shuffle(class0)
    random.shuffle(class1)

    trainIndex0 = round(trainRatio*len(class0))
    trainData = class0[0:trainIndex0]
    testData =  class0[trainIndex0:len(class0)]

    trainIndex1 = round(trainRatio * len(class1))
    trainData = trainData +  class1[0:trainIndex1]
    testData = testData + class1[trainIndex1:len(class1)]

    random.shuffle(trainData)
    random.shuffle(testData)

    return [trainData, testData]

def splitTrainData(trainData, trainPerc):
    trainSplitData = []
    class0 = [i if i[len(i)-1] == 0 else [] for i in trainData]
    class1 = [i if i[len(i)-1] == 1 else [] for i in trainData]

    while ([] in class0):
        class0.remove([])

    while ([] in class1):
        class1.remove([])

    random.shuffle(class0)
    random.shuffle(class1)

    trainIndex0 = round(trainPerc * len(class0))
    trainSplitData = trainSplitData + class0[0:trainIndex0]

    trainIndex1 = round(trainPerc * len(class1))
    trainSplitData = trainSplitData + class1[0:trainIndex1]

    random.shuffle(trainSplitData)

    return trainSplitData

def loadData(filename):
    csvTxt = csv.reader(open(filename))
    data = []
    for row in csvTxt:
        data.append(row)
    return data

def accuracy(actual, predicted):
    counter = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            counter += 1
    return float(1-counter/len(actual))

def column2Float(dataset, column):
    for instance in dataset:
        instance[column] = float(instance[column])
    return dataset
