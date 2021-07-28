import csv
import math
import random
import numpy as np
import pandas as pd

def logisticRegression(filename, num_splits, train_percent):
    statDic = {}
    for j in train_percent:
        statDic[j] = []
    data = loadData(filename)
    for column in range(len(data[0])):
        column2Float(data, column)
    #print(data[0])
    for i in range(num_splits):
        print(i)
        trainTest = splitData(data, 0.8)
        train = trainTest[0]
        test = trainTest[1]
        #print(train[0])
        #print(test)
        testClasses = [i[len(i)-1] for i in test]
        for percentTrain in train_percent:
            #print(percentTrain)
            splitTrain = splitTrainData(train, percentTrain)
            #print(splitTrain)
            predicted = clr(splitTrain, test, 0.000005, 200)
            acc = accuracy(testClasses, predicted)
            print("current error:", acc)
            statDic[percentTrain].append(acc)
    df = pd.DataFrame(statDic)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    #df.to_csv("C:/Users/caulf/Desktop/MLHW2/LRoutput.csv")


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
    testData = class0[trainIndex0:len(class0)]

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

def predictCLR(instance, coefficients):
    power = coefficients[0]
    for i in range(len(instance)-1):
        power += instance[i]*coefficients[i+1]
    #print(power)
    predY = 1.0 / (1.0 + np.exp(-power))
    return predY

def sgdLog(dataset, learning_rate, epochs):
    coefficients = [0 for i in range(len(dataset[0]))]
    for e in range(epochs):
        totalError = 0
        for instance in dataset:
            predY = predictCLR(instance, coefficients)
            error = instance[-1] - predY
            totalError += error**2
            coefficients[0] += learning_rate*error*predY*(1.0-predY)
            for i in range(1,len(coefficients)):
                coefficients[i] += learning_rate*error*predY*(1.0-predY)*instance[i-1]
        #print('epoch=', e, 'lrate=', learning_rate, 'error=%.3f' %totalError)
    return coefficients

def clr(train, test, learning_rate, epochs):
    coefficients = sgdLog(train, learning_rate, epochs)
    #print(coefficients)
    predictions = []
    for entry in test:
        prediction = predictCLR(entry, coefficients)
        predictions.append(round(prediction))
    return predictions

def accuracy(actual, predicted):
    counter = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            counter += 1
    return float(1-counter/len(actual))

def column2Float(dataset,column):
    for instance in dataset:
        instance[column] = float(instance[column])
    return dataset