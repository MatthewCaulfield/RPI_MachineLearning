import logisiticRegression as lr
import csv
import naiveBayesGaussian as nbg
trainingPercents = [5, 10, 15, 20, 25, 30]
numberSplits = 100
fileName = "C:/Users/caulf/Desktop/spambase/spambasedata.csv"
lr.logisticRegression("C:/Users/caulf/Desktop/spambase/spambasedata.csv", numberSplits, trainingPercents)
nbg.naiveBayesGaussian("C:/Users/caulf/Desktop/spambase/spambasedata.csv", numberSplits , trainingPercents)