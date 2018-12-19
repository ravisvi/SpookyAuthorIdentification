# -*- coding: utf-8 -*-
"""
Created on Sat Nov 03 16:03:06 2018

@author: OH YEA
"""
import sys
import csv
import nltk
import itertools
import time
import numpy as np
import operator
from nltk import FreqDist

nltk.download('punkt')

def kaggleize(predictions,file):

	if(len(predictions.shape)==1):
		predictions.shape = [predictions.shape[0],1]

	ids = 1 + np.arange(predictions.shape[0])[None].T
	kaggle_predictions = np.hstack((ids,predictions)).astype(int)
	writer = csv.writer(open(file, 'w'))
	writer.writerow(['# id','Prediction'])
	writer.writerows(kaggle_predictions)


def GetFeatures(localWords, allWords):
    features = {}
    for word in allWords:
        features['contains({})'.format(word)] = (word in localWords)
    return features

outputFile = 'NB.csv'

startTime = time.time()
theAuthors = {'EAP':0, 'HPL':1, 'MWS':2}

trainSet = []
allWords = []
testSet = []
idNumber = []
###########################################################################
with open('train.csv', encoding='utf8') as csvfile:
    theReader = csv.reader(csvfile)
    for row in theReader:
        allWords += nltk.word_tokenize(row[1])

tempDist = FreqDist(allWords)

mostFreq = sorted(tempDist.items(), key=operator.itemgetter(1), reverse=True)
leastFreq = sorted(tempDist.items(), key=operator.itemgetter(1), reverse=False)

allMostFreq = [i[0] for i in mostFreq]
allLeastFreq = [i[0] for i in leastFreq]

theSize = 10
theOrder = "most"

results = []

for i in range(1):
	for j in range(1):
	
		theWords = []
		
		if (theOrder == "most"):
			theWords  = list(allMostFreq)[:theSize]
		else:
			theWords = list(allLeastFreq)[:theSize]

		with open('train.csv', encoding='utf8') as csvfile:
			theReader = csv.reader(csvfile)
			for row in theReader:
				label = theAuthors[row[2]]
				trainSet.append((GetFeatures(nltk.word_tokenize(row[1]), theWords), label))
		
		with open('test.csv', encoding='utf8') as csvfile:
			testReader = csv.reader(csvfile)
			for row in testReader:
				testSet.append((GetFeatures(nltk.word_tokenize(row[1]), theWords)))
				idNumber.append(row[0])

		splitNum = int(round(len(trainSet)*0.8))

		theClass = None
		theClass = nltk.NaiveBayesClassifier.train(trainSet[:splitNum])
		
		theAccuracy = nltk.classify.accuracy(theClass, trainSet[splitNum:])
		print([theSize, theOrder, theAccuracy])
		results.append([theSize, theOrder, theAccuracy])
		

tempThings = []
with open('results.csv', encoding='utf8') as csvfile:
	finalReader = csv.reader(csvfile)
	for row in finalReader:
		tempThings.append(row)

with open('results.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for tempThing in tempThings:
		writer.writerow(tempThing)
	for result in results:
		writer.writerow(result)
		
printCount = 0

thePredictions = []

for item in testSet:
	thePredictions.append(nltk.NaiveBayesClassifier.prob_classify(theClass, item))

with open('names.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')    
    writer.writerow(['id', 'EAP', 'HPL', 'MWS'])
    

    for theThing in thePredictions:
        prediction = theThing
        tempList = []
        tempList.append(idNumber[printCount])
        for label in prediction.samples():
            tempList.append(prediction.prob(label))
        printCount += 1
        writer.writerow(tempList)
