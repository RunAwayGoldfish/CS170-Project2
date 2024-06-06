#py -m pip install opencv-python

import numpy as np
import math


data_array = np.loadtxt('small-test-dataset.txt')

first_column = data_array[:, 0]

# Noramlizing Features
data_normalized = np.copy(data_array)
for i in range(1, data_normalized.shape[1]):
    col = data_normalized[:, i]
    col_mean = np.mean(col)
    col_std = np.std(col)
    data_normalized[:, i] = (col - col_mean) / col_std

data_normalized[:, 0] = first_column


class featuresAndAccuracy:
    # Constructor method (initializes objects)
    def __init__(self, featureList, Accuracy):
        self.featureList = featureList
        self.Accuracy = Accuracy

def backwardSelection(data, currList, remainingFeatureList, currAccuracy):
    if(len(remainingFeatureList) == 0):
        return featuresAndAccuracy(currList, currAccuracy)
    
    newBestAccuracy = -1
    newBestList = []
    addedFeature = -1
    for i in remainingFeatureList:
        newList = currList[:]
        newList.remove(i)
        
        newListAccuracy = findAccuracy(data, newList)
        if(newListAccuracy >= newBestAccuracy):
            newBestAccuracy = newListAccuracy
            newBestList = newList
            addedFeature = i

    if(newBestAccuracy < currAccuracy):
        x = featuresAndAccuracy(currList, currAccuracy)
        return x

    currList = newBestList
    currAccuracy = newBestAccuracy
    remainingFeatureList.remove(addedFeature)
    return backwardSelection(data, currList, remainingFeatureList, currAccuracy)




def findBestFeatures2(data, currList, remainingFeatureList, currAccuracy):
    print("Features remaining", len(remainingFeatureList))
    if(len(remainingFeatureList) == 0):
        return featuresAndAccuracy(currList, currAccuracy)
    
    newBestAccuracy = -1
    newBestList = []
    addedFeature = -1
    for i in remainingFeatureList:
        newList = currList[:]
        newList.append(i)
        
        newListAccuracy = findAccuracy(data, newList)
        if(newListAccuracy >= newBestAccuracy):
            newBestAccuracy = newListAccuracy
            newBestList = newList
            addedFeature = i

    if(newBestAccuracy < currAccuracy):
        x = featuresAndAccuracy(currList, currAccuracy)
        return x

    currList = newBestList
    currAccuracy = newBestAccuracy
    remainingFeatureList.remove(addedFeature)
    return findBestFeatures2(data, currList, remainingFeatureList, currAccuracy)


def main(data, mode="Forward"):
    featureCount = data.shape[1]
    remainingFeatureList = [i for i in range(1, featureCount)]
    currList = []
    currAccuracy = 0
    best = None
    if(mode == "Forward"):
        best = findBestFeatures2(data, currList, remainingFeatureList, currAccuracy)
    elif(mode == "Backwards"):
        currList = remainingFeatureList[:]
        best = backwardSelection(data,currList, remainingFeatureList, currAccuracy)

    print("Best feature list:", best.featureList)
    print("Best feature list accuracy: ", best.Accuracy)



def findAccuracy(data, features):
    correct = 0
    for i in range(data.shape[0]):
        NN = findNearestNeighbor2(i, data, features)
        if(NN == data[i][0]):
            correct +=1
    accuracy = correct / data.shape[0]
    return accuracy

def findNearestNeighbor2(index, data, features):
    item = data[index]
    MinDist = 100000000000
    Classifier = 0
    for i in range(data.shape[0]):
        row = data[i]
        if(i != index):
            dist = 0
            for j in features:
                dist += math.pow(item[j] - row[j],2)
            dist = math.pow(dist,1/2)

            if(dist < MinDist):
                MinDist = dist
                Classifier = row[0]
    return Classifier

#print(findAccuracy(data_normalized, [1,15,27]))



main(data_normalized, mode="Backwards")