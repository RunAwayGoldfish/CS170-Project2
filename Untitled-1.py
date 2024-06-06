#py -m pip install opencv-python

import numpy as np
import math

# Read the data from a file
#df = pd.read_csv('small-test-dataset.txt', delim_whitespace=True, header=None)

data_array = np.loadtxt('large-test-dataset.txt')

first_column = data_array[:, 0]

# Normalize the rest of the columns (excluding the first column)
data_normalized = np.copy(data_array)
for i in range(1, data_normalized.shape[1]):
    col = data_normalized[:, i]
    col_mean = np.mean(col)
    col_std = np.std(col)
    data_normalized[:, i] = (col - col_mean) / col_std

# Restore the first column
data_normalized[:, 0] = first_column


class featuresAndAccuracy:
    # Constructor method (initializes objects)
    def __init__(self, featureList, Accuracy):
        self.featureList = featureList
        self.Accuracy = Accuracy


def findBestFeatures(data, currList, remainingFeatureList, combinations):
    print("Features remaining", len(remainingFeatureList))
    if(len(remainingFeatureList) == 0):
        return
    bestAccuracy = -1
    newBestList = []
    addedFeature = -1
    for i in remainingFeatureList:
        newList = currList[:]
        newList.append(i)
        #print("New list", newList)
        newListAccuracy = findAccuracy(data, newList)
        if(newListAccuracy >= bestAccuracy):
            bestAccuracy = newListAccuracy
            newBestList = newList
            addedFeature = i
    combinations.append(featuresAndAccuracy(newBestList, bestAccuracy))
    remainingFeatureList.remove(addedFeature)
    currList = newBestList
    findBestFeatures(data, currList, remainingFeatureList, combinations)
    
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
        #print(currAccuracy, currAccuracy)
        x = featuresAndAccuracy(currList, currAccuracy)
        return x

    currList = newBestList
    currAccuracy = newBestAccuracy
    remainingFeatureList.remove(addedFeature)
    return findBestFeatures2(data, currList, remainingFeatureList, currAccuracy)
        

def main2(data):
    results1 = main(data)
    results2 = main(data, findBestFeatures2)
    if(results1.Accuracy != results2.Accuracy):
        print("FUCK")
    else:
        print(results1.Accuracy, results1.featureList)


def main(data, searchFunc = findBestFeatures):
    featureCount = data.shape[1]
    remainingFeatureList = [i for i in range(1, featureCount)]
    currList = []
    combinations = []

    print(remainingFeatureList)
    
    if(searchFunc == findBestFeatures):
        findBestFeatures(data, currList, remainingFeatureList, combinations)
        best = featuresAndAccuracy([],0)
        for i in range(len(combinations)):
            if(best.accuracy > combinations[i].accuracy):
                best = combinations[i]
            return best
    if(searchFunc == findBestFeatures2):
        bestFeatureList = []
        currAccuracy = 0
        best = findBestFeatures2(data, currList, remainingFeatureList, currAccuracy)
        return best



def findAccuracy(data, features):
    correct = 0
    for i in range(data.shape[0]):
        NN = findNearestNeighbor2(i, data, features)
        if(NN == data[i][0]):
            correct +=1
    accuracy = correct / data.shape[0]
    #print(accuracy)
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



best = main(data_normalized, findBestFeatures2)
print(best.Accuracy)
print(best.featureList)
