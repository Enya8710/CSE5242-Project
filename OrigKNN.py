import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from sklearn.neighbors import KNeighborsClassifier

def calculateDistance(test_point, train_point, p=2):
    # minkowski distance
    # p = 2 --> euclidean distance
    dist = np.sum((test_point-train_point)**p)
    dist = dist**(1/p)
    return dist

def calculateDistance2(test_point,train_point):
    # Euclidean distance
    dist = np.sqrt(np.sum((test_point-train_point)**2))
    return dist

def origKNN(train_x, test_x, train_y, k, p=2):
    
    predList = []
    
    for test_point in test_x:
        distList = []
        for train_point in train_x:
            distance = calculateDistance(test_point,train_point, p)
            distList.append(distance)
        
        distPd = pd.DataFrame(data = distList, columns= ['distance'], index = train_y.index)
        # default sorting is quicksort options: mergesort heapsort stable
        distSorted = distPd.sort_values(by = ['distance'], kind = 'quicksort')[:k]
        count = Counter(train_y[distSorted.index])
        finalPrediction = count.most_common(1)[0][0]
        predList.append(finalPrediction)
        
    return predList
        
        
irisData = datasets.load_iris()
data,target = irisData.data, irisData.target
df = pd.DataFrame(data = data, columns = irisData.feature_names)
df['target'] = target
x = df.drop('target', axis = 1)
y = df['target']
# print(df.target)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

start = time.time()

predResult = origKNN(train_x, test_x, train_y, 3)

end = time.time()

print("Original KNN running time: {}".format(end-start))
print(accuracy_score(test_y, predResult))


# start = time.time()
# knn = KNeighborsClassifier(n_neighbors=3, p=2)
# knn.fit(train_x, train_y)
# predResult = knn.predict(test_x)
# end = time.time()
# print("built-in KNN running time: {}".format(end-start))
# print(accuracy_score(test_y, predResult))

    
            
          
        
    