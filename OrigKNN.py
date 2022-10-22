import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from sklearn import datasets

print("test")

def calculateDistance(point1, point2, p=1):
    # minkowski distance
    # p = 2 --> euclidean distance
    dimension = len(point1)
    dist = 0
    for i in dimension:
        dist = dist + abs(point1-point2)**p
        
    dist = dist**(1/p)
    
    return dist


def CalculateKNN(train_x, test_x, train_y, test_y, k, p):
    
    