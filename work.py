# You are allow to use libraries for optimisation and linear algebraic caculations
# You are not allow to use libraries for basic pre-processing or classification
# e.g: can :
#           scipy.optimize for gradient descent
#           scipy.linalg.svd for matrix decomposition
#     can't:
#           sklearn.svm for classification.
###############################################
#   optimize:   https://blog.csdn.net/c406495762/article/details/77851973
#   Logistic:   https://www.imooc.com/article/69514
#   PCA:    https://blog.csdn.net/u012162613/article/details/42177327
#   train, test, cross validaton    https://blog.csdn.net/kieven2008/article/details/81582591
##############################################

import h5py
import numpy as np
np.set_printoptions(suppress = True)
import scipy as sci
import matplotlib.pyplot as plt
import operator
import math
import time
from sklearn.decomposition import PCA

with h5py.File('images_training.h5','r') as H:
    data = np.copy(H['data'])
with h5py.File('labels_training.h5','r') as H:
    label = np.copy(H['label'])
with h5py.File('labels_testing_2000.h5','r') as H:
    test_label_2000 = np.copy(H['label'])
with h5py.File('images_testing.h5','r') as H:
    test_data = np.copy(H['data'])

##############################################
def to2D(input):
    outPut = input.reshape(input.shape[0], -1)
    return outPut

# subtact each value with mean value
def subtractMean(data):
    new_data = np.matrix(data)
    mean=np.mean(new_data,axis=0)
    result=new_data - mean
    return result, mean

# n = number of egien vaule to keep
def pca(data, n):
    new_data, mean = subtractMean(data)

    covMat=np.cov(new_data, rowvar = 0)               # get covariance
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) # get eigenvalue , egienvector
    #print('\nEigenvalues \n %a' % eigVals)

    sorted_distances = np.argsort(eigVals)            # sort the eigenvalue
    n_eigVal_index = sorted_distances[-1:-(n+1):-1]   # get the index of minimum slected egien value
    n_eigVect = eigVects[:, n_eigVal_index]           # get the k egien vector
    lowDData = new_data * n_eigVect                   # get the low dimention data
    reconData = (lowDData * n_eigVect.T) + mean       # reconstruct data for debug

    lowDData = np.squeeze(np.asarray(lowDData))   # matrix-> array
    reconData = np.squeeze(np.asarray(reconData))   # matrix-> array

    return lowDData,reconData


##############################################
twoD_data = to2D(data)
twoD_test_data = to2D(test_data)
################################################
# PCA, return numpy array
lowD_data, recon_Data = pca(twoD_data, 100)
lowD_data.shape
#lowD_testData, recon_test_data = pca(twoD_test_data, 0.99)
# slice test data to 2000
################################################


################################################
a = twoD_data[0:1000, :]
b = label[0:1000]
c = twoD_test_data[0:100, :]
d = test_label_2000[0:100]
################################################
def MaxMinNormalization(data,test):
    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min);
    test = (test - min) / (max - min);

    return data, test;

qq,ww =  MaxMinNormalization(twoD_data, twoD_test_data)
################################################

# return the distances between target element and all element in the data set
def euDistance(target, dataSet):
    numSamples = dataSet.shape[0]
    diffMat = np.tile(target,(numSamples,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    return np.sqrt(sqDistances)

def kNNClassify(testData, dataSet, labels, k):

    distances = euDistance(testData, dataSet)
    sorted_distances = np.argsort(distances)

    classCount = {}
    for i in range(k):
        voteLabel = labels[sorted_distances[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex

def fit(a,b,c,d, k):
    train_x = a
    train_y = b
    test_x = c
    test_y = d

    startTime = time.time()

    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in range(numTestSamples):
        predict = kNNClassify(test_x[i], train_x, train_y, k)
        print('predict =', predict, 'actual =', test_y[i])
        if predict == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples

    print('time used =', time.time() - startTime)
    print ('The accuracy is: %.2f%%' % (accuracy * 100))

################################################
fit(qq,label,ww[0:2000,:], test_label_2000, 9)

def testBestK():
    for k in range(10,20):
        print('running k = ', k)
        fit(twoD_data,label,twoD_test_data[0:2000,:], test_label_2000, k)

    print('testBestK finished')
testBestK()

def testPca(data, testData):
    n = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    for i in range(len(n)):
        outPut, recon =  pca(data, n[i])
        testoutput, testRecon = pca(test_data, n[i])

        #fit(outPut,label, testoutput[0:2000,:], test_label_2000, 9)

testPca(twoD_data, twoD_test_data)

sklearnPCA = PCA(n_components = 10)
sk = sklearnPCA.fit_transform(twoD_data)
sktest =  sklearnPCA.fit_transform(twoD_test_data)

a,b  = pca(twoD_data, 10)
c,d  = pca(twoD_test_data, 10)

fit (sk,label, sktest[0:2000,:], test_label_2000, 9)
