import h5py
import numpy as np

with h5py.File('images_training.h5','r') as H:
    data = np.copy(H['data'])

with h5py.File('labels_training.h5','r') as H:
    label = np.copy(H['label'])

#(reshape the data to 30000, 784)
twoD_data = data.reshape(30000, -1)

# run pca and keep 90% eigenValue
lowDinputData = pca(twoD_data, 0.9)




    #get mean by col
def zeroMean(inputData):
    meanVal=np.mean(inputData,axis=0)
    newData=inputData-meanVal
    return newData,meanVal

def pca(inputData,percentage):
    newData,meanVal=zeroMean(inputData)

    covMat=np.cov(newData,rowvar=0)    #get cov
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))     #get eigenValue, eigenVector
    k=getPercentageOfEig(eigVals,percentage)                 #difine how many % of eigVects to keep

    resortEigVal=np.argsort(eigVals)            #resort eigenValue
    k_resortEigVal=resortEigVal[-1:-(k+1):-1]
    k_eigVect=eigVects[:,k_resortEigVal]        #get eigenVector by resort eigenValue
    lowDData=newData*k_eigVect               # get the lower dimension data

    return lowDData


def getPercentageOfEig(eigVals,percentage):     #return the number of eigenValue to keep by percentage
    sortArray=np.sort(eigVals)
    sortArray=sortArray[-1::-1]
    arraySum=sum(sortArray)
    tmpSum=0
    num=0

    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num
