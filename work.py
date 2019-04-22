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
import scipy as sci
import matplotlib.pyplot as plt

with h5py.File('images_training.h5','r') as H:
    data = np.copy(H['data'])
with h5py.File('labels_training.h5','r') as H:
    label = np.copy(H['label'])
with h5py.File('labels_testing_2000.h5','r') as H:
    test_label_2000 = np.copy(H['label'])
with h5py.File('images_testing.h5','r') as H:
    test_data = np.copy(H['data'])


##############################################
#不需要标准化因为图片特征的范围都是 0 ~255
#
#
##############################################
twoD_data = to2D(data)
twoD_data.shape
twoD_data = twoD_data.T
twoD_test_data = to2D(test_data)
twoD_test_data.shape


label.shape


def to2D(input):
    outPut = input.reshape(input.shape[0], -1)
    return outPut

def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData,meanVal

def PCA(dataMat, percentage):
    newData,meanVal=zeroMean(twoD_data)
    covMat=np.cov(newData, rowvar = 0)     #求协方差矩阵

    eigVals,eigVects=np.linalg.eig(np.mat(covMat))     #求特征值， 特征矩阵
    k=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量

    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    k_eigValIndice=eigValIndice[-1:-(k+1):-1]   #最大的n个特征值的下标
    k_eigVect=eigVects[:,k_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=newData*k_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*k_eigVect.T)+meanVal  #重构数据

    return lowDDataMat,reconMat


def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

lowDDataMat, reconMat = PCA(twoD_data, 0.9)
lowDDataMat.shape



######
#logistic regression
######
lowDDataMat_T = lowDDataMat.T
lowDDataMat_T.shape

label_reshape = label[np.newaxis, :]
label_reshape.shape

test_data.shape
test_data_2000
test_label_2000.shape
test_label_2000_reshape = test_label_2000[np.newaxis, :]
test_label_2000_reshape.shape

######
def sigmoid(z):
    y = 1/(1+np.exp(-z))
    return y
# 列向量w, 传入维度dim, 返回shape(dim, 1)
def init_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b
def propagate(w, b, X, Y):
    """

    传参:
    w -- 权重, shape： (num_px * num_px * 3, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (num_px * num_px * 3, m),m为样本数
    Y -- 真实标签，shape： (1,m)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """
    #获取样本数m：
    m = X.shape[1]
    # 前向传播 ：
    A = sigmoid(np.dot(w.T,X)+b)    #调用前面写的sigmoid函数
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m

    # 反向传播：
    dZ = A-Y
    dw = (np.dot(X,dZ.T))/m
    db = (np.sum(dZ))/m
    #返回值：
    grads = {"dw": dw,             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    #定义一个costs数组，存放每若干次迭代后的cost，从而可以画图看看cost的变化趋势：
    costs = []    #进行迭代：
    for i in range(num_iterations):        # 用propagate计算出每次迭代后的cost和梯度：
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        # 用上面得到的梯度来更新参数：
        w = w - learning_rate*dw
        b = b - learning_rate*db
        # 每100次迭代，保存一个cost看看：
        if i % 100 == 0:
            costs.append(cost)
        # 这个可以不在意，我们可以每100次把cost打印出来看看，从而随时掌握模型的进展：
        if print_cost and i % 100 == 0:            print ("Cost after iteration %i: %f" %(i, cost))    #迭代完毕，将最终的各个参数放进字典，并返回：
    params = {"w": w,              "b": b}
    grads = {"dw": dw,             "db": db}
    return params, grads, costs


def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    A = sigmoid(np.dot(w.T,X)+b)
    for  i in range(m):
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction

def logistic_model(X_train, Y_train,   learning_rate=0.1, num_iterations=2000, print_cost=False):
    #获特征维度，初始化参数：
    #X_test,
    #Y_test,
    dim = X_train.shape[0]
    W,b = initialize_with_zeros(dim)    #梯度下降，迭代求出模型参数：
    params,grads,costs = optimize(W,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    W = params['w']
    b = params['b']    #用学得的参数进行预测：
    #prediction_train = predict(W,b,X_test)
    prediction_test = predict(W,b,X_train)    #计算准确率，分别在训练集和测试集上：
    accuracy_train = 1 - np.mean(np.abs(prediction_train - Y_train))
    accuracy_test = 1 - np.mean(np.abs(prediction_test - Y_test))
    print("Accuracy on train set:",accuracy_train )
    print("Accuracy on test set:",accuracy_test )   #为了便于分析和检查，我们把得到的所有参数、超参数都存进一个字典返回出来：
    dic_result = {"costs": costs,
         #"Y_prediction_test": prediction_test ,
         "Y_prediction_train" : prediction_train ,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         #"train_acy":train_acy,
         #"test_acy":test_acy
        }
    return dic_result

dic_result = logistic_model(lowDDataMat_T, label_reshape)
