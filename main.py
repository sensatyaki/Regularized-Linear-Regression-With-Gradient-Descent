
# coding: utf-8

# In[1]:

import numpy as np
import csv
import copy

np.set_printoptions(suppress=True);



# In[2]:

def dataNormalize(X):
    meanVal = []
    stdVal = []
    xNorm = X
    for i in range(X.shape[1]):
        m = np.mean(X[:,i])
        s = np.std(X[:,i])
        meanVal.append(m)
        stdVal.append(s)
        xNorm[:,i] = (xNorm[:,i] - m) / s
    return xNorm,meanVal,stdVal


# In[3]:

def powArr(arr,power):
    res = 0
    for i in arr:
        res = res + (i**power)
    return res
        


# In[ ]:




# In[20]:

def gradientDescentRidge(data,trainY,alpha,lamb,ep):
    
    converged = False
    
    M = data.shape[0]
    W = np.ones(data.shape[1])
    
    #grad = [0]*M
    it = 0
    
    error = (np.sum((np.dot(data,W) - trainY)**2))/(2.0 * M)
    
    regularizedError = error + lamb*np.sum(np.power(W,2))
    
    
    while not converged:
        it = it + 1
        
        loss = np.dot(data,W) - trainY
        grad = (np.dot(data.T,loss))/M + (2.0 * lamb * W)
        #print(grad)
        W = W - (alpha * grad)
        #print(W)
        newError = (np.sum((np.dot(data,W) - trainY)**2))/(2.0 * M)
        newRegularizedError = newError + lamb*np.sum(np.power(W,2))
        
        if(abs(regularizedError-newRegularizedError) < ep):
            converged = True

        regularizedError = newRegularizedError

       
            
    return W


# In[21]:

def gradientDescentLPNorm(data,trainY,alpha,lamb,ep,P):
    
    converged = False
    
    M = data.shape[0]
    W = np.ones(data.shape[1])
    
    #grad = [0]*M
    it = 0
    #print(w)
    print(W.shape)
    print(data.shape)
    error = (np.sum((np.dot(data,W) - trainY)**2))/(2.0 * M)
    
    regularizedError = error + lamb*np.sum(np.power(np.absolute(W),P))
    
    
    
    while not converged:
        
        it = it + 1
        
        loss = np.dot(data,W) - trainY
        grad = (np.dot(data.T,loss))/M + (P * lamb * (np.power(np.absolute(W),P-1)))
        
        W = W - (alpha * grad)
        
        newError = (np.sum((np.dot(data,W) - trainY)**2))/(2.0 * M)
        newRegularizedError = newError + lamb*np.sum(np.power(np.absolute(W),P))
        
        if(abs(regularizedError-newRegularizedError) < ep):
            converged = True

        regularizedError = newRegularizedError

        #if(it>10):
            #converged = True
         
    return W


# In[ ]:




# In[22]:

def extractData():
    data = np.genfromtxt("data/train.csv",delimiter=",",skip_header=1)
    testData = np.genfromtxt("data/test.csv",delimiter=",",skip_header=1)
    fileOutputFirstCol = copy.deepcopy(testData[:,:1])
    testData = testData[:,1:14]
    #N0. of train data to train the model--------------------------
    data = data[0:280,1:15]
    tempY = copy.deepcopy(data[:,-1:])
    trainY=( val[0] for val in tempY);
    trainY=list(trainY);
    trainY=np.array(trainY);
    data = data[:,0:13]
    data,meanVal,stdVal = dataNormalize(data)
    
    testData,testMean,testSd = dataNormalize(testData)
    
    
    
    data=np.column_stack((np.ones((data.shape[0],1)),data))
    
    testData=np.column_stack((np.ones((testData.shape[0],1)),testData))
    print(testData.shape)
    return data,testData,trainY,fileOutputFirstCol


# In[23]:

def featureEngineeredData():
    data = np.genfromtxt("data/train.csv",delimiter=",",skip_header=1)
    testData = np.genfromtxt("data/test.csv",delimiter=",",skip_header=1)
    fileOutputFirstCol = copy.deepcopy(testData[:,:1])
    testData = testData[:,1:14]
    #N0. of train data to train the model--------------------------
    data = data[0:280,1:15]
    tempY = copy.deepcopy(data[:,-1:])
    trainY=( val[0] for val in tempY);
    trainY=list(trainY);
    trainY=np.array(trainY);
    data = data[:,0:13]
    data,meanVal,stdVal = dataNormalize(data)
    
    testData,testMean,testSd = dataNormalize(testData)
    
    #----------------------code for Non_linear closed form-----------------------------------------
    squareData = copy.deepcopy(data)
    rootData = copy.deepcopy(data)
    squareData = (squareData ** 2)
    rootData  = np.power(np.absolute(rootData),0.5)
    
    data = np.column_stack((squareData,data))
    data = np.column_stack((rootData,data))
    
    
    squareTestData = copy.deepcopy(testData)
    rootTestData = copy.deepcopy(testData)
    squareTestData = (squareTestData ** 2)
    rootTestData  = np.power(np.absolute(rootTestData),0.5)
    
    testData = np.column_stack((squareTestData,testData))
    testData = np.column_stack((rootTestData,testData))
    
    #------------------------------------------------------------------------------------------------
    
    data=np.column_stack((np.ones((data.shape[0],1)),data))
    
    testData=np.column_stack((np.ones((testData.shape[0],1)),testData))
    print(testData.shape)
    return data,testData,trainY,fileOutputFirstCol


# In[28]:

def featureEngineeredDataForClosedForm():
    data = np.genfromtxt("data/train.csv",delimiter=",",skip_header=1)
    testData = np.genfromtxt("data/test.csv",delimiter=",",skip_header=1)
    fileOutputFirstCol = copy.deepcopy(testData[:,:1])
    testData = testData[:,1:14]
    #N0. of train data to train the model--------------------------
    data = data[0:280,1:15]
    tempY = copy.deepcopy(data[:,-1:])
    trainY=( val[0] for val in tempY);
    trainY=list(trainY);
    trainY=np.array(trainY);
    data = data[:,0:13]
    
    
    
    squareData = copy.deepcopy(data)
    rootData = copy.deepcopy(data)
    oneFourthData = copy.deepcopy(data)
    oneEighthData = copy.deepcopy(data)
    
    squareData = (squareData ** 2)
    rootData  = np.power(np.absolute(rootData),0.5)
    oneFourthData = np.power(np.absolute(oneFourthData),0.25)
    oneEighthData = np.power(np.absolute(oneEighthData),0.125)
    data = np.column_stack((squareData,data))
    data = np.column_stack((rootData,data))
    data = np.column_stack((oneFourthData,data))
    data = np.column_stack((oneEighthData,data))
    
    squareTestData = copy.deepcopy(testData)
    rootTestData = copy.deepcopy(testData)
    oneFourthTestData = copy.deepcopy(testData)
    oneEighthTestData = copy.deepcopy(testData)
    
    squareTestData = np.power(np.absolute(squareTestData),2)
    rootTestData  = np.power(np.absolute(rootTestData),0.5)
    oneFourthTestData = np.power(np.absolute(oneFourthTestData),0.25)
    oneEighthTestData = np.power(np.absolute(oneEighthTestData),0.125)
    
    testData = np.column_stack((squareTestData,testData))
    testData = np.column_stack((rootTestData,testData))
    testData = np.column_stack((oneFourthTestData,testData))
    testData = np.column_stack((oneEighthTestData,testData))
    #------------------------------------------------------------------------------------------------
    
    
    data=np.column_stack((np.ones((data.shape[0],1)),data))
    
    testData=np.column_stack((np.ones((testData.shape[0],1)),testData))
    
    return data,testData,trainY,fileOutputFirstCol



# In[ ]:




# In[29]:

def ridgeClosedForm(data,Y,lamb):
    
    
    closedRidgeWeightNonLinear=np.dot(np.dot(np.linalg.inv(np.dot(data.T,data) + lamb*np.identity(data.shape[1])),data.T),Y)
    
    return closedRidgeWeightNonLinear


# In[30]:



data,testData,trainY,fileOutputFirstCol = featureEngineeredDataForClosedForm()
ridgeClosedFormW = ridgeClosedForm(data,trainY,0.00009)







createOutputFile(testData,ridgeClosedFormW,fileOutputFirstCol)




# In[12]:

def createOutputFile(testData,weight,fileOutputFirstCol):
    predictedY = np.dot(testData,weight)
    
    output=np.column_stack((fileOutputFirstCol,predictedY))
    np.savetxt('predicted.csv',output,fmt="%d,%.2f",header="ID,MEDV",comments ='')
    



