# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:04:15 2021

@author: Jonathan
"""


import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import math
import tqdm
import tensorflow as tf

#import stock price dfs
dfamzn = pd.read_csv('data/30MinAMZN2yrs.csv', index_col=(0))
dfaapl = pd.read_csv('data/30MinAAPL2yrs.csv', index_col=(0))
dfgoog = pd.read_csv('data/30MinGOOG2yrs.csv', index_col=(0))

#change dates to workable datetimes
for j in range(len(dfamzn)):
    dfamzn.loc[j,'date'] = parse(dfamzn['date'][j])

for j in range(len(dfaapl)):
    dfaapl.loc[j,'date'] = parse(dfaapl['date'][j])
    
for j in range(len(dfgoog)):
    dfgoog.loc[j,'date'] = parse(dfgoog['date'][j])

#import stock vix's
vxamzn = pd.read_csv('data/vxamzndailyprices.csv', header=1)
vxamzn = vxamzn[2180:].reset_index(drop=True)

vxaapl = pd.read_csv('data/vxaapldailyprices.csv', header=1)
vxaapl = vxaapl[2180:].reset_index(drop=True)

vxgoog = pd.read_csv('data/vxgoogdailyprices.csv', header=1)
vxgoog = vxgoog[2180:].reset_index(drop=True)

#add vix and sentiment features to 30min stock df
def addingDailyFeatures(vxStock, dfStock):
    dfStock = dfStock.copy()
    dfStock.loc[:,'vixOpen'] = 0
    dfStock.loc[:,'vixHigh'] = 0
    dfStock.loc[:,'vixLow'] = 0
    dfStock.loc[:,'vixClose'] = 0
    for i in range(len(dfStock)):
        for j in range(len(vxStock)):
            if dfStock.loc[i,'date'].date()==parse(vxStock.loc[j,'Date']).date():
                dfStock.loc[i,'vixOpen'] = vxStock.loc[j,'Open']
                dfStock.loc[i,'vixHigh'] = vxStock.loc[j,'High']
                dfStock.loc[i,'vixLow'] = vxStock.loc[j,'Low']
                dfStock.loc[i,'vixClose'] = vxStock.loc[j,'Close']
                print(i)
                break
    return dfStock

dfamzn = addingDailyFeatures(vxamzn, dfamzn)
dfaapl= addingDailyFeatures(vxaapl, dfaapl)
dfgoog = addingDailyFeatures(vxgoog, dfgoog)

#Create return features and target columns
def creatingReturnFeat(dfStock):
    df = dfStock.copy()
    #create vix-based return features
    df.loc[df.volume==0,'volume']=df['volume'].mean()
    for i in range(len(df.vixOpen)):
        df.loc[i,'vixRand'] = (np.random.random()*(df.loc[i,'vixHigh']-df.loc[i,'vixLow']))+df.loc[i,'vixLow']
    df['vixAve'] = (df.vixHigh+df.vixLow)/2
    vixTimestep = 14000
    df['close_open'] = np.log(df.close/df.opn)/((df.vixAve/100)/math.sqrt(vixTimestep))
    df['high_open'] = np.log(df.high/df.opn)/((df.vixAve/100)/math.sqrt(vixTimestep))
    df['low_open'] = np.log(df.opn/df.low)/((df.vixAve/100)/math.sqrt(vixTimestep))
    #create vix-based magnitude target
    df['large'] = 0
    df['small'] = 0
    df['magnitude'] = abs(np.log(df.close/df.opn))/((df.vixAve/100)/math.sqrt(vixTimestep))
    df.loc[(df.magnitude >= 0.675), 'large'] = 1
    df.loc[(df.magnitude < 0.675), 'small'] = 1
    for i in range(len(df['volume'])-1):
        df.loc[i+1,'volPercent'] = np.log(df.loc[i+1,'volume']/df.loc[i,'volume'])
    df['vixClose'] = df.vixClose/100
    #create direction target
    df['up'] = 0
    df['down'] = 0
    df.loc[(df.close_open >= 0), 'up'] = 1
    df.loc[(df.close_open < 0), 'down'] = 1 
    return df

dfamzn = creatingReturnFeat(dfamzn)
dfaapl = creatingReturnFeat(dfaapl)
dfgoog = creatingReturnFeat(dfgoog)

#shift target column and drop last row 
def shiftTarget(dfStock):
    df = dfStock.copy()
    df['large'] = df['large'].shift(-1)
    df['small'] = df['small'].shift(-1)
    df['up'] = df['up'].shift(-1)
    df['down'] = df['down'].shift(-1)        
    return df[78:(len(df)-1)].reset_index(drop=True)

dfamznShiftedTar = shiftTarget(dfamzn)
dfaaplShiftedTar = shiftTarget(dfaapl)
dfgoogShiftedTar = shiftTarget(dfgoog)

#split df into train/test
def trainTestSplit(dfStock):
    df = dfStock.copy()
    dfTrain = df[:4000]
    dfTest = df[4000:]
    return dfTrain, dfTest

dfamznTrain, dfamznTest = trainTestSplit(dfamznShiftedTar)
dfaaplTrain, dfaaplTest = trainTestSplit(dfaaplShiftedTar)
dfgoogTrain, dfgoogTest = trainTestSplit(dfgoogShiftedTar)

#combine train sets
def combTrainSet(*args):
    frms = [*args]
    combinedTrain = pd.concat(frms).reset_index(drop=True)
    return combinedTrain

combinedTrainNoNa = combTrainSet(dfamznTrain, dfaaplTrain, dfgoogTrain)

#build column transformer pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

def createPipeline(combinedTrainNoNa):
    dfTrain = combinedTrainNoNa.copy()[['close_open','high_open','low_open','volPercent','vixClose']]
    dfTrainTargets = combinedTrainNoNa.copy()[['large','small','up','down']]
    
    logReturns = ['close_open','high_open','low_open','volPercent']
    vix = ['vixClose']
    
    full_pipeline = ColumnTransformer([
        ("ret", StandardScaler(), logReturns),
        ("pass", 'passthrough', vix)
        ])   
    
    transformedTrainSet = full_pipeline.fit_transform(dfTrain)
    transformedTrainSet = transformedTrainSet, dfTrainTargets

    return transformedTrainSet, full_pipeline
        
transformedTrainSet, full_pipeline = createPipeline(combinedTrainNoNa)

#prep and transform test sets
def prepTestData(dfTest):
    df = dfTest.copy()
    dfNoTarget = df.copy()[['close_open','high_open','low_open','volPercent','vixClose']]
    testTargets = df.copy()[['large','small','up','down']]
    transformedTestSet = full_pipeline.transform(dfNoTarget)      
    return transformedTestSet, testTargets

amznTransformedTestSet = prepTestData(dfamznTest)
aaplTransformedTestSet = prepTestData(dfaaplTest)
googTransformedTestSet = prepTestData(dfgoogTest)

#shuffle datasets
from sklearn.utils import shuffle

def shuffleData(tup):
    shuffledData = shuffle(tup[0], tup[1], random_state=37)
    return shuffledData

shufTrainSet = shuffleData(transformedTrainSet)
amznShufTestSet = shuffleData(amznTransformedTestSet)
aaplShufTestSet = shuffleData(aaplTransformedTestSet)
googShufTestSet = shuffleData(googTransformedTestSet)

preppedMagData = (shufTrainSet[0], np.array(shufTrainSet[1][['large','small']]),
                  amznShufTestSet[0], np.array(amznShufTestSet[1][['large','small']]),
                  aaplShufTestSet[0], np.array(aaplShufTestSet[1][['large','small']]),
                  googShufTestSet[0], np.array(googShufTestSet[1][['large','small']]))
preppedDirData = (shufTrainSet[0], np.array(shufTrainSet[1][['up','down']]),
                  amznShufTestSet[0], np.array(amznShufTestSet[1][['up','down']]),
                  aaplShufTestSet[0], np.array(aaplShufTestSet[1][['up','down']]),
                  googShufTestSet[0], np.array(googShufTestSet[1][['up','down']]))

#functions to evaluate models and learning curves
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

def evalModel(yTrue, yPred):
    print(confusion_matrix(yTrue,yPred))
    print("Accuracy:", accuracy_score(yTrue,yPred))
    print("F1:", f1_score(yTrue,yPred))
    print("Precision:", precision_score(yTrue,yPred))
    print("Recall:", recall_score(yTrue,yPred))   

def plotRocCurve(yTrue, yPred):
    fpr, tpr, thresholds = roc_curve(yTrue, yPred)
    plt.plot(fpr, tpr, linewidth=2, label=None)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])                                    
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True) 
    print("AUC:", roc_auc_score(yTrue,yPred))   

def plotPrecRecall(yTrue, yPred):
    precisions, recalls, thresholds = precision_recall_curve(yTrue, yPred)
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

def plotLoss():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('tain loss vs val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def plotAcc():
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('train acc vs val acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


yTrue = np.argmax(preppedMagData[7], axis=1)
                  
#logistic regression modle with grid search and cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = [
    {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.01,0.1,0.3,0.5,0.7,0.9,1]},
    {'solver': ['liblinear'], 'penalty': ['l1'], 'C': [0.01,0.1,0.3,0.5,0.7,0.9,1]},
    {'solver': ['saga'], 'penalty': ['elasticnet'], 'l1_ratio': [0,0.1,0.3,0.5,0.7,0.9,1]},
    ]

logReg = LogisticRegression(random_state=42, max_iter=2000)
logRegGS = GridSearchCV(logReg, param_grid, cv=5, scoring='accuracy', return_train_score=True)
logRegGS.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

logRegGS.best_params_
bestLogReg = logRegGS.best_estimator_
results = logRegGS.cv_results_
dfResults = pd.DataFrame(results)

for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(mean_score, params)

evalModel(yTrue, bestLogReg.predict(preppedMagData[6]))
plotPrecRecall(yTrue, bestLogReg.decision_function(preppedMagData[6]))
plotRocCurve(yTrue, bestLogReg.decision_function(preppedMagData[6]))

#KNeighbors
from sklearn.neighbors import KNeighborsClassifier

param_grid = [
    {'n_neighbors': [3,15,35,75,105,155,205]}
    ]

knn = KNeighborsClassifier()
knnGS = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', return_train_score=True)
knnGS.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

knnGS.best_params_
bestKnn = knnGS.best_estimator_
results = knnGS.cv_results_
dfResults = pd.DataFrame(results)

for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(mean_score, params)

evalModel(yTrue, bestKnn.predict(preppedMagData[6]))
plotPrecRecall(yTrue, bestKnn.predict_proba(preppedMagData[6])[:,1])
plotRocCurve(yTrue, bestKnn.predict_proba(preppedMagData[6])[:,1])

#SVM models - grid search with multiple kernels
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5]},
    {'kernel': ['rbf'], 'gamma': [0.1, 0.5, 1, 5], 'C': [0.1, 0.5, 1, 5]},
    {'kernel': ['poly'], 'degree': [1, 3, 5],'coef0': [0.1, 0.5, 1, 5], 'C': [0.1, 0.5, 1, 5]}
    ]

SVCmod = SVC(probability=True)
SVCmodGS = GridSearchCV(SVCmod, param_grid, cv=5, scoring='accuracy', return_train_score=True)
SVCmodGS.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

SVCmodGS.best_params_
bestSVCmod = SVCmodGS.best_estimator_
results = SVCmodGS.cv_results_
dfResults = pd.DataFrame(results)

for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(mean_score, params)

evalModel(yTrue, bestSVCmod.predict(preppedMagData[6]))
plotPrecRecall(yTrue, bestSVCmod.decision_function(preppedMagData[6]))
plotRocCurve(yTrue, bestSVCmod.decision_function(preppedMagData[6]))

#sgd svm
sgdSVM = SGDClassifier(max_iter=2000, random_state=42) #SGDClassifier with hinge loss (svm)
sgdSVM.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

evalModel(yTrue, sgdSVM.predict(preppedMagData[6]))
plotPrecRecall(yTrue, sgdSVM.decision_function(preppedMagData[6]))
plotRocCurve(yTrue, sgdSVM.decision_function(preppedMagData[6]))

#build random forest model
from sklearn.ensemble import RandomForestClassifier

param_grid = [
    {'n_estimators': [50,100,300,500,1000],'min_samples_split': [2,4,7,10], 
     'min_samples_leaf': [1,2,4,7,10,15,20]}
    ]

RF = RandomForestClassifier(random_state=42)
RFGS = GridSearchCV(RF, param_grid, cv=5, scoring='accuracy', return_train_score=True)
RFGS.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

RFGS.best_params_
bestRF = RFGS.best_estimator_
results = RFGS.cv_results_
dfResults = pd.DataFrame(results)

for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(mean_score, params)

evalModel(yTrue, bestRF.predict(preppedMagData[6]))
plotPrecRecall(yTrue, bestRF.predict_proba(preppedMagData[6])[:,1])
plotRocCurve(yTrue, bestRF.predict_proba(preppedMagData[6])[:,1])

#feature importance with random forest
columns = ['tNeg','tNeut','tPos','dNeg','dNeut','dPos', 'artCnt', 'close_open','high_open','low_open','spy_close_open',
           'spy_high_open', 'spy_low_open','volPercent', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'vixClose','spy_vix']

for name, score in zip(columns, bestRF.feature_importances_):
    print(name, score)

#Ada Boost
from sklearn.ensemble import AdaBoostClassifier

param_grid = [
    {'n_estimators': [10, 100, 200, 500, 1000], 'learning_rate': [0.1, 0.5, 1, 5, 10]}
    ]

ada = AdaBoostClassifier(algorithm="SAMME.R", random_state=42)
adaGS = GridSearchCV(ada, param_grid, cv=5, scoring='accuracy', return_train_score=True)
adaGS.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

adaGS.best_params_
bestada = adaGS.best_estimator_
results = adaGS.cv_results_
dfResults = pd.DataFrame(results)

for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(mean_score, params)

evalModel(yTrue, bestada.predict(preppedMagData[6]))
plotPrecRecall(yTrue, bestada.decision_function(preppedMagData[6]))
plotRocCurve(yTrue, bestada.decision_function(preppedMagData[6]))

#gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

param_grid = [
    {'n_estimators': [50, 200, 500, 1000], 'learning_rate': [0.1, 0.5, 1, 5],
     'subsample': [0.1, 0.5, 1], 'max_depth': [2, 3], }
    ]

gbrt = GradientBoostingClassifier(random_state=42)
gbrtGS = GridSearchCV(gbrt, param_grid, cv=5, scoring='accuracy', return_train_score=True)
gbrtGS.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

gbrtGS.best_params_
bestgbrt = gbrtGS.best_estimator_
results = gbrtGS.cv_results_
dfResults = pd.DataFrame(results)

for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(mean_score, params)

evalModel(yTrue, bestgbrt.predict(preppedMagData[6]))
plotPrecRecall(yTrue, bestgbrt.decision_function(preppedMagData[6]))
plotRocCurve(yTrue, bestgbrt.decision_function(preppedMagData[6]))

#Voting Classifier with best model from each grid search
from sklearn.ensemble import VotingClassifier

logReg = LogisticRegression(random_state=42, max_iter=2000, C=0.7, penalty='l1', solver='liblinear')
knn = KNeighborsClassifier(n_neighbors=50)
RF = RandomForestClassifier(random_state=42, min_samples_leaf=2, min_samples_split=10, n_estimators=500)
SVCmod = SVC(probability=True, kernel='poly', degree=3, C=0.5, coef0=0.5)
ada = AdaBoostClassifier(algorithm="SAMME.R", random_state=42, learning_rate=0.1, n_estimators=200)
gbrt = GradientBoostingClassifier(random_state=42, learning_rate=0.1, max_depth=3, n_estimators=50, subsample=1)

voting_clf = VotingClassifier(estimators=[('lr', logReg), ('knn', knn), ('RF', RF), ('ada', ada), 
                                          ('gbrt', gbrt), ('SVCmod', SVCmod)], voting='soft')
voting_clf.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

evalModel(yTrue, voting_clf.predict(preppedMagData[6]))
plotPrecRecall(yTrue, voting_clf.decision_function(preppedMagData[6]))
plotRocCurve(yTrue, voting_clf.decision_function(preppedMagData[6]))



#models for WebAPP
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

gbrt = GradientBoostingClassifier(random_state=42, learning_rate=0.1, max_depth=3, n_estimators=50, subsample=1)
gbrt.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))

evalModel(yTrue, gbrt.predict(preppedMagData[6]))
plotPrecRecall(yTrue, gbrt.decision_function(preppedMagData[6]))
plotRocCurve(yTrue, gbrt.decision_function(preppedMagData[6]))

pickle.dump(gbrt, open('gbrt.pkl', 'wb'))
pickle.dump(full_pipeline, open('full_pipeline.pkl','wb'))

newMod = pickle.load(open('gbrt.pkl','rb'))
newPipeline = pickle.load(open('full_pipeline.pkl','rb'))


ada = AdaBoostClassifier(algorithm="SAMME.R", random_state=42, learning_rate=0.1, n_estimators=200)
ada.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))
evalModel(yTrue, ada.predict(preppedMagData[6]))
pickle.dump(ada, open('ada.pkl', 'wb'))


RF = RandomForestClassifier(random_state=42, min_samples_leaf=2, min_samples_split=10, n_estimators=500)
RF.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))
evalModel(yTrue, RF.predict(preppedMagData[6]))
pickle.dump(RF, open('RF.pkl', 'wb'))


SVC = SVC(probability=True, kernel='poly', degree=3, C=0.5, coef0=0.5)
SVC.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))
evalModel(yTrue, SVC.predict(preppedMagData[6]))
pickle.dump(SVC, open('SVC.pkl', 'wb'))


logReg = LogisticRegression(random_state=42, max_iter=2000, C=0.7, penalty='l1', solver='liblinear')
logReg.fit(preppedMagData[0], np.argmax(preppedMagData[1], axis=1))
evalModel(yTrue, logReg.predict(preppedMagData[6]))
pickle.dump(logReg, open('logReg.pkl', 'wb'))


#build, compile, and train Dense network 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

n_features = preppedMagData[0].shape[1]

np.random.seed(42)
tf.random.set_seed(42)

opt = Adam(1e-5)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[n_features]),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
history = model.fit(x=preppedMagData[0], y=preppedMagData[1], epochs=50, validation_data=(preppedMagData[2],preppedMagData[3]), shuffle=(True), batch_size = 16)
 
evalModel(yTrue, np.argmax(model.predict(preppedMagData[6]),axis=1))
plotPrecRecall(yTrue, model.predict(preppedMagData[6])[:,1])
plotRocCurve(yTrue, model.predict(preppedMagData[6])[:,1])
plotLoss()
plotAcc()

#Monte Carlo Dropout
tf.random.set_seed(42)
np.random.seed(42)

y_probs = np.stack([model(preppedMagData[6], training=True) for sample in range(100)])
y_probAve = y_probs.mean(axis=0)
y_std = y_probs.std(axis=0)

y_pred = np.argmax(y_probAve, axis=1)

#adjusting threshold
thresh=.70
z = model.predict(preppedMagData[6])
z1 = pd.DataFrame(z)
zz = np.array(preppedMagData[7])
z1['T'] = zz[:,0]
z1['F'] = zz[:,1]
z2 = z1.loc[(z1[0]>thresh) | (z1[1]>thresh)]
w = np.argmax(np.array(z2[['T','F']]), axis = 1)
v = np.argmax(np.array(z2[[0,1]]), axis = 1)

print(confusion_matrix(w,v))
print("Accuracy:", accuracy_score(w,v))
print("F1:", f1_score(w,v))
print("Precision:", precision_score(w,v))
print("Recall:", recall_score(w,v))
print("AUC:", roc_auc_score(w,v))    
    
