# -*- coding: utf-8 -*-

## import modules
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import roc_curve,auc
import matplotlib.patches as patches
import numpy as np
from scipy import interp
from sklearn.decomposition import PCA
import pandas as pd

## parameters
plot = False
neg_control = False

## making an empty pandas data frame and appending to it

d = {'1_algorithm': ['lasso', 'random forest'], '2_AUC': [0, 0]}
df = pd.DataFrame(data=d)

# change specific index of data frame using row and column names
#df.loc[0,'1_algorithm'] = 'changed'

## simple pandas commands list

# select columns by name
# dataset['sepal-length']
# select rows and columns
# dataset.iloc[0:3,1:5]
# get head of dataset
# print(dataset.head(20))
# get summary stats
# print(dataset.describe())
# get unique entries in specified column
# dataset['class'].unique()
# select by column entry from list
# dataset = dataset.loc[dataset['class'].isin(stuff)]
# convert character strings to binary
# sampleDF['housing'] = sampleDF['housing'].map({'yes': 1, 'no': 0})
# dimensions of data
# dataset.shape

## import data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

## get a ROC curve from the k-fold cross validation
# https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv

# select two groups only (simpler)
#stuff = ['Iris-setosa','Iris-versicolor']
stuff = ['Iris-setosa','Iris-virginica']
#stuff = ['Iris-versicolor','Iris-virginica']
dataset = dataset.loc[dataset['class'].isin(stuff)]

# make binary
#dataset['class'] = dataset['class'].map({'Iris-setosa': 1, 'Iris-versicolor': 0})
dataset['class'] = dataset['class'].map({'Iris-setosa': 1, 'Iris-virginica': 0})
#dataset['class'] = dataset['class'].map({'Iris-versicolor': 1, 'Iris-virginica': 0})

if neg_control == True:
    # shuffle outcome variable to test null model
    dataset['class'] = np.random.permutation(dataset['class'].values)

# split the data into response and explanatory
X = dataset.loc[:, dataset.columns != 'class']
Y = dataset.loc[:,'class']

### FIRST: more complex method of cross validation

# set up models and parameters
clf = RandomForestClassifier()
log = LogisticRegression(penalty='l1', solver='liblinear')

# make list of models to loop over
models = [clf,log]

# choose cross validation method
cv = model_selection.StratifiedKFold(n_splits=10,shuffle=False)

#
if plot:
    # set up plot
    fig1 = plt.figure(figsize=[12,12])
    font = {'family' : 'normal',
        'size'   : 20}
    matplotlib.rc('font', **font)
    
# do k fold cross validation loop
tprs = []
aucs = []
aucs2 = []

#
mean_fpr = np.linspace(0,1,100)
for train,test in cv.split(X,Y):
    # loop over each type of model
    i = 1
    for model in models:
        prediction = models[i-1].fit(X.iloc[train],Y.iloc[train]).predict_proba(X.iloc[test])
        # calculate tpr and fpr
        fpr, tpr, t = roc_curve(Y.values[test], prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        # calculate AUC
        roc_auc = auc(fpr, tpr)
        # append the AUC
        if i == 1: # we are on random forest
            aucs.append(roc_auc)
        elif i == 2: # we are on logistic regression
            aucs2.append(roc_auc)
        i = i + 1    
    # can plot individual ROC curves, but probably better to plot just the mean
    # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    
performance_randomforest = np.mean(aucs)
performance_lasso = np.mean(aucs2)

# add the AUCs to a results data frame
df.loc[0,'2_AUC'] = performance_randomforest
df.loc[1,'2_AUC'] = performance_lasso

# print results
print(df)

#
if plot:
    # do plot
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
    
    # set up plot details
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.show()

### SECOND: simpler way of cross validation
    
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# set up model
log = LogisticRegression(penalty='l1', solver='liblinear')
forest = RandomForestClassifier()
ada = AdaBoostClassifier()
gtree = GradientBoostingClassifier()

# set up cross validation method
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=50)

# do cross validation
log_scores = cross_val_score(log, X, Y, cv=rskf, scoring='roc_auc')
forest_scores = cross_val_score(forest, X, Y, cv=rskf, scoring='roc_auc')
ada_scores = cross_val_score(ada, X, Y, cv=rskf, scoring='roc_auc')
gtree_scores = cross_val_score(gtree, X, Y, cv=rskf, scoring='roc_auc')

# collect results
print("AUC: %0.2f (+/- %0.2f)" % (log_scores.mean(), log_scores.std() * 2))
print("AUC: %0.2f (+/- %0.2f)" % (forest_scores.mean(), forest_scores.std() * 2))
print("AUC: %0.2f (+/- %0.2f)" % (ada_scores.mean(), ada_scores.std() * 2))
print("AUC: %0.2f (+/- %0.2f)" % (gtree_scores.mean(), gtree_scores.std() * 2))




    