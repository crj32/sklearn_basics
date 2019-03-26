#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## simple sklearn example with nested cross-validation and model hyperparameter
## tuning on the iris dataset

## import modules
import pandas
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

## parameters
plot = False
neg_control = False

### get data and reformat

## import data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# select two groups only (simpler)
#stuff = ['Iris-setosa','Iris-versicolor']
#stuff = ['Iris-setosa','Iris-virginica']
stuff = ['Iris-versicolor','Iris-virginica']
dataset = dataset.loc[dataset['class'].isin(stuff)]

# make binary
#dataset['class'] = dataset['class'].map({'Iris-setosa': 1, 'Iris-versicolor': 0})
#dataset['class'] = dataset['class'].map({'Iris-setosa': 1, 'Iris-virginica': 0})
dataset['class'] = dataset['class'].map({'Iris-versicolor': 1, 'Iris-virginica': 0})

# optionally do a negative control
if neg_control == True:
    # shuffle outcome variable to test null model
    dataset['class'] = np.random.permutation(dataset['class'].values)

# add noise features from Gaussian distribution to the iris data
x = np.zeros(100)
for i in range(0, 1000):
    col_name = "char_" + str(i)
    dataset[col_name] = np.random.normal(0, 1, dataset.shape[0])

# split the data into response and explanatory
X = dataset.loc[:, dataset.columns != 'class']
Y = dataset.loc[:,'class']

### nested cross validation with hyperparameter tuning
    
## 1. set up models
log = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
forest = RandomForestClassifier()
ada = AdaBoostClassifier()
gtree = GradientBoostingClassifier()

## 2. create hyperparameter spaces

# log model hyperparameter
penalty = ['l1', 'l2']
C = np.logspace(-4, 4, 20)
log_hyper = dict(C=C, penalty=penalty)

# random forest model hyperparameters 

## 3. set up cross validation method
inner_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

## 4. set up inner cross validation parameter tuning, can use this to get AUC
log.model = GridSearchCV(estimator=log, param_grid=log_hyper, cv=inner_cv, scoring='roc_auc')
log.model.fit(X, Y)
print(log.model.best_score_)

## 5. do the nested cross validation
log_scores = cross_val_score(log.model, X, Y, scoring='roc_auc', cv=outer_cv)
print("AUC: %0.2f (+/- %0.2f)" % (log_scores.mean(), log_scores.std() * 2))
