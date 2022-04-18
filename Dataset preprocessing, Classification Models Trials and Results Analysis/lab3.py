#!/usr/bin/env python
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from common import describe_data, test_env
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from common.classification_metrics import print_metrics

# pylint: disable-msg=E0611
#from common import describe_data, test_env


def read_data(file):
    """Return pandas dataFrame read from Excel file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    y_column = 'In university after 4 semesters'

    # Features can be excluded by adding column name to list
    drop_columns = []

    categorical_columns = [
        'Faculty',
        'Paid tuition',
        'Study load',
        'Previous school level',
        'Previous school study language',
        'Recognition',
        'Study language',
        'Foreign student'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values
    # Encode y. Naive solution
    y = np.where(y == 'No', 0, y)
    y = np.where(y == 'Yes', 1, y)
    y = y.astype(float)

    # Drop also dependent variable variable column to leave only features
    drop_columns.append(y_column)
    df = df.drop(labels=drop_columns, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # STUDENT SHALL ENCODE CATEGORICAL FEATURES
    for i in categorical_columns:
        df[i] = df[i].fillna(value='Missing')
    dum = pd.get_dummies(df, prefix_sep=":", columns=categorical_columns)
    df = dum.drop(columns=['Paid tuition:No', 'Foreign student:No', 'Previous school study language:Not known',
                           'Study load:Partial', 'Faculty:School of Engineering', 'Recognition:Missing',
                           'Study language:Estonian'])
    df = df.fillna(value=0)
    # Handle missing data. At this point only exam points should be missing
    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        describe_data.print_nan_counts(df)

    # STUDENT SHALL HANDLE MISSING VALUES

    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


# STUDENT SHALL CREATE FUNCTIONS FOR LOGISTIC REGRESSION CLASSIFIER, KNN
# CLASSIFIER, SVM CLASSIFIER, NAIVE BAYES CLASSIFIER, DECISION TREE
# CLASSIFIER AND RANDOM FOREST CLASSIFIER

def logisticR(train_X, train_y, test_X, test_y):

    # classifier
    logReg = LogisticRegression(solver='newton-cg', multi_class='multinomial')

    # Fit
    logReg.fit(train_X, train_y)

    # Predict
    predict_y = logReg.predict(test_X)

    # Print
    print_metrics(test_y, predict_y, 'logistic Regression Test Data')


def KNNneighbor(train_X, train_y, test_X, test_y):

    # classifier
    KNN = KNeighborsClassifier(n_neighbors=5)

    # Fit
    KNN.fit(train_X, train_y)

    # Predict
    predict_y = KNN.predict(test_X)

    # Print
    print_metrics(test_y, predict_y, 'KNN Test Data')


def sVc(train_X, train_y, test_X, test_y):

    # classifier
    classf = SVC(kernel='sigmoid', random_state=0,
                 gamma=.01, C=1, probability=True)

    # Fit
    classf.fit(train_X, train_y)

    # Predict
    predict_y = classf.predict(test_X)

    # Print
    print_metrics(test_y, predict_y, 'SVC Test Data')


def naiveBayes(train_X, train_y, test_X, test_y):

    # classifier
    classf = MultinomialNB()

    # Fit
    classf.fit(train_X, train_y)

    # Predict
    predict_y = classf.predict(test_X)

    # Print
    print_metrics(test_y, predict_y, 'Naive Bayes Test Data')


def decisionTree(train_X, train_y, test_X, test_y):
    # classifier
    classf = DecisionTreeClassifier()

    # Fit
    classf.fit(train_X, train_y)

    # Predict
    predict_y = classf.predict(test_X)

    # Print
    print_metrics(test_y, predict_y, 'Decision Tree Test Data')


def randomForest(train_X, train_y, test_X, test_y):
    # classifier
    classf = RandomForestClassifier(n_estimators=12)

    # Fit
    classf.fit(train_X, train_y)

    # Predict
    predict_y = classf.predict(test_X)

    # Print
    print_metrics(test_y, predict_y, 'Rand Forest Test Data')


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')
    # STUDENT SHALL CALL PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS WITH
    # FILE NAME AS ARGUMENT

    describe_data.print_overview(students)
    describe_data.print_categorical(students)

    # Filter students
    describe_data.print_overview(
        students[students['In university after 4 semesters'] == 'Yes'])
    describe_data.print_categorical(
        students[students['In university after 4 semesters'] == 'Yes'])

    students_X, students_y = preprocess_data(students)
    train_X, test_X, train_y, test_y = train_test_split(
        students_X, students_y, test_size=0.25, random_state=51)

    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    # np.set_printoptions(threshold=np.inf)

    sc = StandardScaler()
    # naive bayes complained about negative values passed by standard sc, so I used MinMax instead
    mm_scaler = MinMaxScaler()
    # only scaling non-dummy features seems to be more accurate
    train_X[:, 0:5] = mm_scaler.fit_transform(train_X[:, 0:5])
    test_X[:, 0:5] = mm_scaler.transform(test_X[:, 0:5])

    # STUDENT SHALL CALL CREATED CLASSIFIERS FUNCTIONS
    logisticR(train_X, train_y, test_X, test_y)
    KNNneighbor(train_X, train_y, test_X, test_y)
    sVc(train_X, train_y, test_X, test_y)
    naiveBayes(train_X, train_y, test_X, test_y)
    decisionTree(train_X, train_y, test_X, test_y)
    randomForest(train_X, train_y, test_X, test_y)

    print('Done')
