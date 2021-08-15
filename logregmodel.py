#!/usr/bin/env python3
"""
DEPRECATED. This gave 74% constant prediction.

Using combined-annotations.csv, find a logistic regression CV model to
fit them, maximizing accuracy.

Used to produce iteration 0 from gold.
"""

import sklearn.linear_model
import sklearn.metrics

import csv
import numpy as np

MODEL = "openl3gold"
FOLDS = 3
SCORING = "accuracy"

X = []
y = []
for score, _, _, label in csv.reader(open(f"data/iterations/{MODEL}/combined-annotations.csv")):
    X.append([float(score)])
    y.append(int(label))

X = np.array(X)
y = np.array(y)
print(X)
print(y)
clf = sklearn.linear_model.LogisticRegressionCV(
    Cs=[10 ** i for i in range(-3, 3)], cv=FOLDS, scoring=SCORING, random_state=42,
)
clf.fit(X, y)

print(np.mean(y))

print(clf.predict_proba([[i / 10] for i in range(11)]))
