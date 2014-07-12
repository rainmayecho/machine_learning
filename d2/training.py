from _d2 import _calc, plot_2D
import numpy as np
import sqlite3 as lite
import matplotlib.pylab as pl
import pickle
import json, sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy.random import RandomState
from dbsetup import get_new_match

'''
-------------------------------------
Selects matches from known groups and
trains the following classifiers:

 - svm.SVC classifier
 - LogisticRegression classifier

The classifier is then presented
with a new (control) match
to predict skill group
-------------------------------------
'''

con = lite.connect('test.db')
with con:
    con.row_factory = lite.Row
    cur = con.cursor()
    measurements = []
    groups = {1: 'normal', 2: 'high', 3: 'very_high'}
    target = []
    for group in groups:
        cur.execute('''SELECT * FROM %s_matches''' %(groups[group]))
        while True:
            row = cur.fetchone()
            d = {}
            if row == None:
                break
            players = pickle.loads(str(row['players']))
            d = _calc(players)
            measurements.append(d)
            target.append(group)

vec = DictVectorizer()
X = vec.fit_transform(measurements).toarray()
Y = np.array(target)

clf = svm.SVC(gamma=1, C=1)
clf2 = LogisticRegression().fit(X,Y)
clf.fit(X,Y)

new_data = _calc(get_new_match())
X_new = vec.fit_transform(new_data).toarray()

print clf.predict(X_new)
print clf2.predict_proba(X_new)

        
