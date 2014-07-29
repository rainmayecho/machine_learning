import numpy as np
import pickle
import random
import os
import math
import sklearn
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier as ABC, GradientBoostingClassifier as GBC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.cross_validation import train_test_split, cross_val_score


def _save_data(filename):
    with open('training.csv', 'r') as f:
        c = {-1: lambda x: x == 's'} ## conversion function for labels
        dataset = np.loadtxt(f, delimiter=',', skiprows=1, converters=c)
        
        X = dataset[:,1:31] ## Feature vectors
        y = dataset[:,32]   ## Labels
        w = dataset[:,31]   ## Weights
        pickle.dump([X, y, w], open(filename, 'w'))
        print X.shape, y.shape, w.shape

def _train(X, y, w):
    np.random.seed(42)
    ratio = .70
    X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
        X, y, w, test_size=1-ratio)

    ## Try to use existing classifier
    clfname ='GradientBoostingClassifier3.p'
    try:
        with open(clfname, 'r') as g:
            clf = pickle.load(g)
    except IOError:
        clf = GBC(n_estimators=50, max_depth=5, min_samples_leaf=500, max_features=10, verbose=1)
        clf.fit(X_train, y_train)
        with open(clfname, 'w') as g:
            pickle.dump(clf, g)
        print "Classifier created."

    prob_train = clf.predict_proba(X_train)[:,1]
    prob_valid = clf.predict_proba(X_valid)[:,1]

    threshold = np.percentile(prob_train, 84)

    tar_train = prob_train > threshold
    tar_valid = prob_valid > threshold

    pos_train = w_train*(y_train==1.0)*(1.0/ratio)
    neg_train = w_train*(y_train==0.0)*(1.0/ratio)
    pos_valid = w_valid*(y_valid==1.0)*(1.0/(1-ratio))
    neg_valid = w_valid*(y_valid==0.0)*(1.0/(1-ratio))

    s_train = sum(pos_train*(tar_train==1.0))
    b_train = sum(neg_train*(tar_train==1.0))
    s_valid = sum(pos_valid*(tar_valid==1.0))
    b_valid = sum(neg_valid*(tar_valid==1.0))

    ams = lambda s, b: math.sqrt(2.*((s+b+10.)*math.log(1.+s/(b+10.))-s))

    print '----------------------------------------------'
    print 'AMS score for training set (%2.f %% of data):' %(ratio*100), ams(s_train, b_train)
    print 'AMS score for validation set (%2.f %% of data):' %((1-ratio)*100), ams(s_valid, b_valid)
    print 'Classifier score:', clf.score(X_valid, y_valid)
    
    return clf

if __name__ == '__main__':
    filename = 'fullfeatures.p'
    try:
        f = open(filename,'r')
    except:
        _save_data(filename)
        f = open(filename, 'r')
    with f:
        X, y, w = pickle.load(f)
        clf = _train(X, y, w)

    
