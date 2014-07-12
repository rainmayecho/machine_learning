import sklearn
import numpy as np
import matplotlib.pylab as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from itertools import cycle
from matplotlib.colors import ListedColormap
from sklearn.lda import LDA


def plot_2D(data, target, target_names):
    '''
    -------------------------------------------
    Plots two features for a data set
    or after PCA dimensionality reduction.
    -------------------------------------------
    '''
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    fig, ax = plt.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    for i, c, label in zip(target_ids, colors, target_names):
        ax.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)

    plt.legend()
    plt.show()

def parse(line, mode):
    '''
    -------------------------------------------
    Takes a single line and mode:
        line - string associated with data point
        mode - training (0) or classifying (1)
        
    Parses and returns a tuple:
    (t, d)
        t - the integer target (0 or 1)
        d - the dictionary of features
    -------------------------------------------
    '''
    features = line.split()
    classmap = {'-1': 0, '+1': 1}
    d = {}
    try:
        t = classmap[features[1-mode]]
    except KeyError:
        t = features[1-mode]
    for feature in features[2-mode:]:
        kvp = feature.split(':')
        d[int(kvp[0])] = float(kvp[1])
    return  t, d


f = open('input00.txt', 'r')
g = open ('output00.txt','w')
with f, g:
    data = [line for line in f]

    ## N - the number of training set points
    ## M - the dimension of the feature vector
    N, M = (int(e) for e in data.pop(0).split())

    target, observed = [], []
    classmap = {'-1': 0, '+1': 1, 0: '-1', 1: '+1'}

    ## Loading set of feature vectors
    ## and the associated target vector
    for n in xrange(N):
        line = data.pop(0)
        t, d = parse(line, 0) ## mode = 0, training
        observed.append(d)
        target.append(t)
        
            
    vec = DictVectorizer()
    X = vec.fit_transform(observed).toarray()
    Y = np.array(target)
    clf = LDA()
    clf.fit(X,Y)

    ## q - the number of classifications requested
    q = int(data.pop(0))
    for x in xrange(q):
        t, d = parse(data.pop(0), 1) ## mode = 1, predicting
        X_new = vec.fit_transform(d).toarray()
        g.write('%s %s\n' %(t, classmap[clf.predict(X_new)[0]]))
        print '%s %s' %(t, classmap[clf.predict(X_new)[0]])

