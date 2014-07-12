import json, sklearn
import numpy as np
import matplotlib.pylab as pl
import sqlite3 as lite
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy.random import RandomState
from itertools import cycle

def plot_2D(data, target, target_names):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    fig, ax = pl.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    for i, c, label in zip(target_ids, colors, target_names):
        ax.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)

    pl.legend()
    pl.show()

def _calc(players):
    d = {}
    rgpm, dgpm, rxpm, dxpm, rk, dk, rd, dd, ra, da, rlh, dlh, rg, dg = (0,)*14
    for player in players[:5]:
        rgpm += player['gold_per_min']
        rxpm += player['xp_per_min']
        rk += player['kills']
        rd += player['deaths']
        ra += player['assists']
        rlh += player['last_hits']
        rg += player['gold_spent'] + player['gold']
    for player in players[5:]:
        dgpm += player['gold_per_min']
        dxpm += player['xp_per_min']
        dk += player['kills']
        dd += player['deaths']
        da += player['assists']
        dlh += player['last_hits']
        dg += player['gold_spent'] + player['gold']

    ## Some possibly important metrics
    d['farm'] = rg + dg
    d['xpm'] = (rxpm + dxpm)/2.0
    d['last_hits'] = rlh + dlh
    d['involvement'] = (ra/(4.0*rk + 1) + da/(4.0*dk + 1))/2.0
    return d
    
    
if __name__ == '__main__':
    con = lite.connect('test.db')
    with con:
        con.row_factory = lite.Row
        cur = con.cursor()
        groups = ['normal_', 'high_', 'very_high_']
        cur.execute('''SELECT distinct * FROM normal_matches''')
        measurements = []
        for group in groups[2:]:
            cur.execute('''SELECT distinct * FROM %smatches
                        WHERE duration >= 1000''' %(group))
            while True:
                row = cur.fetchone()
                d = {}
                if row == None:
                    break
                players = pickle.loads(str(row['players']))
                duration = row['duration']
                d = _calc(players)
                measurements.append(d)

            vec = DictVectorizer()
            X = vec.fit_transform(measurements).toarray()
            clf = svm.SVC(gamma=0.001, C=100)
            pca = PCA(n_components=2, whiten=True).fit(X)
            X_pca = pca.transform(X)

            np.round(X_pca.mean(axis=0), decimals=5)
            np.round(X_pca.std(axis=0), decimals=5)
            np.round(np.corrcoef(X_pca.T), decimals=5)

            rng = RandomState(42)
            kmeans = KMeans(n_clusters=3, random_state=rng).fit(X_pca)
            np.round(kmeans.cluster_centers_, decimals=2)
            kmeans.labels_[:10]
            kmeans.labels_[-10:]

            plot_2D(X_pca, kmeans.labels_, ['c0', 'c1', 'c2'])

