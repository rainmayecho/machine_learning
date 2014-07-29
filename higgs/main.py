import numpy as np
import pickle


def rank_order(v):
    v = sorted(v, key=lambda x: x[1])[::-1]
    
    final = []
    i = len(v)
    print 'Generating Rank Order...',
    for e in v:
        a, b, c = e
        n = (a, i, c)
        final.append(n)
        i -= 1
    final = sorted(final, key=lambda x: x[0])
    print 'Done.'
    return final
    

if __name__ == '__main__':
    with open('GradientBoostingClassifier.p','r') as f, open('test.csv','r') as g,\
          open('submission_gbc.csv', 'w') as out:
        clf = pickle.load(f)
        out.write('EventId,RankOrder,Class\n')

        m = {0 : 'b', 1: 's'}
        print 'Loading test dataset...',
        dataset = np.loadtxt(g, delimiter=',', skiprows=1)
        event_ids = [int(e) for e in dataset[:,0]]
        points = dataset[:,1:31]
        print 'loaded!'
        
        print 'Running classifier...',
        prob_test = clf.predict_proba(points)[:,1]
        print 'Done.'
        
        threshold = np.percentile(prob_test, 85)
        
        print 'Separating signal from noise...',
        tar_test = prob_test > threshold
        print 'Done.'

        results = zip(event_ids, prob_test, tar_test)
        
        final = rank_order(results)
        print 'Writing to file...',
        for e in final:
            a, b, c = e
            out.write('%i,%i,%s\n' %(a, b, m[c]))
        print 'Done.'
        
        
