import numpy as np
import json
import random
import pickle
from collections import defaultdict
from collections import Counter

from nltk.corpus import stopwords

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

###################################
#        FILE MANIPULATIONS       #
###################################

def save_file(filename, data):
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)

def load_file(filename):
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    return data

def get_data(file='aggressive_dedup.json', year='2001', limit=100000000, size=10000):
    data = defaultdict(list)
    time = defaultdict(list)
    ratings = defaultdict(list)
    seen = {}
    indices = defaultdict(list)
    
    with open(file) as infile:
        i = 0
        for line in infile:
            x = json.loads(line)
            yr = x['reviewTime'][-4:]
            if yr >= year:
                if yr not in seen:
                    seen[yr] = 1
                else:
                    seen[yr] += 1
                if len(data[yr]) < size:
                    data[yr].append(x.get('reviewText'))
                    time[yr].append(x.get('reviewTime'))
                    ratings[yr].append(int(x.get('overall')))
                    indices[yr].append(i)
                else:
                    if np.random.uniform() < size/seen[yr]:
                        p = np.random.randint(size)
                        data[yr][p] = x.get('reviewText')
                        time[yr][p] = x.get('reviewTime')
                        ratings[yr][p] = int(x.get('overall'))
                        indices[yr][p] = i        
            i += 1
            if i > limit:
                break
    return data, time, ratings, seen, indices

###################################
#            RATINGS              #
###################################

def binarify(val):
    if 4 > val:
        return 0
    else:
        return 1

def simplify_ratings(data, ratings):
    keys = sorted(list(ratings.keys()))
    for key in keys:
        ratings[key] = [binarify(x) for x in ratings.get(key)]

###################################
#              NLP                #
###################################

def stopwords_filter(data):
    stop_words = set(stopwords.words('english'))
    x = sorted(list(data.keys()))
    for key in x:
        for i, m in enumerate(data[key]):
            data[key][i] = ' '.join( (w for w in m.split() if w not in stop_words) )
    return data

###################################
#         MODEL and SPLITS        #
###################################

def split_eighty_twenty(n):
    indices = np.random.permutation(n)
    threshold = int(np.floor(n*0.8))
    return indices[:threshold], indices[threshold:]

def test_train(data, ratings, savestring):
    train_idx, test_idx = split_eighty_twenty(len(data))
    save_file(savestring + '/train_idx.data', train_idx)
    save_file(savestring + '/test_idx.data', test_idx)
    X_train, X_test = np.array(data)[train_idx], np.array(data)[test_idx]
    y_train, y_test = np.array(ratings)[train_idx], np.array(ratings)[test_idx]
    return X_train, X_test, y_train, y_test

def predict_scores(data, ratings, CV, LR, i, yr, savestring, features):   
    X_train, X_test, y_train, y_test = test_train(data, ratings, savestring)

    if i == 0:
        X_train = CV.fit_transform(X_train)
        save_file(savestring + '/CV_' + str(yr) + '.data', CV)
        X_test = CV.transform(X_test)
        features[:] = list(CV.get_feature_names())
    else:
        X_train = CV.transform(X_train)
        X_test = CV.transform(X_test)

    LR.fit(X_train,y_train)
    save_file(savestring + '/LR_' + str(yr) + '.data', LR)
    y_pred = LR.predict(X_test)
    print(LR.score(X_test, y_test))
    print(yr)
    print('f1:', f1_score(y_test, y_pred, pos_label=1, average=None))
    print('recall:', recall_score(y_test, y_pred, pos_label=1, average=None))
    print('precision:', precision_score(y_test, y_pred, pos_label=1, average=None))
    print()
    return LR, CV


###################################
#             PRINTS              #
###################################

def selection_print(data, ratings):
    x = sorted(list(data.keys()))
    for each in x:
        print(each, '>>>', len(data[each]), np.round(np.array(list((Counter(ratings[each]).values())))/len(ratings[each]), 2))