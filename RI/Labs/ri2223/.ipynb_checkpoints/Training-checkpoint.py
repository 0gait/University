import pickle
from sklearn.linear_model import LogisticRegression
from Utils import normalizeScores
from sklearn.model_selection import KFold
from Evaluation import getMetrics
import numpy as np   

def getYTrain(q_doc_scores,q_revDocs):
    y_train = [] 
    for key in q_doc_scores.keys():
        [q,doc] = key.split("_")
        if doc in q_revDocs[q]:
            y_train.append(1)
        else:
            y_train.append(0)

    return y_train

def combineScores(q_doc_scores,q_revDocs, normalize,min,max,stop_words,c,class_weight):
    scores = list(q_doc_scores.values())
    if normalize:
        scores = normalizeScores(scores)

    y_train = getYTrain(q_doc_scores,q_revDocs)
    clf = LogisticRegression(random_state=0,  max_iter= 500, class_weight = class_weight, C = c).fit(scores, y_train)  

    coef = clf.coef_
    for i, key in enumerate(q_doc_scores.keys()):
        result = np.dot(coef, scores[i])
        q_doc_scores[key] = result[0]

    pickle.dump( q_doc_scores, open( "./data/q_doc_scores_combined_{min}_{max}_{stop_words}_{normalized}.p".format(min = min, max = max, stop_words=stop_words,normalized = normalize), "wb" ) )

def getBestC(q_doc_scores,q_revDocs, normalize,min,max,stop_words,class_weight,c_list,lam, eval, n_splits): 
    norm_scores = list(q_doc_scores.values())        
    if normalize:
        norm_scores = normalizeScores(norm_scores)
    
    y_train = np.array(getYTrain(q_doc_scores,q_revDocs))

    kf = KFold(n_splits=n_splits,shuffle=False, random_state=None)

    weight_c_metricsavg = {}
    for weight in class_weight:
        for c in c_list:
            p10_total,recall_total,ap_total,ndcg5_total,mrr_total = 0,0,0,0,0
            for train_index, test_index in kf.split(norm_scores):
                X_train, X_test = norm_scores[train_index], norm_scores[test_index]
                y_train_fold, y_test = y_train[train_index], y_train[test_index]
                clf = LogisticRegression(random_state=0,C = c,class_weight = weight).fit(X_train, y_train_fold) 
                coef = clf.coef_
                q_d_scores = {}
                q_set = []
                counter = 0
                for i, key in enumerate(q_doc_scores.keys()):
                    q = key.split("_")[0]
                    if i in test_index:
                        if not q in q_set: 
                            q_set.append(q)
                        result = np.dot(coef, X_test[counter])
                        q_d_scores[key] = result[0]
                        counter+=1
                
                [p10_avg, recall_avg,ap_avg,ndcg5_avg,mrr_avg] = getMetrics(q_doc_scores = q_d_scores,min = min,max = max,stop_words = stop_words,lam=lam,query_set = q_set,eval = eval,normalized=normalized,c = 1.0,class_weight = weight)
                p10_total += p10_avg
                recall_total += recall_avg
                ap_total += ap_avg
                ndcg5_total += ndcg5_avg
                mrr_total += mrr_avg
            [p10_avg, recall_avg,ap_avg,ndcg5_avg,mrr_avg] = [p10_total/n_splits,recall_total/n_splits,ap_total/n_splits,ndcg5_total/n_splits,mrr_total/n_splits]
            weight_c_metricsavg["{weight}_{c}".format(weight = weight,c=c)] = [p10_avg, recall_avg,ap_avg,ndcg5_avg,mrr_avg]
    pickle.dump( weight_c_metricsavg, open( "./weight_c_metricsavg.p", "wb" ) )

def filterBestCombinationCW_C(weight_c_metricsavg):
    p10_values = [i[0] for i in weight_c_metricsavg.values()]
    p10_values.sort(reverse=True)
    best_combinations = []
    for key in  weight_c_metricsavg.keys():
        if weight_c_metricsavg[key][0] == p10_values[0]:
            best_combinations.append(key)

    return best_combinations


