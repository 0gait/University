import numpy as np   
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


def getMetrics(q_doc_scores,min,max,stop_words,lam,query_set,eval,normalized,c,class_weight):
    p10_total,recall_total,ap_total,ndcg5_total,mrr_total = 0,0,0,0,0
    metricsByQuery = {}
    setLen = len(query_set)
    avg_precision_11point = np.zeros(11)
    q_scores = {case: [] for case in query_set}
    q_docs = {case: [] for case in query_set}
    for key in q_doc_scores:
        [q,doc] = key.split("_")
        if (q in query_set):
            q_scores[q].append(q_doc_scores[key])
            q_docs[q].append(doc)

    for caseid in q_scores:
        results = pd.DataFrame(list(zip(q_docs[caseid], q_scores[caseid])), columns = ['_id', 'score'])
        results_ord = results.sort_values(by=['score'], ascending = False)
        [p10, recall, ap, ndcg5, mrr] = eval.eval(results_ord, caseid)
        metricsByQuery[caseid] = [p10, recall, ap, ndcg5, mrr]
        [precision_11point, recall_11point, total_relv_ret] = eval.evalPR(results_ord, caseid)
        if (np.shape(recall_11point) != (0,)):
            avg_precision_11point = avg_precision_11point + precision_11point
        p10_total += p10
        recall_total += recall
        ap_total += ap
        ndcg5_total += ndcg5
        mrr_total += mrr

    [p10_avg, recall_avg,ap_avg,ndcg5_avg,mrr_avg] = [p10_total/setLen,recall_total/setLen,ap_total/setLen,ndcg5_total/setLen,mrr_total/setLen]
    pickle.dump( [p10_avg, recall_avg, ap_avg, ndcg5_avg, mrr_avg],
    open( "./metrics/metrics_avg_parameters_{query_set}_{min}_{max}_{stop_words}_{lam}_{normalized}_c{c}_class_weight_{class_weight}.p".format(query_set = setLen,min = min, max = max, stop_words = stop_words,lam = lam,normalized = normalized,c = c,class_weight = class_weight), "wb" ))
    pickle.dump( metricsByQuery,
    open( "./metrics/metricsByQuery_parameters_{query_set}_{min}_{max}_{stop_words}_{lam}_{normalized}_c{c}_class_weight_{class_weight}.p".format(query_set = setLen,min = min, max = max, stop_words = stop_words,lam = lam,normalized = normalized,c = c,class_weight = class_weight), "wb" ))
    print("-----------------------------------------------")
    print("Avg P10: " + str(p10_avg))
    print("Avg recall: " + str(recall_avg))
    print("Avg ap: " + str(ap_avg))
    print("Avg ndcg5: " + str(ndcg5_avg))
    print("Avg mrr: " + str(mrr_avg))

    return [p10_avg, recall_avg,ap_avg,ndcg5_avg,mrr_avg]