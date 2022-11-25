import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

metrics = ["p10", "recall", "ap", "ndcg5", "mrr"]
metrics_no_recall = ["p10", "ap", "ndcg5", "mrr"]
c = ['aquamarine', "blue", 'xkcd:sky blue', 'tab:gray']

def buildQueryLenByPerf():

    queryLen = pickle.load(open('./data/queriesLen.p', "rb"))
    DOT = "."
    fig, axes = plt.subplots(2,2, figsize = (12,8))
    fig.suptitle('Training Set', fontsize=16, fontweight='bold', y=0.95)
    fig.subplots_adjust(hspace=0.45)

    openList = [open('./performances/metricsByQuery_parameters_48_tfidf_1_1_None.p', "rb"), open('./performances/metricsByQuery_parameters_48_tfidf_1_2_None.p', "rb"), 
                open('./performances/metricsByQuery_parameters_48_tfidf_1_1_english.p', "rb"), open('./performances/metricsByQuery_parameters_48_tfidf_1_2_english.p', "rb")]

    for count, (i, j) in enumerate(itertools.product(range(2), range(2))):
        lenList = []
        p10Perf = []
        apPerf = []
        ndcg5Perf = []
        mrrPerf = []

        dic = pickle.load(openList[count])

        for key in dic:
            lenList.append(queryLen.get(key))
            p10Perf.append(dic.get(key)[0])
            apPerf.append(dic.get(key)[2])
            ndcg5Perf.append(dic.get(key)[3])
            mrrPerf.append(dic.get(key)[4])


        ax = axes[i][j]
        ax.set_xlabel("performance")
        ax.set_ylabel("query length")
        ax.plot(p10Perf, lenList, DOT, label="p10", color="red")
        ax.plot(apPerf, lenList, DOT, label="ap", color="green")
        ax.plot(ndcg5Perf, lenList, DOT, label="ndcg5", color="blue")
        ax.plot(mrrPerf, lenList, DOT, label="mrr", color="sandybrown")

        if count == 3:
            ax.legend(loc=1)

    ax = axes[0][0]      
    ax.set_title("TFIDF w/o Stop-words and Uni")
    ax = axes[0][1]      
    ax.set_title("TFIDF w/o Stop-words and Uni/Di")
    ax = axes[1][0]      
    ax.set_title("TFIDF w/ Stop-words and Uni")
    ax = axes[1][1]      
    ax.set_title("TFIDF w/ Stop-words and Uni/Di")


def buildAvgPerf():
    #training set/tfidf
    p10_avg, recall_avg, ap_avg, ndcg5_avg, mrr_avg = pickle.load(open('./performances/metrics_avg_parameters_48_lmjm_1_1_None.p', "rb"))
    p10_avg2, recall_avg2, ap_avg2, ndcg5_avg2, mrr_avg2 = pickle.load(open('./performances/metrics_avg_parameters_48_lmjm_1_2_None.p', "rb"))
    p10_avg3, recall_avg3, ap_avg3, ndcg5_avg3, mrr_avg3 = pickle.load(open('./performances/metrics_avg_parameters_48_lmjm_1_1_english.p', "rb"))
    p10_avg4, recall_avg4, ap_avg4, ndcg5_avg4, mrr_avg4 = pickle.load(open('./performances/metrics_avg_parameters_48_lmjm_1_2_english.p', "rb"))

    #tfidf training set
    d = {"Name": ["p10", "ap", "ndcg5", "mrr"],"W/O stop-words;uni": [p10_avg, ap_avg, ndcg5_avg, mrr_avg],"W/O stop-words;uni/di": [p10_avg2, ap_avg2, ndcg5_avg2, mrr_avg2],"W/ stop-words;uni": [p10_avg3, ap_avg3, ndcg5_avg3, mrr_avg3],"W/ stop-words;uni/di": [p10_avg4, ap_avg4, ndcg5_avg4, mrr_avg4]}
    df=pd.DataFrame(d, index=["p10", "ap", "ndcg5", "mrr"])

    ###########################################
    p10_avg, recall_avg, ap_avg, ndcg5_avg, mrr_avg = pickle.load(open('./performances/metrics_avg_parameters_48_lmjm_1_1_None.p', "rb"))
    p10_avg2, recall_avg2, ap_avg2, ndcg5_avg2, mrr_avg2 = pickle.load(open('./performances/metrics_avg_parameters_48_lmjm_1_2_None.p', "rb"))
    p10_avg3, recall_avg3, ap_avg3, ndcg5_avg3, mrr_avg3 = pickle.load(open('./performances/metrics_avg_parameters_48_lmjm_1_1_english.p', "rb"))
    p10_avg4, recall_avg4, ap_avg4, ndcg5_avg4, mrr_avg4 = pickle.load(open('./performances/metrics_avg_parameters_48_lmjm_1_2_english.p', "rb"))

    #lmjm training set
    d2 = {"Name": ["p10", "ap", "ndcg5", "mrr"],"W/O stop-words;uni": [p10_avg, ap_avg, ndcg5_avg, mrr_avg],"W/O stop-words;uni/di": [p10_avg2, ap_avg2, ndcg5_avg2, mrr_avg2],"W/ stop-words;uni": [p10_avg3, ap_avg3, ndcg5_avg3, mrr_avg3],"W/ stop-words;uni/di": [p10_avg4, ap_avg4, ndcg5_avg4, mrr_avg4]}
    df2=pd.DataFrame(d2, index=["p10", "ap", "ndcg5", "mrr"])

    ################################################
    p10_avg, recall_avg, ap_avg, ndcg5_avg, mrr_avg = pickle.load(open('./performances/metrics_avg_parameters_12_tfidf_1_1_None.p', "rb"))
    p10_avg2, recall_avg2, ap_avg2, ndcg5_avg2, mrr_avg2 = pickle.load(open('./performances/metrics_avg_parameters_12_tfidf_1_2_None.p', "rb"))
    p10_avg3, recall_avg3, ap_avg3, ndcg5_avg3, mrr_avg3 = pickle.load(open('./performances/metrics_avg_parameters_12_tfidf_1_1_english.p', "rb"))
    p10_avg4, recall_avg4, ap_avg4, ndcg5_avg4, mrr_avg4 = pickle.load(open('./performances/metrics_avg_parameters_12_tfidf_1_2_english.p', "rb"))

    #tfidf test set
    d3 = {"Name": ["p10", "ap", "ndcg5", "mrr"],"W/O stop-words;uni": [p10_avg, ap_avg, ndcg5_avg, mrr_avg],"W/O stop-words;uni/di": [p10_avg2, ap_avg2, ndcg5_avg2, mrr_avg2],"W/ stop-words;uni": [p10_avg3, ap_avg3, ndcg5_avg3, mrr_avg3],"W/ stop-words;uni/di": [p10_avg4, ap_avg4, ndcg5_avg4, mrr_avg4]}
    df3=pd.DataFrame(d3, index=["p10", "ap", "ndcg5", "mrr"])

    #########################
    p10_avg, recall_avg, ap_avg, ndcg5_avg, mrr_avg = pickle.load(open('./performances/metrics_avg_parameters_12_lmjm_1_1_None.p', "rb"))
    p10_avg2, recall_avg2, ap_avg2, ndcg5_avg2, mrr_avg2 = pickle.load(open('./performances/metrics_avg_parameters_12_lmjm_1_2_None.p', "rb"))
    p10_avg3, recall_avg3, ap_avg3, ndcg5_avg3, mrr_avg3 = pickle.load(open('./performances/metrics_avg_parameters_12_lmjm_1_1_english.p', "rb"))
    p10_avg4, recall_avg4, ap_avg4, ndcg5_avg4, mrr_avg4 = pickle.load(open('./performances/metrics_avg_parameters_12_lmjm_1_2_english.p', "rb"))

    #lmjm test set
    d4 = {"Name": ["p10", "ap", "ndcg5", "mrr"],"W/O stop-words;uni": [p10_avg, ap_avg, ndcg5_avg, mrr_avg],"W/O stop-words;uni/di": [p10_avg2, ap_avg2, ndcg5_avg2, mrr_avg2],"W/ stop-words;uni": [p10_avg3, ap_avg3, ndcg5_avg3, mrr_avg3],"W/ stop-words;uni/di": [p10_avg4, ap_avg4, ndcg5_avg4, mrr_avg4]}
    df4=pd.DataFrame(d4, index=["p10", "ap", "ndcg5", "mrr"])

    fig, axes = plt.subplots(figsize=(9,19),nrows=2, ncols=2)
    fig.suptitle('Training Set', fontsize=16, fontweight='bold', y=0.95)
    fig.subplots_adjust(hspace=0.45)
    plt.figtext(0.5, 0.5, 'Test Set', ha='center', va='center', fontsize=16, fontweight='bold')

    c = ['aquamarine', "blue", 'xkcd:sky blue', 'tab:gray']
    df.plot.bar(ax = axes[0][0], legend=None, title = "TFIDF", width=0.8, figsize=(11, 8), color=c)
    df2.plot.bar(ax = axes[0][1], legend=None, title = "JM", width=0.8, color=c)
    df3.plot.bar(ax = axes[1][0], legend=None, title = "TFIDF", width=0.8, color=c)
    df4.plot.bar(ax = axes[1][1], title = "JM", width=0.8, color=c)


def buildCombinedScores():

    nonNormList = [open('./metrics/metrics_avg_parameters_1_1_None_0.6_False.p', "rb"), open('./metrics/metrics_avg_parameters_1_2_None_0.6_False.p', "rb"), 
                open('./metrics/metrics_avg_parameters_1_1_english_0.6_False.p', "rb"), open('./metrics/metrics_avg_parameters_1_2_english_0.6_False.p', "rb")]

    normList = [open('./metrics/metrics_avg_parameters_1_1_None_0.6_True.p', "rb"), open('./metrics/metrics_avg_parameters_1_2_None_0.6_True.p', "rb"), 
                open('./metrics/metrics_avg_parameters_1_1_english_0.6_True.p', "rb"), open('./metrics/metrics_avg_parameters_1_2_english_0.6_True.p', "rb")]

    allLists = [nonNormList, normList]
    df = None

    for index in range(2):
        auxList = allLists[index]

        list1 = pickle.load(auxList[0])
        del list1[1]
        list2 = pickle.load(auxList[1])
        del list2[1]
        list3 = pickle.load(auxList[2])
        del list3[1]
        list4 = pickle.load(auxList[3])
        del list4[1]

        d = {"Name": metrics_no_recall,"W/O stop-words;uni": list1,"W/O stop-words;uni/di": list2,"W/ stop-words;uni": list3,"W/ stop-words;uni/di": list4}
        df=pd.DataFrame(d, index=metrics_no_recall)
        if index == 0:
            df.plot.barh(legend=None, title = "Non Normalization", width=0.8, color=c)
        else:
            ax = df.plot.barh(title = "Normalization", width=0.8, color=c)
            ax.legend(loc='upper right', prop={'size': 7})


    """ nonNormList = [open('./metrics/metrics_avg_parameters_1_1_None_0.6_False.p', "rb"), open('./metrics/metrics_avg_parameters_1_2_None_0.6_False.p', "rb"), 
                open('./metrics/metrics_avg_parameters_1_1_english_0.6_False.p', "rb"), open('./metrics/metrics_avg_parameters_1_2_english_0.6_False.p', "rb")]

    normList = [open('./metrics/metrics_avg_parameters_1_1_None_0.6_True.p', "rb"), open('./metrics/metrics_avg_parameters_1_2_None_0.6_True.p', "rb"), 
                open('./metrics/metrics_avg_parameters_1_1_english_0.6_True.p', "rb"), open('./metrics/metrics_avg_parameters_1_2_english_0.6_True.p', "rb")]

    allLists = [nonNormList, normList]
    fig, axes = plt.subplots(1,2, figsize = (12,8))
    fig.suptitle('Combined Document Scores', fontsize=16, fontweight='bold', y=0.95)
    fig.subplots_adjust(hspace=0.45)

    for count, (i, j) in enumerate(itertools.product(range(1), range(2))):
        p10 = []
        ap = []
        ndcg5 = []
        mrr = []

        auxList = allLists[count]

        for index in range(len(auxList)):
            l = pickle.load(auxList[index])
            p10.append(l[0])
            ap.append(l[2])
            ndcg5.append(l[3])
            mrr.append(l[4])

        ax = axes[i][j]
        ax.set_xlabel("Score")
        ax.plot.barh(p10, label="p10", color="red")
        ax.plot.barh(ap, label="ap", color="green")
        ax.plot.barh(ndcg5, label="ndcg5", color="blue")
        ax.plot.barh(mrr, label="mrr", color="sandybrown")

        if count == 1:
            ax.legend(loc=1)

    ax = axes[0][0]      
    ax.set_title("Non Normalization")
    ax = axes[0][1]      
    ax.set_title("Normalization") """
        
