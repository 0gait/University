from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import pickle
import numpy as np   
import Utils
import math

def calculateScores(min,max,stop_words,analyzer,allCorpus,query_set,cases,eval,lam,clinical_trials):        
    q_doc_scores = Utils.createEmptyQDocScores(query_set)
    calculateTfidf(min=min,max=max,stop_words=stop_words,analyzer=analyzer,allCorpus=allCorpus,query_set=query_set,cases=cases,eval = eval,clinical_trials=clinical_trials, q_doc_scores=q_doc_scores)
    calculateLMJM(min=min,max=max,stop_words=stop_words,analyzer=analyzer,allCorpus=allCorpus,query_set=query_set,cases=cases,eval = eval,lam=lam,clinical_trials=clinical_trials,q_doc_scores = q_doc_scores)
    print(q_doc_scores)
    pickle.dump( q_doc_scores, open( "./data/q_doc_scores_{min}_{max}_{stop_words}.p".format(min = min, max = max, stop_words=stop_words),"wb"))

def calculateTfidf(min,max,stop_words,analyzer,allCorpus,query_set,cases,eval,clinical_trials,q_doc_scores):
    for field in allCorpus.keys():
        vectorizer = TfidfVectorizer(ngram_range=(min,max), analyzer=analyzer, stop_words=stop_words)
        vectorizer.fit(allCorpus[field])
        #Compute the corpus representation
        X = vectorizer.transform(allCorpus[field])
        getScoresFromField('tfidf',X,vectorizer,query_set,cases,None,None,None,clinical_trials,q_doc_scores)
        print("DONE VSM")

def calculateLMJM(min,max,stop_words,analyzer,allCorpus,query_set,cases,eval,lam,clinical_trials,q_doc_scores):        
    for field in allCorpus.keys():
        vectorizer = CountVectorizer(ngram_range=(min,max), analyzer=analyzer, stop_words = stop_words)
        X = vectorizer.fit_transform(allCorpus[field])
        X = X.todense()
        tf = np.sum(X,axis=0)
        total_terms = np.sum(tf,axis=1)

        #Compute doc length (sum the rows of the count matrix)
        doc_length = np.sum(X,axis=1)

        #Compute the term prob in all corpus : p(t|Mc) (Divide sum of columns of Mc by the sum of all terms)
        p_t_mc = tf/total_terms
        p_t_mc = p_t_mc.transpose()

        #Compute the term prob in the document: p(t|Md) (Divide the Mc by the sum of rows of Mc)
        p_t_md = X / doc_length
        getScoresFromField('lmjm',X,vectorizer,query_set,cases,p_t_mc,p_t_md,lam,clinical_trials,q_doc_scores)
        print("DONE LMJM")



def getScoresFromField(model,X,vectorizer,query_set,cases,p_t_mc,p_t_md,lam,clinical_trials,q_doc_scores):
    for caseid in query_set:
        query = cases[caseid].title
        query_indexes = vectorizer.transform([query])

        if model == 'tfidf':
            doc_scores = 1 - pairwise_distances(X, query_indexes, metric='cosine')
        elif model == 'lmjm':
            prev_doc_scores = np.log(lam*p_t_md*query_indexes.transpose() + (1-lam)*query_indexes*p_t_mc)
            doc_scores = [[0.0] if math.isnan(score) else score.tolist()[0] for score in prev_doc_scores]

        results = pd.DataFrame(list(zip(clinical_trials.keys(), doc_scores)), columns = ['_id', 'score'])
            
        scores = results['score']
        doc_ids = results['_id']

        for i in range(len(scores)):
            doc_id = doc_ids[i]
            pair = "{q}_{ct}".format(q = caseid, ct = doc_id)
            if pair in q_doc_scores.keys():
                score = scores[i][0]
                q_doc_scores[pair].append(score)






