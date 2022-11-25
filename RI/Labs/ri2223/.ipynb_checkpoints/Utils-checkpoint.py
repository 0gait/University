import pickle
from sklearn.preprocessing import StandardScaler

def isMale(gender):
    vocab_male = ["male", "boy" , "guy", "man", "masculine individual"]
    gender = gender.lower()

    for word in vocab_male:
        value = word in gender
    return value

def isFemale(gender):
    vocab_female = ["female", "girl", "woman", "feminine individual"]
    gender = gender.lower()

    for word in vocab_female:
        value = word in gender
    return value


def isUnder30(min_age,max_age):
    return min_age < "30" or max_age < "30"

def isInRange30_60(min_age,max_age):
    return (min_age >= "30" and min_age <= "60") or (max_age >= "30" and max_age <= "60")

def isOver60(min_age,max_age):
    return min_age > "60" or max_age > "60"

def assignLabels(q_revDocs,clinical_trials):
    labels = {}
    for q,docs in q_revDocs.items():
        allMale = True
        allFemale = True
        under30 = True
        range30_60 = True
        over60 = True
        for doc_id in docs:
            ct = clinical_trials[doc_id]
            if not isMale(ct.gender):allMale= False
            if not isFemale(ct.gender): allFemale = False
            if not isUnder30(ct.minimum_age,ct.maximum_age): under30 = False
            if not isInRange30_60(ct.minimum_age,ct.maximum_age): range30_60 = False
            if not isOver60(ct.minimum_age,ct.maximum_age): over60 = False


        if allMale and under30: labels[q] = 7
        elif allMale and range30_60: labels[q] = 2
        elif allMale and over60: labels[q] = 3
        elif allFemale and under30: labels[q] = 4
        elif allFemale and range30_60: labels[q] = 5
        elif allFemale and over60: labels[q] = 6
        elif under30: labels[q] = 7
        elif range30_60: labels[q] = 8
        elif over60: labels[q] = 9
        else : labels[q] = 10
    
    return labels

def createEmptyQDocScores(query_set):
    q_docs = pickle.load( open( "./data/q_docs.p", "rb" ) )
    q_doc_scores = {}

    for q in query_set:
        for ct in q_docs[q]:
            q_doc_scores["{q}_{ct}".format(q = q, ct = ct)] = []

    return q_doc_scores

def split_sets(cases,q_label):
    q_ids = list(cases.keys());
    labels = [q_label[q] for q in q_ids] 
    [training_set,test_set] = train_test_split(q_ids, test_size=0.2, train_size = 0.8, random_state=None, shuffle = True, stratify = labels)
    pickle.dump( training_set, open( "./sets/training_set.p", "wb" ) )
    pickle.dump( test_set, open( "./sets/test_set.p", "wb" ) )

def get_q_rev_docs(cases, eval):
    q_revDocs = {case: [] for case in cases.keys()}

    for i, rel in enumerate(eval.relevance_judgments['rel']):
        query = eval.relevance_judgments['query_id'][i]
        doc = eval.relevance_judgments['docid'][i]
        if rel >= 1:
            q_revDocs[str(query)].append(str(doc))

    pickle.dump( q_revDocs, open( "./data/q_revDocs.p", "wb" ) )

def get_q_docs(cases, eval):
    q_docs = {case: [] for case in cases.keys()}
    y_train = []
    for i, rel in enumerate(eval.relevance_judgments['rel']):
        query = eval.relevance_judgments['query_id'][i]
        doc = eval.relevance_judgments['docid'][i]
        q_docs[str(query)].append(str(doc))

    pickle.dump( q_docs, open( "./data/q_docs.p", "wb" ) )

def normalizeScores(scores):
    scaler = StandardScaler()
    scaled_scores = scaler.fit_transform(scores)
    return scaled_scores