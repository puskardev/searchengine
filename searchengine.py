# Puskar Dev
# Student ID  : 1001516630
# Programming Assignment 1
# Date - 10/10/2020

import math
import os
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from math import log10
from math import sqrt


tokenizer = RegexpTokenizer(r'[a-zA-Z]+')       #regular expression for tokenizing

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

corpusroot = './presidential_debates'  # root directory for the corpus

vectors={}                             #tf-idf vectors for all documents
df=Counter()                           #stores document frequency for tokens

# stores term frequency for all tokens in all documents.
tf_alltok={}

for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()

    # tokenizing the document which is read
    tokens = tokenizer.tokenize(doc)

    # stop-word removal and stemming
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

    # Counter(tokens) returns a dict like structure, where the key is items in token while values are the count of corresponding items.
    tf=Counter(tokens)

    # add tf, which has tokens and their term frequency on a given doc to dict tf_alltok{} where key is filename for that doc.
    tf_alltok[filename]=tf.copy()

    # list() retruns all the unqiue elements/tokens in tf.
    df+=Counter(list(tf))

    # empty tf for the next iteration/reading next doc.
    tf.clear()

# total no. of documents in the corpus
N = len(tf_alltok)

#-----------------------------------------------------------------

# GET-IDF FUNCTION
# getidf retruns inverse document frequency (idf) for a token.
def getidf(token):
    if df[token]==0:
        return -1
    return log10(N/df[token])

#------------------------------------------------------------------------
# tfidf_weight() retruns the tf-idf weight for a token without normalization
def tfidf_weight(filename, token):
    return (1+log10(tf_alltok[filename][token]))*getidf(token)

#------------------------------------------------------------
# NORMALIZATION OF TOKENS

#posting list to store each token in the corpus with descending order of weights
post_list={}
#used for calculating lengths of documents
lengths=Counter()

# 1. calculating tf-idf vectors and lengths of documents
for filename in tf_alltok:
    #initialize the tf-idf vector for each doc
    vectors[filename]=Counter()
    sum_square=0
    for token in tf_alltok[filename]:        
        weight = tfidf_weight(filename, token)        
        vectors[filename][token]=weight        
        sum_square = sum_square + weight**2   
    
    lengths[filename]=math.sqrt(sum_square)

# 2.  normalizing the tf-idf weights in vectors{} dict.
for filename in vectors:
    for token in vectors[filename]:       
        vectors[filename][token]= vectors[filename][token] / lengths[filename]
        if token not in post_list:            
            post_list[token]=Counter()       
        post_list[token][filename]=vectors[filename][token]

#-----------------------------------------------------------------------

# GETWEIGTH function
#returns normalized tf-idf weight of a token in a document
def getweight(filename,token):
    return vectors[filename][token]

#-------------------------------------------------------------------

#QUERY function
# returns a tuple in the form of (filename of the document, score), where the document is the query final_docwer with respect to "qstring".

def query(qstring):
    #change qstring to lower case
    qstring=qstring.lower()

    query_tf={}
    q_length=0

    check=0

    temp_docs={}
    u_bound={}

    #initializing a counter for calculating cosine similarity for each token and a doc.
    cosine_score=Counter()

    for token in qstring.split():
        # stem each token using PorterStemmer
        token=stemmer.stem(token)

        # (7.2) If the token 𝑡 doesn't exist in the corpus, ignore it.
        if token not in post_list:
            continue

        # if a token has idf != 0
        if getidf(token)!=0:
            # most_common(10) returns top-10 elements in posting list based on weight. And zip assigns each value of those 10 tuples to
            temp_docs[token], weights = zip(*post_list[token].most_common(10))
        else:
            # since all value in posting-list is 0. every item on the posting list is returned.
            temp_docs[token], weights = zip(*post_list[token].most_common())

        # upper bound on token's weight
        u_bound[token]=weights[9]

        if check==1:
            # all_tok_doc is a set that will keep track of docs that have all tokens
            all_tok_doc= set(temp_docs[token]) & all_tok_doc
        else:
            all_tok_doc=set(temp_docs[token])
            check=1

        query_tf[token]= 1 + log10(qstring.count(token))
        q_length+=query_tf[token]**2

    # Normalizing length for query.
    q_length=sqrt(q_length)

    for doc in vectors:
        score=0
        for token in query_tf:
            if doc in temp_docs[token]:
                # (7.3) score calculates doc's similarity score if document is in top 10. using eqn: sim(q,d) = w(t,q)*w(t,d)
                nweight_query = query_tf[token] / q_length
                score = score +  nweight_query * post_list[token][doc]
            else:
                # (7.4) if not in top -10, calculate score using upper-bound on token's weight in doc's vector.
                nweight_query = query_tf[token] / q_length
                score = score + nweight_query * u_bound[token]

        cosine_score[doc]=score

    # find document with best similarity.
    best_doc = cosine_score.most_common(1)

    final_doc,final_weight = zip(*best_doc)

    try:
        if final_doc[0] in all_tok_doc:
            # (7.3) return the tuple with highest similarity score.
            return final_doc[0],final_weight[0]
        else:
            #(7.4) if upper bound score is greater it will return fetch more
            return "fetch more",0
    
    except UnboundLocalError:
        return "None",0
#-------------------------------------------------------------------------------------

print("\nSAMPLE RESULTS - as shown in notebook\n")

print("%.12f" % getidf("health"))
print("%.12f" % getidf("agenda"))
print("%.12f" % getidf("vector"))
print("%.12f" % getidf("reason"))
print("%.12f" % getidf("hispan"))
print("%.12f" % getidf("hispanic"))
print("\n")
print("%.12f" % getweight("2012-10-03.txt","health"))
print("%.12f" % getweight("1960-10-21.txt","reason"))
print("%.12f" % getweight("1976-10-22.txt","agenda"))
print("%.12f" % getweight("2012-10-16.txt","hispan"))
print("%.12f" % getweight("2012-10-16.txt","hispanic"))
print("\n")
print("(%s, %.12f)" % query("health insurance wall street"))
print("(%s, %.12f)" % query("particular constitutional amendment"))
print("(%s, %.12f)" % query("terror attack"))
print("(%s, %.12f)" % query("vector entropy"))

#---------------------------------------------------------------------------------------
