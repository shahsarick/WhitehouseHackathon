# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 13:15:24 2017

@author: Rohan
"""

from bs4 import BeautifulSoup # For HTML parsing
import urllib2 # Website connections
import re # Regular expressions
from time import sleep # To prevent overwhelming the server between connections
from collections import Counter # Keep track of our term counts
from nltk.corpus import stopwords # Filter out stopwords, such as 'the', 'or', 'and'
import pandas as pd # For converting results to a dataframe and bar chart plots
import nltk
# Imports
# Basics
from __future__ import print_function, division
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
# gensim
from gensim import corpora, models, similarities, matutils
# sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics.pairwise as smp

from sklearn.decomposition import NMF

# logging for gensim (set to INFO)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

d1 = open('debate_1_1.txt', 'r')

x= d1.read()

import re 
x1 =re.sub("[\(\[].*?[\)\]]", "", x)
x2 = x1.replace('\n','')
x3 = x2.replace('\'','')
text = x3

lines = (line.strip() for line in text.splitlines()) # break into lines
    
        
        
chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) # break multi-headlines into a line each
    
def chunk_space(chunk):
    chunk_out = chunk + ' ' # Need to fix spacing issue
    return chunk_out  
        
    
text = ''.join(chunk_space(chunk) for chunk in chunks if chunk).encode('utf-8') # Get rid of all blank lines and ends of line
        
        
    # Now clean out all of the unicode junk (this line works great!!!)
        
try:
    text = text.decode('unicode_escape').encode('ascii', 'ignore') # Need this as some websites aren't formatted
except:                                                            # in a way that this works, can occasionally throw
   return                                                         # an exception
       
        
    text = re.sub("[^a-zA-Z.]"," ", text)  # Now get rid of any terms that aren't words (include 3 for d3.js)
                                                # Also include + for C++
        
       
text = text.lower().split()  # Go to lower case and split them apart
        
text = ' '.join(text)      
 # Filter out any stop words
    #text = [w for w in text ]
        
        
        
    #text = list(set(text)) # Last, just get the set of these. Ignore counts (we are just looking at whether a term existed
                            # or not on the website)
        
text =' '.join(text)

text = text.decode('unicode_escape').encode('ascii', 'ignore')



from __future__ import print_function, division
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

# gensim
from gensim import corpora, models, similarities, matutils
# sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics.pairwise as smp

from sklearn.decomposition import NMF

# logging for gensim (set to INFO)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import scipy as sp
import numpy as np
from pandas import *

text = [line.decode('utf-8').strip() for line in text]

tfidf = TfidfVectorizer(token_pattern="\\b[a-zA-Z][a-zA-Z]+\\b", 
                        min_df=10)

tfidf_vecs = tfidf.fit_transform(text.split('.'))



tfidf_corpus = matutils.Sparse2Corpus(tfidf_vecs.transpose())

# Row indices
id2word = dict((v, k) for k, v in tfidf.vocabulary_.items())

# This is a hack for Python 3!
id2word = corpora.Dictionary.from_corpus(tfidf_corpus, 
                                         id2word=id2word)
                                         
                                         
lsi = models.LsiModel(tfidf_corpus, id2word=id2word, num_topics=50)
lsi_corpus = lsi[tfidf_corpus]

ng_lsi = matutils.corpus2dense(lsi_corpus, num_terms=50).transpose()
ng_lsi.shape                                        


from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

K = range(1,50)
KM = [KMeans(n_clusters=k).fit(ng_lsi) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(ng_lsi, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/ng_lsi.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(ng_lsi)**2)/ng_lsi.shape[0]
bss = tss-wcss

kIdx = 10-1

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')



from sklearn.metrics import silhouette_score
import seaborn as sns
s = []
for n_clusters in range(2,30):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(ng_lsi)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    s.append(silhouette_score(ng_lsi, labels, metric='euclidean'))

plt.plot(s)
plt.ylabel("Silouette")
plt.xlabel("k")
plt.title("Silouette for K-means cell's behaviour")
sns.despine()


# Create KMeans
kmeans = KMeans(n_clusters=10, random_state= 100)

# Cluster
ng_lsi_clusters = kmeans.fit_predict(ng_lsi)

ng_lsi_clusters

print(ng_lsi_clusters[0:5])
g[0:5]

np.bincount(ng_lsi_clusters)



text2 = text.split('.')
text2[1]

df = pd.DataFrame(text2)

df['clusterno'] = ng_lsi_clusters


df2 = df.loc[df['clusterno'] == 10]

df2[0].str.cat(sep=', ')

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

from sklearn.metrics.pairwise import cosine_similarity

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=5, dissimilarity="precomputed", random_state=1)

distances = cosine_similarity(tfidf_vecs)

pos = mds.fit_transform(distances)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

import seaborn as sns

cluster_names = {0: 'Donald bitching about the current state of things (Nafta, Obama, war etc)', 
                 1: 'Donald talking about Hillarys terrible experience and mistakes of the past', 
                 2: 'Post debate interviews with callers', 
                 3: 'Life story and past experiences of both Hillary and Trump', 
                 4: 'Creating American jobs, Manufacturing and the Rise of Crime',
                 5: 'Trumps disbelief in Global warming, Clintons mistakes in Benghazi and Libya', 
                 6: 'Thank You and ending statements', 
                 7: 'Donald on the Affordable Care Act, illegal immigrants, American Debt, Inner Cities', 
                 8: 'Clinton on Donalds housing mishaps,refusal to show his taxes, and Gun Control. Donald on Clintons e-mails, Nuclear energy', 
                 9: 'Moderators constantly asking candidates to not go over their designated time'
                  
                 }

df.clusterno = df.clusterno.astype(int)
clusters = []
clusters = df.clusterno

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 



groups = df.groupby('label')



fig, ax = plt.subplots(figsize=(30, 14))


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 



for name, group in groups:
     
     ax.plot(group.x, group.y, marker='o', linestyle=' ', ms=5, 
         label=cluster_names[name],color = tableau20[name],
         mec='none')
     ax.set_aspect('auto')
     ax.tick_params(\
         axis= 'x',          
         which='both',      
         bottom='off',     
         top='off',         
         labelbottom='off')
     ax.tick_params(\
         axis= 'y',         
         which='both',      
         left='off',      
         top='off',       
         labelleft='off')

ax.legend(numpoints=3) 
plt.show() 
