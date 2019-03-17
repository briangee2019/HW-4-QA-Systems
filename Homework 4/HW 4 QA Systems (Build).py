#!/usr/bin/env python
# coding: utf-8

# # Preparation: Libraries/Packages and Data

# In[1]:


# Import Libraries
import os
import csv
import numpy as np
import pandas as pd
import pprint
#pickle
import pickle
# regex
import re
# nltk
import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer,word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.linear_model import LogisticRegression # (setting multi_class=”multinomial”)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# textblob
from textblob import TextBlob
# itertools
import itertools
from itertools import islice
# spacy
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()



# In[ ]:


# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

# Use NLTK's PorterStemmer
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return words

# Define take function for random selection
def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


# # Document Retrieval:

# In[5]:


# Get full list of filenames
os.chdir("Data")
files = os.listdir()

# Get dictionary {matrix index: filename}
doc_index_dict = {}
i=0
for file in files:
    doc_index_dict[i] = file
    i+=1

# Get dictionary {doc_name, text}
doc_text_dict = {}
for file in files:
    with open(file,errors='ignore') as f:
        item = f.read()
    doc_text_dict[file] = item


# In[ ]:


os.chdir("..")
os.chdir("Saved Objects")
#pickle the dictionaries we made and save in "Saved Objects folder"
pickle.dump(doc_index_dict,open('doc_index_dict.pickle','wb'))
pickle.dump(doc_text_dict,open('doc_text_dict.pickle','wb'))


# In[13]:


# Create list of all texts (order matters! for later)
texts = [] #put all document texts in list for vectorizers
for tuple in take(730,doc_text_dict.items()):
    texts.append(tuple[1])
pickle.dump(texts,open("texts.pickle",'wb'))


# ## Score Documents for Retrieval
# Keyword, Document, Subset of Document
# http://jonathansoma.com/lede/algorithms-2017/classes/more-text-analysis/counting-and-stemming/

# In[16]:


# Make Tf-idf vectorizer for documents
tfidf_vec = TfidfVectorizer(tokenizer = textblob_tokenizer,
                      use_idf = False,norm='l2')
                      #vocabulary = stem_kw) # L - TWO (cosine similarity)

# Say hey vectorizer, please read our stuff
matrix = tfidf_vec.fit_transform(texts)

# And make a dataframe out of it
tfidf_docs = pd.DataFrame(matrix.toarray(), columns=tfidf_vec.get_feature_names())
pickle.dump(tfidf_docs,open("tfidf_docs.pickle",'wb'))
