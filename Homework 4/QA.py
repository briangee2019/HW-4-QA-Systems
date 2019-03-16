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


# In[2]:


# Working Directories!
# Start in root of submission folder***


# # Question Analysis

# ## Determine Question Type: 
# Person, Location, Quantity, etc.
# 
# Steps: 
# * Read all articles into dictionary: {doc_name: content, . . . }
# * Create labeled training data to train classfier (multi-level response)
# 

# In[3]:


# Get question input: (string)
question = input("What is your question?\n")


# In[54]:


#question = 'Who is the CEO of Google?'
#question = 'What percent drop or increase in unemployment is associated with GDP?'
#question = 'Which companies went bankrupt in 2014?'
#question = 'What affects GDP?'
# Keyword heuristic: all cardinal numbers, nouns, adjectives, adverbs
keywords = [tuples[0] for tuples in pos_tag(word_tokenize(question))
            if tuples[1][0:2] in ['NN','RB','JJ','CD']] # removed VB
# drop wh-words and stopwords
wh = ['which','what','who','where','when','why']
for kw in keywords:
    if kw.lower() in wh:
        keywords.remove(kw)
for kw in keywords:
    if kw.lower() in stopwords.words('english'):
        keywords.remove(kw)


# In[55]:


# Classify question:
# Which companies 
# What affects GDP? What percentage drop
# Who is the CEO of company X? 
if 'who' in question.lower():
    qtype = 'PERSON'
if 'which' in question.lower():
    qtype = 'ORG'
if 'what' in question.lower():
    qtype = 'NN'
if 'percent' in question.lower() or '%' in question.lower():
    qtype = 'PERCENT'


# ## Extract Keywords (Query Generation):
# Entities, Names

# In[56]:


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


# In[57]:


# Stem keywords
stem_kw = []
for i in keywords:
    stem_kw += (textblob_tokenizer(i))
# compani gives useless info, too common. Remove it, since we already have qtype
#if 'compani' in stem_kw:
#    stem_kw.remove('compani')


# # Document Retrieval:

# In[8]:


os.chdir("Saved Objects")

# Load doc_index_dict from pickle file
doc_index_dict = pickle.load(open( "doc_index_dict.pickle","rb" ))
# Load doc_text_dict from pickle file
doc_text_dict = pickle.load( open( "doc_text_dict.pickle","rb" ))
# Load list of article texts
texts = pickle.load( open( "texts.pickle", "rb" ))


# ## Score Documents for Retrieval
# Keyword, Document, Subset of Document
# 
# http://jonathansoma.com/lede/algorithms-2017/classes/more-text-analysis/counting-and-stemming/

# In[9]:


# Load Tf-idf matrix for documents
tfidf_docs = pickle.load( open( "tfidf_docs.pickle", "rb" ))


# In[58]:


# For words inside of question and their synonyms:
K = 10
top_docs = list(tfidf_docs[stem_kw].sum(axis=1).nlargest(K).index.values)
# Sum up tf-idf scores for docs and return indices of top K documents


# In[59]:


# Create list of top K documents for answer analysis:
returned_doc_names = []
for doc in top_docs:
    returned_doc_names.append(doc_index_dict[doc]) #get all relevant document names (___.txt)

returned_docs = {}
for filename in returned_doc_names:
    returned_docs[filename] = doc_text_dict[filename]


# # Answer Analysis

# In[60]:


# Get all sentences from returned documents
sentences = []
for key,value in returned_docs.items():
    sentences += sent_tokenize(value)


# In[61]:


# Compute tf-idf scores for each sentence: treat sentences as documents

# Use count vectorizer to get matrix of terms as features and documents as indices for tfidf
cvec = CountVectorizer(vocabulary = stem_kw,tokenizer=textblob_tokenizer)
matrix = cvec.fit_transform(sentences)
cmatrix = pd.DataFrame(matrix.toarray(), columns=cvec.get_feature_names())

# Make Tf-idf vectorizer
tfidf_vec = TfidfVectorizer(tokenizer = textblob_tokenizer,
                      use_idf = False,norm='l2',
                      vocabulary = stem_kw) # L - TWO (cosine similarity)

# Say hey vectorizer, please read our stuff
matrix = tfidf_vec.fit_transform(sentences)

# And make a dataframe out of it
tfidf_results = pd.DataFrame(matrix.toarray(), columns=tfidf_vec.get_feature_names())


# In[62]:


# Compute scores for each sentence
#start: tfidf_results
score = tfidf_results
score['Sentence Keywords'] = 0

# Create vector to put into scoring dataframe
words_list = score['Sentence Keywords'].tolist()


# In[63]:


# remove months here, too prominent
for i in range(len(sentences)-1):
    # for each sentence
    words = [tuples[0] for tuples in pos_tag(word_tokenize(sentences[i]))
             if tuples[1][0:2] in ['NN','RB','JJ']] #remove VB
    for word in words:
        if word.lower() in stopwords.words('english'):
            words.remove(word)
    # stem words
    stem_words = []
    for word in words:
        stem_words += textblob_tokenizer(word)
    # assign stemmed sentence back into matrix
    words_list[i] = stem_words

score['Sentence Keywords'] = words_list


# In[81]:


# Calculate scores:
#1. Check for bigram matches +2 if matched (bigrams matching any pairs of words in question)
score['Score'] = 0

# Get question bigrams
def find_bigrams(input_list):
    bigram_list = []
    for i in range(len(input_list)-1):
        bigram_list.append((input_list[i], input_list[i+1]))
    return bigram_list
q_bigrams = find_bigrams(stem_kw)

# Turn score['Sentence Keywords'] into list of lists
sent_kw = score['Sentence Keywords'].tolist()
score_list = score['Score'].tolist()

#2. Scoring sentences
for i in range(0,len(score['Sentence Keywords'])-1):
    #for pair in [pairs for pairs in itertools.product(stem_kw, repeat=2)]: #all pairs of q_kws
        #if pair in find_bigrams(sent_kw[i]):
    if q_bigrams in find_bigrams(sent_kw[i]):
        score_list[i] += 2 
    for kw in stem_kw:
        if kw in sent_kw[i]:
            score_list[i] += score[kw].loc[i] #add tfidf score if in sentence
        if kw not in sent_kw[i]:
            score_list[i] -= tfidf_docs[kw].sum() #subtract tfidf score if word not contained
            
score['Sentence Keywords'] = sent_kw
score['Score'] = score_list


# In[82]:


# Calculate final answer scores:
# Sum up tf-idf scores for docs and return indices of top K documents
K = 10
top_sents = list(score['Score'].nlargest(K).index.values)

# Return top sentences as answer
answer_sents = [sentences[i] for i in top_sents]


# In[83]:


top_sents


# In[89]:


score.loc[772]


# ## Tag Retrieved Documents with Type:
# Person, Location, Quantity, etc.

# In[85]:


# Use nlp to tag entities in the answer sentences
#print([(X.text, X.label_) for X in answer.ents])

clean_sent_list = []
dumb_words = ['i','we','you','your','these','eyes','our']

# Remove stopwords in answer sentences -> output is list of list of words
for i in range(len(answer_sents)-1):
    clean_sent_list.append(word_tokenize(answer_sents[i]))
    for word in clean_sent_list[i]:
        if word.lower() in stopwords.words('english') or word.lower() in dumb_words:
            clean_sent_list[i].remove(word)
            
 # Stitch back together sentences from lists
for lists in clean_sent_list:
    i = 0
    sentence = ''
    for word in lists:
        sentence += word + ' '
    answer_sents[i] = sentence
    i += 1


# In[86]:


# Get list of ceos and companies and pickle them
ceos = pickle.load(open( "ceos_list.pickle","rb" ))
comps = pickle.load(open("comps_list.pickle","rb"))


# In[87]:


# Match answer type to question type and print answers
for answers in answer_sents:    
    if qtype == "NN":
        for chunk in nlp(answers).noun_chunks:
            print(chunk.text)
            print(answers, '\n')
    else:
        for ent in nlp(answers).ents:
            if ent.label_ == qtype:
                if qtype == 'PERSON':
                    if str(ent) in ceos:
                        print(ent)
                        print(answers,'\n')
                #elif qtype == 'ORG':
                 #   if str(ent) in comps:
                  #      print(ent)
                   #     print(answers,'\n')
                else: 
                    print(ent)
                    print(answers,'\n')


# In[ ]:




