#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 09:58:36 2022

"""


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# In case you need to download nltk.
# import nltk
# nltk.download('english')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

#
train_data_path = ('./feedback-prize-english-language-learning/train.csv')
test_data_path = ('./feedback-prize-english-language-learning/test.csv')


# Loads data into a dataframe.
def load_data(path):
    corpus = pd.read_csv(path)
    return corpus

def tokenize_data(data):
    tokens = word_tokenize(data)
    return tokens

def lowercase_data(data):
    lowercase = np.char.lower(data)
    return lowercase

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-.,/:;<=>?@[\]^_`{|}~\n"
    no_punctuation = data
    for i in symbols:
        no_punctuation = np.char.replace(no_punctuation, i, ' ')
        no_punctuation = np.char.replace(no_punctuation, "'", "")        
        
    return no_punctuation

def numbers_to_words(data):
    newData = []
    for i in data:
        if i != str(i):
            num = num2words(i)
            newData.append(num)
        else:
            newData.append(i)
    return newData

def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    words = [w for w in data if not w in stop_words]
    return words

def lemmatize_words(data):
    a = []
    lemmatizer = WordNetLemmatizer()
    for i in data:
        lemmatized_word = lemmatizer.lemmatize(i)
        a.append(lemmatized_word)
    return a


# This function removes special characters and stop words.
# Also, it makes all words lowercase.
def data_cleanning(data):
    df = data
    # Loop through data and clean it.
    for i in data['full_text']:
       
        ndata = tokenize_data(i)
        ndata = remove_punctuation(ndata)
        ndata = lowercase_data(ndata)
        ndata = numbers_to_words(ndata)
        ndata = remove_stop_words(ndata)
        ndata = lemmatize_words(ndata)
        ndata = numbers_to_words(ndata)
        ndata = ' '.join(ndata)
        ndata = remove_punctuation(ndata)
        
        df = df.replace(i, ndata)

    return pd.DataFrame(df)

def split_test_validation(data):    
    train_d, val_d = train_test_split(data, test_size=0.2, random_state=1)
    train_d = pd.DataFrame(train_d)
    val_d = pd.DataFrame(val_d)
    return train_d, val_d
    
def tf_idf(data):
    
    vectorizer = TfidfVectorizer(max_features=100,
                                 min_df=5,
                                 max_df=0.8,
                                 ngram_range=(1,3))
    vectors = vectorizer.fit_transform(data)
    dense = vectors.todense()
    denselist = dense.tolist()
    feature_names = vectorizer.get_feature_names_out()
    
    all_keywords = []
    for i in denselist:
        x = 0
        keywords = []
        for word in i:
            if word > 0:
                keywords.append(feature_names[x])
            x=x+1
        all_keywords.append(keywords)
    
            
    
    return vectors, denselist, feature_names, all_keywords


def tfidf_pipeline(data):
    dataf = load_data(data)

    train_data, validation_data = split_test_validation(dataf)

    clean_data = data_cleanning(train_data)
    vectors, denselist, feature_names, all_keywords = tf_idf(clean_data['full_text'])
    y = train_data[['vocabulary', 'syntax']]
    
    # Convert vectoried data with keywords into a dataframe
    data=[]
    for i in range(len(all_keywords)):
        data.insert(i, {'words':all_keywords[i],'vectors':denselist[i],})
    dataframe = pd.DataFrame(data)

    return dataframe, y, clean_data, vectors, denselist, feature_names, all_keywords


dataframe, y, clean_data, vectors, denselist, feature_names, all_keywords = tfidf_pipeline(train_data_path)
print(dataframe)
dataframe.to_csv('./tfidfData.csv')


