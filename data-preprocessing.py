#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 09:58:36 2022

@author: spyros
"""


import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn import datasets
from sklearn import svm
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.linear_model import LinearRegression

# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')

train_dataPath = ('~/Documents/Stevens/MachineLearning_695-A/Final_Project/feedback-prize-english-language-learning/train.csv')
test_dataPath = ('~/Documents/Stevens/MachineLearning_695-A/Final_Project/feedback-prize-english-language-learning/test.csv')

# Loads data into a dataframe.
def load_data(data):
    doc = pd.read_csv(data)
    return doc.copy(True)


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
        no_punctuation = np.char.replace(data, i, ' ')
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
def data_preprocessing(data):

    # # Loop through data and clean it.
    # for i in range(0,len(df)):
    #     text = df['full_text'][i]
    #     tokens = word_tokenize(text)
    #     # Make everything lowercase
    #     tokens = [w.lower() for w in tokens]
    #     # remove punctuation
    #     table = str.maketrans('', '', string.punctuation)
    #     stripped = [w.translate(table) for w in tokens]
    #     # remove remaining tokens that are not alphabetic
    #     words = [word for word in stripped if word.isalpha()]
    #     # remove stopwords
    #     stop_words = set(stopwords.words('english'))
    #     words = [w for w in words if not w in stop_words]
    #     # Lemmatize words
    #     lemmatizer = WordNetLemmatizer()
    #     newWords = []
    #     for i in words:
    #        lemmatizedWord = lemmatizer.lemmatize(i)
    #        newWords.append(lemmatizedWord)
      
    #     # combine array elements into a string
    #     words = ' '.join(words)
    #     # place string bach in dataset
    #     df = df.replace(df['full_text'][i], words)
    
    # return df
    df = data
    for i in data['full_text']:
      
        ndata = tokenize_data(i)
        ndata = lowercase_data(ndata)
        ndata = remove_punctuation(ndata)
        ndata = numbers_to_words(ndata)
        ndata = remove_stop_words(ndata)
        ndata = remove_punctuation(ndata)
        ndata = lemmatize_words(ndata)
        ndata = ' '.join(ndata)
        df = df.replace(i, ndata)
    return df
    
def tf_idf(data):
    tfidfVectorizer = TfidfVectorizer(use_idf=True)
    tfidf = tfidfVectorizer.fit_transform(data)
    df = pd.DataFrame(tfidf.toarray())
    return tfidf

def split_test_validation_byID(data):    
    train_d, val_d = train_test_split(data, test_size=0.2, random_state=1)
    return train_d, val_d


# Don't use kfold for now. Let's first see how it performs without it. 
# def implement_kFold(data):
#     kf = KFold(n_splits=5, shuffle=True, random_state=1)    
#     return kf.split(data)


def lin_reg(x,y):
   linreg = LinearRegression()
   
    

def model1_pipeline(data):
    dataf = load_data(data)
    train_data, validation_data = split_test_validation_byID(dataf)

    df = data_preprocessing(train_data)
    # X = tf_idf(df['full_text'])
    # Y = train_data['vocabulary']
    # print(X)
    
    return df.head()
    # return tfidf
    
# model1_pipeline(train_dataPath)

x = model1_pipeline(train_dataPath)
print(x)
