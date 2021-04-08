from sklearn import svm
import pandas as pd
import numpy as np
import random
from nltk.tokenize import word_tokenize
from featx import bag_of_words, high_information_words
import nltk.classify
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from xgboost import XGBClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from classification import precision_recall
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from collections import Counter
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import seaborn as sn

    
# Preprocess the overview, remove non-letters, convert to lower case,
# remove stopwords, return space joined texts and genres as labels
# Make sure that overview contains strings
def make_feats(df_2):
    features = list ()
    labels = list()
    df_2['overview']= df_2['overview'].apply(str)
    df_2['overview'] = df_2['overview'].str.lower() 
    for index, row in df_2.iterrows():
        text = row['overview']
        text = re.sub("[^a-zA-Z]", " ", text)
        text = text.lower().split()
        stop_word_list = set(stopwords.words("english"))  
        text = [word for word in text if not word in stop_word_list]
        features.append((" ".join(text)))
        labels.append(row['genres'])
    return features, labels


# Make bag of words as a vector
def make_Bags(features):
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, max_features = 5000) 
    bag = (vectorizer.fit_transform(features)).toarray() 
    return bag


# Train an svm classifier
def train_svm(X_train, y_train):
  classifier = svm.SVC(kernel='linear')
  classifier.fit(X_train, y_train)
  return classifier


# Train gradient boosting classifier (XGB)
def train_xgb(X_train, y_train):
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return classifier


# Get unique genres
def uniqueGenres(df_2):
    genres = []
    all_genres = df_2['genres'].tolist()
    [genres.append(x) for x in all_genres if x not in genres]
    return genres


def main():
    df = pd.read_csv('df_single.csv')
    start_time = time.time()

    # for sampling small set of data to see if works.
    df_2 = df.sample(frac=1, random_state=10)
    features, labels = make_feats(df_2)
    bag_of_words = make_Bags(features)
    X_train, X_test, y_train, y_test = train_test_split(bag_of_words, labels, test_size=0.1,random_state=1)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.125, random_state=1)    
    #genres = uniqueGenres(df_2)
    
    # SVM
    svm_classifier = svm_train(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    svm_cm = confusion_matrix(y_test, y_pred) 
    unique_labels = np.unique(list(y_test) + list(y_pred))
    df_svm_cm = pd.DataFrame(svm_cm, index = unique_labels,
                 columns = unique_labels)
    sn.heatmap(df_svm_cm, annot=True, fmt='g')
    plt.show()
    with open('svm_svc_linear.pickle', 'wb') as f:
        pickle.dump(svm_classifier, f)   
        
       
    # GBM 
    gbm_classifier =  train_xgb(X_train_y_train)
    y_pred = gbm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    svm_cm = confusion_matrix(y_test, y_pred)
    
    unique_labels = np.unique(list(y_test) + list(y_pred))
    df_svm_cm = pd.DataFrame(svm_cm, index = unique_labels,
                 columns = unique_labels)
    sn.heatmap(df_svm_cm, annot=True, fmt='g')
    plt.show()
    print(accuracy)
    with open('xgb_default.pickle', 'wb') as f:
        pickle.dump(gbm_classifier, f)   
    

    print("--- %s seconds ---" % (time.time() - start_time)) 
if __name__ == "__main__":
    main()

