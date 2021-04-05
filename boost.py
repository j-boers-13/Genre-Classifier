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

# Train an svm classifier
# Hier kunnen we de SVM soort veranderen in:
# SVC, NuSVC, LinearSVC
def train_svm(feats):
    classifier = nltk.classify.SklearnClassifier(LinearSVC(max_iter=1000)).train(feats)
    return classifier


def train_svm2(features, labels):
  classifier = svm.LinearSVC()
  classifier.fit(features, labels)
  return classifier


# Preprocess and make features
def make_feats(df_2):
    feats = list ()
	# Make sure that overview contains strings
	# Lowercase the strings
    df_2['overview']= df_2['overview'].apply(str)
    df_2['overview'] = df_2['overview'].str.lower() 
    for index, row in df_2.iterrows():
        tokens = word_tokenize(row['overview'])
        # Remove punctuation
        tokens = [word for word in tokens if word.isalnum()] 
        bag = bag_of_words(tokens)
        feats.append((bag, row['genres']))
    return feats
    
    
# Maak features en labels als twee variabelen   
def make_feats2(df_2):
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



def split_train_test(feature_samples, split=0.8):
    """
    returns two feats
    
    splits a labelled dataset into two disjoint subsets train and test
    """
    train_feats = []
    test_feats = []

    random.Random(0).shuffle(feature_samples) # randomise dataset before splitting into train and test
    cutoff = int(len(feature_samples) * split)
    train_feats, test_feats = feature_samples[:cutoff], feature_samples[cutoff:]    

    print("\n##### Splitting datasets...")
    print("  Training set: %i" % len(train_feats))
    print("  Test set: %i" % len(test_feats))

    return train_feats, test_feats

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def main():
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    
    df = pd.read_csv('df_merged.csv')
    
    
    # Sample from entire data
    # Run with multiple genres per movie
    df_2 = df.sample(frac=1, random_state=10)
    

    # Single genre per movie
    df_2['genres'] = df_2['genres'].apply(lambda x: x.split(','))
    for index, row in df_2.iterrows():
       if len(row['genres']) != 1:
            df_2.drop(index, inplace=True)   

    # back to string
    df_2['genres'] = df_2['genres'].apply(lambda x: ', '.join([str(i) for i in x]))

    feats, labels = make_feats2(df_2)

    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, max_features = 5000) 
    bow_train = (vectorizer.fit_transform(feats)).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(bow_train, labels, test_size=0.2,random_state=10)
    classifierxd = train_svm2(X_train, y_train)
    #accuracy = nltk.classify.accuracy(classifier, X_test)
    y_pred = classifierxd.predict(X_test)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    






if __name__ == "__main__":
    main()
