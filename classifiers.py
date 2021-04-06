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


# Preprocess and make features bag of words, old from assignment
# Make sure that overview contains strings
# Lowercase the strings
# Remove punctuation
def make_feats(df_2):
    feats = list ()
    df_2['overview']= df_2['overview'].apply(str)
    df_2['overview'] = df_2['overview'].str.lower() 
    for index, row in df_2.iterrows():
        tokens = word_tokenize(row['overview'])
        tokens = [word for word in tokens if word.isalnum()] 
        bag = bag_of_words(tokens)
        feats.append((bag, row['genres']))
    return feats
    
    
# Preprocess the overview, remove non-letters, convert to lower case,
# remove stopwords, return space joined texts and genres as labels
# Make sure that overview contains strings
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


# Make bag of words as a vector
def makeBags(features):
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, max_features = 5000) 
    bag = (vectorizer.fit_transform(features)).toarray() 
    return bag


# Train an svm classifier
# Hier kunnen we de SVM soort veranderen in:
# SVC, NuSVC, LinearSVC
def train_svm(feats):
    classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(feats)
    return classifier


# Train an svm classifier for bag of words as vectors
# Hier kunnen we de SVM soort veranderen in:
# SVC, NuSVC, LinearSVC
def train_svm2(X_train, y_train):
  classifier = svm.LinearSVC()
  classifier.fit(X_train, y_train)
  return classifier


# Train gradient boosting classifier
# XGB classifier
def train_xgb(X_train, y_train):
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return classifier

# Parameters XGB
#BOW_XGB = xgb.XGBClassifier(max_depth=7, n_estimators=300, objective="binary:logistic", random_state=1, tree_method='gpu_hist', predictor='gpu_predictor')
#BOW_XGB_scores = cross_val_score(BOW_XGB, bow_train, train.sentiment, cv=3, n_jobs=-1)
#print("Averaged CV Accuracy: %0.2f (+/- %0.2f)" % (BOW_XGB_scores.mean(), BOW_XGB_scores.std() * 2))


# Get unique genres
def uniqueGenres(df_2):
    genres = []
    all_genres = df_2['genres'].tolist()
    [genres.append(x) for x in all_genres if x not in genres]
    return genres


def evaluation(classifier, test_feats, categories):
    """
    calculates and prints evaluation measures
    """
    print ("\n##### Evaluation...")
    print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))
    precisions, recalls = precision_recall(classifier, test_feats)
    f_measures = calculate_f(precisions, recalls)  
    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11s|%-11s|%-11s|" % ("category","precision","recall","F-measure"))
    print(" |-----------|-----------|-----------|-----------|")
    for category in categories:
        if precisions[category] is None:
            print(" |%-11s|%-11s|%-11s|%-11s|" % (category, "NA", "NA", "NA"))
        else:
            try:
                print(" |%-11s|%-11f|%-11f|%-11s|" % (category, precisions[category], recalls[category],f_measures[category]
                ))
            except ZeroDivisionError:
                pass
    print(" |-----------|-----------|-----------|-----------|")
    print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))
    
    
def calculate_f(precisions, recalls):
    f_measures = {}
    for category in precisions:
        try: 
            f_measures[category] = round(((2 * precisions[category] * recalls[category]) / (precisions[category] + recalls[category])), 6)
        except TypeError:
            f_measures[category] = "NA"
        except ZeroDivisionError:
            pass
    return f_measures


# Dummy classifier gemaakt voor presentatie
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
    #freq = []
    #score = 0
    #for i in train_feats:
   ##    freq.append(i[1])
    #for i in test_feats:
    #    if i[1] == 'Drama':
    #        score = score + 1
    #print(score)


def main():
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    
    df = pd.read_csv('df_single.csv')
    df_2 = df.sample(frac=1, random_state=10)
    print(df_2)
    
    
    # Dit is manier van opdracht, accuracy: 0.6642335766423357
    # Evaluation met keyError
    feats = make_feats(df_2)
    train_feats, test_feats = train_test_split(feats, test_size=0.1,random_state=1)
    train_feats, dev_feats = train_test_split(train_feats, test_size=0.125, random_state=1) 
    svm_classifier = train_svm(train_feats)
    accuracy = nltk.classify.accuracy(svm_classifier, test_feats)
    print(accuracy)
    genres = uniqueGenres(df_2)
    #evaluation(svm_classifier, test_feats, genres)


    # Dit is bag of words als vectors, accuracy: 0.602711157455683
    # evaluation is hier ook rip 
    features, labels = make_feats2(df_2)
    bag_of_words = makeBags(features)
    X_train, X_test, y_train, y_test = train_test_split(bag_of_words, labels, test_size=0.1,random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)    
    genres = uniqueGenres(df_2)
    print(len(genres))
    svm_classifier = train_svm2(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #evaluation(svm_classifier, X_test, y_test, genres)
    #analysis(classifier)    
        
    # GBM met vectors, accuracy: 0.6527632950990615
    gbm_classifier = train_xgb(X_train, y_train)
    y_pred = gbm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
  
    


if __name__ == "__main__":
    main()
