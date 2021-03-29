from sklearn import svm
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from featx import bag_of_words, high_information_words
import nltk.classify
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier


# Train an svm classifier
# Hier kunnen we de SVM soort veranderen in:
# SVC, NuSVC, LinearSVC
def train_svm(feats):
    classifier = nltk.classify.SklearnClassifier(LinearSVC(max_iter=100)).train(feats)
    return classifier


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
            print(" |%-11s|%-11f|%-11f|%-11s|" % (category, precisions[category], recalls[category], "?"))
    print(" |-----------|-----------|-----------|-----------|")


def main():
    df = pd.read_csv('movies.csv')
    print(df['plot'][0:10])
    df['plot'] = df['plot'].str.lower()
    df = df.sample(frac=0.001, random_state=10)
    print(df)
    df['plot']= df['plot'].apply(str)
    feats = list ()
    for index, row in df.iterrows():
        #print(row['plot'], row['genre'])    
        tokens = word_tokenize(row['plot'])
        #print(tokens)
        bag = bag_of_words(tokens)
        #print(bag)
        feats.append((bag, row['genre']))
        
    train_feats, test_feats = split_train_test(feats)
    classifier = train_svm(train_feats)


if __name__ == "__main__":
    main()
