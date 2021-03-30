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
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from nltk.classify.scikitlearn import SklearnClassifier
from classification import precision_recall

# Train an svm classifier
# Hier kunnen we de SVM soort veranderen in:
# SVC, NuSVC, LinearSVC
def train_svm(feats):
    classifier = nltk.classify.SklearnClassifier(SVC(max_iter=100)).train(feats)
    return classifier

# dit werkt niet met tuple en label
# def train_svm(feats):
#  classifier = svm.SVC()
#  classifier.fit(feats)
#  return classifier


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


def calculate_f(precisions, recalls):
    f_measures = {}
    for category in precisions:
        try: 
            f_measures[category] = round(((2 * precisions[category] * recalls[category]) / (precisions[category] + recalls[category])), 6)
        except TypeError:
            f_measures[category] = "NA"
    return f_measures


def main():
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    
    df = pd.read_csv('df_merged.csv')
    #print(df['overview'][0:10])
    df_2 = df.sample(frac=0.001, random_state=10)
    #print(df_2)
    feats = make_feats(df_2)
    genres = df_2['genres'].tolist()
    res = []
    [res.append(x) for x in genres if x not in res]
    print(res)
        
    train_feats, test_feats = split_train_test(feats)    
    classifier = train_svm(feats)
    print(classifier)
    #evaluation(classifier, test_feats, genres)
    #analysis(classifier)


if __name__ == "__main__":
    main()
