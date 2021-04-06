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
    print("### Loading datasets")
    df_genre = pd.read_csv('data/title.basics.tsv',sep='\t')
    df_crew = pd.read_csv('data/title.crew.tsv',sep='\t')
    df_descriptions = pd.read_csv('data/movies_metadata.csv')

    # Get only the movies from genre dataset
    df_genre = df_genre[df_genre['titleType'] == 'movie']

    # Remove movies without a genre
    df_genre = df_genre[df_genre['genres'] != '\\N']
    
    # Merge genre data and crew data
    df_merged = pd.merge(left=df_genre, right=df_crew, left_on='tconst', right_on='tconst')
    df_merged = pd.merge(left = df_merged,
                right = df_descriptions[['imdb_id','overview']],
                left_on='tconst', right_on='imdb_id')
    print(len(df_merged))
    print(df_merged.head(5))
    df_merged.to_csv('df_merged.csv', encoding='utf-8', index=False)
    
    df_merged['genres'] = df_merged['genres'].apply(lambda x: x.split(','))
    for index, row in df_merged.iterrows():
       if len(row['genres']) != 1:
            df_merged.drop(index, inplace=True)
    print(df_merged)
    df_merged['genres'] = df_merged['genres'].apply(lambda x: ', '.join([str(i) for i in x]))
    df_merged.to_csv('df_single.csv', encoding='utf-8', index=False)     

if __name__ == "__main__":
    main()


