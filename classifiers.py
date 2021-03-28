from sklearn import svm
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Train an svm classifier
# Hier kunnen we de SVM soort veranderen in:
# SVC, NuSVC, LinearSVC
def train_svm(feature_samples, labels):
    classifier = svm.SVC()
    classifier.fit(feature_samples, labels)
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

def main():
    df_genre = pd.read_csv('data/title.basics.tsv',sep='\t')
    df_genre = df_genre[df_genre['titleType'] == 'movie']
    df_genre = df_genre[df_genre['genres'] != '\\N']
    print("reading crew")
    df_crew = pd.read_csv('data/title.crew.tsv',sep='\t')

    df_merged_full = pd.merge(left=df_genre, right=df_crew, left_on='tconst', right_on='tconst')

    # Random sample half of the data
    df_merged = df_merged_full.sample(frac=0.5)
    # Split genre string into array
    df_merged['genres'] = df_merged['genres'].apply(lambda x: x.split(','))
    # Create a row per genre
    df_merged = df_merged.explode('genres').fillna('')
    # Preview data to see if 1 genre per row
    print(df_merged.head(5))

    # Get one hot encoded sparse-matrix
    one_hot_encoder = OneHotEncoder(categories='auto')
    feature_matrix = one_hot_encoder.fit_transform(df_merged[['writers','directors']])
    
    # Get list of genres per observation
    labels = df_merged['genres'].tolist()    

    # Train SVM model
    train_svm(feature_matrix, labels)


if __name__ == "__main__":
    main()
