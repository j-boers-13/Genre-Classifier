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
    pd.set_option('display.max_columns', None)

    df_genre = pd.read_csv('data/title.basics.tsv',sep='\t')
    df_genre = df_genre[df_genre['titleType'] == 'movie']
    df_genre = df_genre[df_genre['genres'] != '\\N']
    print("reading crew")
    df_crew = pd.read_csv('data/title.crew.tsv',sep='\t')

    df_merged_full = pd.merge(left=df_genre, right=df_crew, left_on='tconst', right_on='tconst')

    # Random sample half of the data
    df_merged = df_merged_full.sample(frac=0.5)

    # Split genres, writers and directors strings into array
    df_merged['genres'] = df_merged['genres'].apply(lambda x: x.split(','))
    df_merged['writers'] = df_merged['writers'].apply(lambda x: x.split(','))
    df_merged['directors'] = df_merged['directors'].apply(lambda x: x.split(','))

    # Create directors df
    df_directors = pd.DataFrame([pd.Series(x) for x in df_merged.directors])
    df_directors.columns = ['director_{}'.format(x+1) for x in df_directors.columns]

    # Create writers df
    df_writers = pd.DataFrame([pd.Series(x) for x in df_merged.writers])
    df_writers.columns = ['writer_{}'.format(x+1) for x in df_writers.columns]

    # Create a duplicate row per genre
    df_merged = df_merged.explode('genres').fillna('')
    df_merged = df_merged.reset_index()
    df_directors = df_directors.reset_index()
    df_writers = df_writers.reset_index()
    df_merged = pd.concat([df_merged, df_directors], axis=1)
    df_merged = pd.concat([df_merged, df_writers], axis=1)
    
    # Get one hot encoded sparse-matrix
    one_hot_encoder = OneHotEncoder(categories='auto')
    column_names = [col for col in df_merged.columns if 'director' in col].extend([col for col in df_merged.columns if 'writer' in col])

    feature_matrix = one_hot_encoder.fit_transform(df_merged[column_names])
    
    # Get list of genres per observation
    labels = df_merged['genres'].tolist()    

    # Train SVM model
    train_svm(feature_matrix, labels)


if __name__ == "__main__":
    main()
